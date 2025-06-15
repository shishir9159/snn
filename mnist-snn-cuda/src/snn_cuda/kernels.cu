// Winner-take-all STDP layer. One block presents whole images; compile with
// --fmad=false so float sums round like the Rust model.
//
// Training uses a single block that loops over a chunk of images (weights change
// image to image). Evaluation leaves weights alone, so blocks take images in
// parallel. Regular coding is closed form: input i spikes at t iff t % period[i] == 0.
//
// Per image: build the whole spike schedule, then walk time in slices. Within a
// slice the drive of every (step, neuron) pair is summed in parallel up front;
// the serial step loop only integrates from shared memory and meets one barrier.
// A neuron spiking mid-slice changes its own weights, so its later drives are redone.

#define MAXOUT 1024

__constant__ float KERNEL[5][5] = {
    {-0.5f, -0.125f, 0.25f, -0.125f, -0.5f},
    {-0.125f, 0.25f, 0.625f, 0.25f, -0.125f},
    {0.25f, 0.625f, 1.0f, 0.625f, 0.25f},
    {-0.125f, 0.25f, 0.625f, 0.25f, -0.125f},
    {-0.5f, -0.125f, 0.25f, -0.125f, -0.5f}};

// fcfg slots
enum { REST, FLOOR, LEAK, POT_MIN, POT_MAX, FREQ_MIN, FREQ_MAX, PERIOD_SCALE, SIGMA,
       W_MIN, W_MAX, THRESHOLD_DIV, SILENT_PENALTY, W_NORM, THETA_RATE, TABLE = 16 };
// icfg slots
enum { N_IN, N_OUT, WIDTH, HEIGHT, STEPS, T_REF, T_BACK, T_FORE, SLICE, MAX_SPIKES };

__device__ __forceinline__ float interp(float v, float in_lo, float in_hi, float out_lo, float out_hi) {
    if (in_hi <= in_lo) return out_lo;
    float r = fminf(fmaxf((v - in_lo) / (in_hi - in_lo), 0.0f), 1.0f);
    return out_lo + r * (out_hi - out_lo);
}

// spike period in steps, 0 when the input never spikes within the image
__device__ __forceinline__ int period(float pot, const float* f, int steps) {
    if (!(pot > 0.0f)) return 0;
    float freq = interp(pot, f[POT_MIN], f[POT_MAX], f[FREQ_MIN], f[FREQ_MAX]);
    float per = fmaxf(ceilf(f[PERIOD_SCALE] / freq), 1.0f);
    return per <= (float)steps ? (int)per : 0;
}

__device__ __forceinline__ bool better(float am, int aj, float bm, int bj) {
    return am > bm || (am == bm && aj > bj);
}

// sum of neuron j's weights from the inputs spiking at step s, ascending, as the CPU model does
__device__ __forceinline__ float drive(const float* w, const unsigned short* act,
                                       const int* st, int s, int n_out, int j) {
    float d = 0.0f;
    for (int a = st[s]; a < st[s + 1]; a++) d += w[(size_t)act[a] * n_out + j];
    return d;
}

// rec_*: per-step record of one presentation for the live view (null when unused)
extern "C" __global__ void present(
    const unsigned char* pixels, const int* order, int n_img, int learn,
    const float* fcfg, const int* icfg,
    float* w,      // input-major: w[i * n_out + j]
    float* theta,  // homeostatic threshold scale per neuron
    int* winners, unsigned short* spikes, float* bases,
    unsigned char* rec_act, float* rec_pot, unsigned char* rec_fired)
{
    extern __shared__ int sm[];
    const int n_in = icfg[N_IN], n_out = icfg[N_OUT], W = icfg[WIDTH], H = icfg[HEIGHT];
    const int steps = icfg[STEPS], t_ref = icfg[T_REF], tb = icfg[T_BACK], tf = icfg[T_FORE];
    const int slice = icfg[SLICE], max_spikes = icfg[MAX_SPIKES];
    const int tid = threadIdx.x, B = blockDim.x, lane = tid & 31, warp = tid >> 5;
    const int nwarps = (B + 31) >> 5;
    int* per = sm;                                    // n_in, 0 = silent
    int* lit = per + n_in;                            // inputs that spike, ascending
    int* st = lit + n_in;                             // steps + 2, start of each step in act
    float* col = (float*)(st + steps + 2);            // n_in, the winner's weights
    float* drv = col + n_in;                          // slice * n_out
    unsigned short* act = (unsigned short*)(drv + slice * n_out);  // max_spikes
    __shared__ int wc[32], wi[32], s_winner, s_nfired, s_max, fl[MAXOUT];
    __shared__ float wm[32], s_len;

    const float rest = fcfg[REST], floor_ = fcfg[FLOOR], leak = fcfg[LEAK];
    const int run = (n_in + B - 1) / B;

    for (int k = blockIdx.x; k < n_img; k += gridDim.x) {
        const int img = order ? order[k] : k;
        const unsigned char* px = pixels + (size_t)img * n_in;

        for (int s = tid; s < steps + 2; s += B) st[s] = 0;
        __syncthreads();

        // receptive field, then spike periods and per-step spike counts
        for (int i = tid; i < n_in; i += B) {
            const int y = i / W, x = i % W;
            float pot = 0.0f;
            for (int ky = 0; ky < 5; ky++) {
                const int sy = y - 2 + ky;
                if (sy < 0 || sy >= H) continue;
                for (int kx = 0; kx < 5; kx++) {
                    const int sx = x - 2 + kx;
                    if (sx < 0 || sx >= W) continue;
                    const unsigned char p = px[sy * W + sx];
                    if (p) pot += KERNEL[ky][kx] * ((float)p / 255.0f);
                }
            }
            const int pr = period(pot, fcfg, steps);
            per[i] = pr;
            if (pr) for (int t = pr; t <= steps; t += pr) atomicAdd(&st[t], 1);
        }
        __syncthreads();

        // peak simultaneous spikes, and counts -> start offsets (one warp, lanes own runs)
        if (warp == 0) {
            const int n = steps + 2, per_lane = (n + 31) / 32, lo = lane * per_lane;
            const int hi = min(lo + per_lane, n);
            int sum = 0, mx = 0;
            for (int s = lo; s < hi; s++) { sum += st[s]; mx = max(mx, st[s]); }
            int scan = sum;
            for (int o = 1; o < 32; o <<= 1) {
                int v = __shfl_up_sync(0xffffffffu, scan, o);
                if (lane >= o) scan += v;
            }
            for (int o = 16; o > 0; o >>= 1) mx = max(mx, __shfl_down_sync(0xffffffffu, mx, o));
            int acc = scan - sum;
            for (int s = lo; s < hi; s++) { int c = st[s]; st[s] = acc; acc += c; }
            if (lane == 0) s_max = mx;
        }

        // inputs that spike at all, ascending: each thread owns a run of inputs
        int c = 0;
        for (int m = 0; m < run && tid * run + m < n_in; m++) c += per[tid * run + m] > 0;
        int scan = c;
        for (int o = 1; o < 32; o <<= 1) {
            int v = __shfl_up_sync(0xffffffffu, scan, o);
            if (lane >= o) scan += v;
        }
        if (lane == 31) wc[warp] = scan;
        __syncthreads();
        int off = scan - c, nlit = 0;
        for (int q = 0; q < nwarps; q++) {
            if (q < warp) off += wc[q];
            nlit += wc[q];
        }
        for (int m = 0; m < run && tid * run + m < n_in; m++)
            if (per[tid * run + m] > 0) lit[off++] = tid * run + m;
        if (tid == 0) { s_winner = -1; s_nfired = 0; }
        __syncthreads();

        // each step collects its spiking inputs, ascending
        for (int t = 1 + tid; t <= steps; t += B) {
            int pos = st[t];
            for (int l = 0; l < nlit; l++) {
                const int i = lit[l];
                if (t % per[i] == 0 && pos < max_spikes) act[pos++] = (unsigned short)i;
            }
        }
        __syncthreads();

        if (rec_act)
            for (int t = 1 + tid; t <= steps; t += B)
                for (int a = st[t]; a < st[t + 1]; a++) rec_act[(size_t)(t - 1) * n_in + act[a]] = 1;

        // a blank image gives no drive; keep the threshold above rest
        const float base = fmaxf((float)s_max / fcfg[THRESHOLD_DIV], 1.0f);
        const float thr = tid < n_out ? base * (1.0f + theta[tid]) : 0.0f;

        float p = rest;
        int t_rest = -1, nspk = 0;

        for (int c0 = 0; c0 < steps; c0 += slice) {
            const int tca = min(slice, steps - c0);
            for (int idx = tid; idx < tca * n_out; idx += B) {
                const int tt = idx / n_out;
                drv[idx] = drive(w, act, st, c0 + 1 + tt, n_out, idx - tt * n_out);
            }
            __syncthreads();

            for (int tt = 0; tt < tca; tt++) {
                const int t = c0 + 1 + tt;
                if (tid < n_out && !(t <= t_rest)) {
                    p += drv[tt * n_out + tid];
                    if (p > rest) p = fmaxf(p - leak, rest);
                    if (p < floor_) p = floor_;
                }
                bool fired = false;
                // one barrier per step; only a neuron at threshold needs the rest
                if (__syncthreads_or(tid < n_out && p >= thr)) {
                    // first neuron furthest past its threshold wins and inhibits the rest
                    if (s_winner < 0) {
                        float m = -3.0e38f;
                        int j = tid;
                        if (tid < n_out && !(t <= t_rest)) m = p - thr;
                        for (int o = 16; o > 0; o >>= 1) {
                            float om = __shfl_down_sync(0xffffffffu, m, o);
                            int oj = __shfl_down_sync(0xffffffffu, j, o);
                            if (better(om, oj, m, j)) { m = om; j = oj; }
                        }
                        if (lane == 0) { wm[warp] = m; wi[warp] = j; }
                        __syncthreads();
                        if (tid == 0) {
                            float bm = wm[0]; int bj = wi[0];
                            for (int q = 1; q < nwarps; q++)
                                if (better(wm[q], wi[q], bm, bj)) { bm = wm[q]; bj = wi[q]; }
                            s_winner = bm >= 0.0f ? bj : -1;
                        }
                        __syncthreads();
                        if (s_winner >= 0 && tid < n_out && tid != s_winner) p = floor_;
                    }
                    if (tid < n_out && p >= thr) {
                        fired = true;
                        p = rest;
                        t_rest = t + t_ref;
                        nspk++;
                        if (learn) fl[atomicAdd(&s_nfired, 1)] = tid;
                    }
                    __syncthreads();

                    const int nf = s_nfired;
                    if (learn && nf > 0) {
                        const int lo_s = max(t - tb, 1), hi_s = min(t + tf, steps);
                        for (int f = 0; f < nf; f++) {
                            const int j = fl[f];
                            for (int l = tid; l < nlit; l += B) {
                                const int i = lit[l], pr = per[i];
                                float wt = w[(size_t)i * n_out + j];
                                for (int s = ((lo_s + pr - 1) / pr) * pr; s <= hi_s; s += pr) {
                                    const float dw = fcfg[TABLE + s + tb - t];
                                    wt = dw > 0.0f ? wt + fcfg[SIGMA] * dw * (fcfg[W_MAX] - wt)
                                                   : wt + fcfg[SIGMA] * dw * (wt - fcfg[W_MIN]);
                                }
                                w[(size_t)i * n_out + j] = wt;
                            }
                        }
                        __syncthreads();
                        // the neuron's later drives in this slice used the old weights
                        for (int f = 0; f < nf; f++)
                            for (int u = tt + 1 + tid; u < tca; u += B)
                                drv[u * n_out + fl[f]] = drive(w, act, st, c0 + 1 + u, n_out, fl[f]);
                        if (tid == 0) s_nfired = 0;
                        __syncthreads();
                    }
                }
                if (rec_pot && tid < n_out) {
                    rec_pot[(size_t)(t - 1) * n_out + tid] = p;
                    rec_fired[(size_t)(t - 1) * n_out + tid] = fired;
                }
            }
            __syncthreads();
        }

        if (tid < n_out) spikes[(size_t)k * n_out + tid] = (unsigned short)nspk;
        if (tid == 0) { winners[k] = s_winner; bases[k] = base; }

        if (learn) {
            const int win = s_winner;
            if (win >= 0) {
                // depress synapses from inputs that stayed silent, then rescale the row
                for (int i = tid; i < n_in; i += B) {
                    float wt = w[(size_t)i * n_out + win];
                    if (!per[i]) wt = fmaxf(wt - fcfg[SILENT_PENALTY], fcfg[W_MIN]);
                    col[i] = wt;
                }
                __syncthreads();
                if (tid == 0) {
                    float len = 0.0f;
                    for (int i = 0; i < n_in; i++) len += col[i] * col[i];
                    s_len = sqrtf(len);
                }
                __syncthreads();
                const float norm = fcfg[W_NORM];
                const float len = s_len;
                for (int i = tid; i < n_in; i += B) {
                    float wt = col[i];
                    if (norm > 0.0f && len > 0.0f) wt = fminf(wt * (norm / len), fcfg[W_MAX]);
                    w[(size_t)i * n_out + win] = wt;
                }
            }
            // drive every neuron toward an equal share of wins
            if (tid < n_out) {
                const float share = 1.0f / (float)n_out;
                const float won = (win == tid) ? 1.0f : 0.0f;
                theta[tid] = fmaxf(theta[tid] + fcfg[THETA_RATE] * (won - share), -0.9f);
            }
        }
        __syncthreads();
    }
}
