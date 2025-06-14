// Winner-take-all STDP layer. One block presents whole images; compile with
// --fmad=false so float sums round like the Rust model.
//
// Training uses a single block that loops over a chunk of images (weights change
// image to image). Evaluation leaves weights alone, so blocks take images in
// parallel. Regular coding is closed form: input i spikes at t iff t % period[i] == 0.

#define MAXOUT 1024
#define MAXRUN 8

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
enum { N_IN, N_OUT, WIDTH, HEIGHT, STEPS, T_REF, T_BACK, T_FORE };

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
    const int tid = threadIdx.x, B = blockDim.x, lane = tid & 31, warp = tid >> 5;
    const int nwarps = (B + 31) >> 5;
    int* per = sm;                 // n_in
    int* act = per + n_in;         // n_in
    int* cnt = act + n_in;         // steps + 2
    float* col = (float*)(cnt + steps + 2);  // n_in, the winner's weights
    __shared__ int wc[32], wi[32], s_winner, s_nfired, s_max, fl[MAXOUT];
    __shared__ float wm[32], s_len;

    const float rest = fcfg[REST], floor_ = fcfg[FLOOR], leak = fcfg[LEAK];
    const int per_thread = (n_in + B - 1) / B;

    for (int k = blockIdx.x; k < n_img; k += gridDim.x) {
        const int img = order ? order[k] : k;
        const unsigned char* px = pixels + (size_t)img * n_in;

        for (int s = tid; s < steps + 2; s += B) cnt[s] = 0;
        if (tid == 0) s_max = 0;
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
            if (pr) for (int t = pr; t <= steps; t += pr) atomicAdd(&cnt[t], 1);
        }
        __syncthreads();
        for (int t = 1 + tid; t <= steps; t += B) atomicMax(&s_max, cnt[t]);
        __syncthreads();
        // a blank image gives no drive; keep the threshold above rest
        const float base = fmaxf((float)s_max / fcfg[THRESHOLD_DIV], 1.0f);
        const float thr = tid < n_out ? base * (1.0f + theta[tid]) : 0.0f;

        float p = rest;
        int t_rest = -1, nspk = 0;
        if (tid == 0) s_winner = -1;
        __syncthreads();

        for (int t = 1; t <= steps; t++) {
            // compact the inputs spiking now, ascending: each thread owns a run of inputs
            int mine[MAXRUN], c = 0;  // MAXRUN covers n_in <= MAXRUN * blockDim
            const int lo = tid * per_thread;
            for (int m = 0; m < per_thread && lo + m < n_in; m++) {
                const int i = lo + m, pr = per[i];
                if (pr && t % pr == 0) mine[c++] = i;
            }
            int scan = c;
            for (int o = 1; o < 32; o <<= 1) {
                int v = __shfl_up_sync(0xffffffffu, scan, o);
                if (lane >= o) scan += v;
            }
            if (lane == 31) wc[warp] = scan;
            if (tid == 0) s_nfired = 0;
            __syncthreads();
            int off = scan - c, nact = 0;
            for (int q = 0; q < nwarps; q++) {
                if (q < warp) off += wc[q];
                nact += wc[q];
            }
            for (int m = 0; m < c; m++) {
                act[off + m] = mine[m];
                if (rec_act) rec_act[(size_t)(t - 1) * n_in + mine[m]] = 1;
            }
            __syncthreads();

            // integrate: sum the active columns in input order, as the CPU model does
            if (tid < n_out) {
                float drive = 0.0f;
                for (int a = 0; a < nact; a++) drive += w[(size_t)act[a] * n_out + tid];
                if (!(t <= t_rest)) {
                    p += drive;
                    if (p > rest) p = fmaxf(p - leak, rest);
                    if (p < floor_) p = floor_;
                }
            }

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
                p = rest;
                t_rest = t + t_ref;
                nspk++;
                fl[atomicAdd(&s_nfired, 1)] = tid;
            }
            if (rec_pot && tid < n_out) {
                rec_pot[(size_t)(t - 1) * n_out + tid] = p;
                rec_fired[(size_t)(t - 1) * n_out + tid] = (t_rest == t + t_ref);
            }
            __syncthreads();

            if (learn && s_nfired > 0) {
                const int lo_s = max(t - tb, 1), hi_s = min(t + tf, steps);
                for (int f = 0; f < s_nfired; f++) {
                    const int j = fl[f];
                    for (int i = tid; i < n_in; i += B) {
                        const int pr = per[i];
                        if (!pr) continue;
                        float wt = w[(size_t)i * n_out + j];
                        for (int s = ((lo_s + pr - 1) / pr) * pr; s <= hi_s; s += pr) {
                            const float dw = fcfg[TABLE + s + tb - t];
                            wt = dw > 0.0f ? wt + fcfg[SIGMA] * dw * (fcfg[W_MAX] - wt)
                                           : wt + fcfg[SIGMA] * dw * (wt - fcfg[W_MIN]);
                        }
                        w[(size_t)i * n_out + j] = wt;
                    }
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
