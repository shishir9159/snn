// Winner-take-all STDP layer, one neuron per thread. Compile with --fmad=false so
// float sums round like the Rust model.
//
// Neurons only meet at the first winner: until some neuron reaches threshold each
// runs on its own, then every other neuron is pushed to the floor, and from there on
// each neuron (and the STDP on its own weights) is independent again.
//
// Global-memory latency, not arithmetic, bounds a lone thread. So before the winner
// each neuron gathers a slice of steps' weights with many loads in flight; the winner
// copies its weight column to shared memory and all neuron threads help it (drives
// of the remaining steps, STDP per input, normalisation); neurons that cannot climb
// back from the floor are skipped. The input spike schedule depends on the image only,
// so spare "producer" warps build the next image's schedule meanwhile.
// Regular coding is closed form: input i spikes at t iff t % period[i] == 0.
//
// Training uses a single block that loops over a chunk of images (weights change
// image to image). Evaluation leaves weights alone, so blocks take images in parallel.

#define NBAR_C 1  // named barriers: neuron threads, producer threads
#define NBAR_P 2
#define G 16      // weights in flight per STDP batch
#define MAX_THREADS 384  // launch bound, caps registers; keep equal to layer.MAX_THREADS

// -DPROFILE: the winner thread and producer 0 of block 0 add the cycles they spend per image
// to prof[]: 0 slices before the first spike, 1 winner barriers, 7 the winner's rounds,
// 2 the other neurons, 3 post-image, 4 waiting for the rest, 5 schedule build, 6 producer wait
#ifdef PROFILE
__device__ unsigned long long prof[8];
#define MARK(slot) { long long n_ = clock64(); pc[slot] += n_ - mark; mark = n_; }
#else
#define MARK(slot)
#endif

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
enum { N_IN, N_OUT, WIDTH, HEIGHT, STEPS, T_REF, T_BACK, T_FORE, MAX_SPIKES };

__device__ __forceinline__ void bar(int id, int n) {
    asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory");
}

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

// One image's input spikes. Inputs that spike at all are `lit` (ascending) with
// their multiply-high divisor `litc`: for n, d < 2^16, n / d == mulhi(n, c) and
// d divides n iff n * c <= c - 1 (mod 2^32), where c = 2^32 / d rounded up.
struct Sched {
    unsigned short* per;  // n_in, 0 = silent
    unsigned short* lit;
    unsigned* litc;
    int* st;              // steps + 2, where each step's inputs start in act
    unsigned short* nx;   // end of the slice of steps that starts at t, see slice_end
    unsigned short* act;  // inputs spiking at each step, ascending
};

// Builds the schedule of image `px` with the P producer threads.
__device__ void build(const unsigned char* px, const Sched& s, float* vpad, int* nlit_out,
                      int* max_out, int* wcp, const float* f, const int* ic, int pt, int P, int cap,
                      unsigned char* rec_act)
{
    const int n_in = ic[N_IN], W = ic[WIDTH], H = ic[HEIGHT], steps = ic[STEPS];
    const int PW = W + 4, lane = pt & 31, warp = pt >> 5, nw = P >> 5;

    for (int t = pt; t < steps + 2; t += P) s.st[t] = 0;
    // pixels as floats in a zero border, so the 5x5 taps need no bounds checks
    const int plane = PW * (H + 4);
    for (int q0 = pt; q0 < plane; q0 += 8 * P) {
        float v8[8];
#pragma unroll
        for (int u = 0; u < 8; u++) {
            const int q = q0 + u * P, y = q / PW - 2, x = q % PW - 2;
            v8[u] = (q < plane && y >= 0 && y < H && x >= 0 && x < W) ? (float)px[y * W + x] / 255.0f : 0.0f;
        }
#pragma unroll
        for (int u = 0; u < 8; u++)
            if (q0 + u * P < plane) vpad[q0 + u * P] = v8[u];
    }
    bar(NBAR_P, P);

    // receptive field -> period -> per-step spike counts
    for (int i = pt; i < n_in; i += P) {
        const int y = i / W, x = i - y * W;
        const float* v = vpad + y * PW + x;
        float pot = 0.0f;
#pragma unroll
        for (int ky = 0; ky < 5; ky++)
#pragma unroll
            for (int kx = 0; kx < 5; kx++) pot += KERNEL[ky][kx] * v[ky * PW + kx];
        const int pr = period(pot, f, steps);
        s.per[i] = (unsigned short)pr;
        if (pr) for (int t = pr; t <= steps; t += pr) atomicAdd(&s.st[t], 1);
    }
    bar(NBAR_P, P);

    // peak simultaneous spikes, and counts -> start offsets (one warp, lanes own runs)
    if (warp == 0) {
        const int n = steps + 2, per_lane = (n + 31) / 32, lo = lane * per_lane;
        const int hi = min(lo + per_lane, n);
        int sum = 0, mx = 0;
        for (int t = lo; t < hi; t++) { sum += s.st[t]; mx = max(mx, s.st[t]); }
        int scan = sum;
        for (int o = 1; o < 32; o <<= 1) {
            int v = __shfl_up_sync(0xffffffffu, scan, o);
            if (lane >= o) scan += v;
        }
        for (int o = 16; o > 0; o >>= 1) mx = max(mx, __shfl_down_sync(0xffffffffu, mx, o));
        int acc = scan - sum;
        for (int t = lo; t < hi; t++) { int c = s.st[t]; s.st[t] = acc; acc += c; }
        if (lane == 0) *max_out = mx;
    }

    // lit inputs, ascending: each thread owns a run of inputs
    const int run = (n_in + P - 1) / P, first = pt * run, last = min(first + run, n_in);
    int c = 0;
    for (int i = first; i < last; i++) c += s.per[i] > 0;
    int scan = c;
    for (int o = 1; o < 32; o <<= 1) {
        int v = __shfl_up_sync(0xffffffffu, scan, o);
        if (lane >= o) scan += v;
    }
    if (lane == 31) wcp[warp] = scan;
    bar(NBAR_P, P);
    int off = scan - c, nlit = 0;
    for (int q = 0; q < nw; q++) {
        if (q < warp) off += wcp[q];
        nlit += wcp[q];
    }
    for (int i = first; i < last; i++) {
        const int pr = s.per[i];
        if (pr) { s.lit[off] = (unsigned short)i; s.litc[off] = 0xFFFFFFFFu / pr + 1; off++; }
    }
    if (pt == 0) *nlit_out = nlit;
    bar(NBAR_P, P);

    for (int t = 1 + pt; t <= steps; t += P) {
        int ts = 16, t1;
        for (;; ts >>= 1) {
            t1 = min(t + ts, steps + 1);
            if (s.st[t1] - s.st[t] <= cap || ts == 1) break;
        }
        s.nx[t] = (unsigned short)t1;
    }

    // each warp collects whole steps: its lanes test 32 lit inputs at a time, so the
    // spiking ones come out ascending
    const unsigned* __restrict__ litc = s.litc;
    const unsigned short* __restrict__ lit = s.lit;
    unsigned short* __restrict__ act = s.act;
    for (int t = 1 + warp; t <= steps; t += nw) {
        int pos = s.st[t];
#pragma unroll 4
        for (int l0 = 0; l0 < nlit; l0 += 32) {
            const int l = l0 + lane;
            const unsigned cc = l < nlit ? litc[l] : 1u;
            const bool hit = (unsigned)(t * cc) <= cc - 1;
            const unsigned mask = __ballot_sync(0xffffffffu, hit);
            if (hit) act[pos + __popc(mask & ((1u << lane) - 1))] = lit[l];
            pos += __popc(mask);
        }
    }
    if (rec_act) {
        bar(NBAR_P, P);
        for (int t = 1 + pt; t <= steps; t += P)
            for (int a = s.st[t]; a < s.st[t + 1]; a++) rec_act[(size_t)(t - 1) * n_in + s.act[a]] = 1;
    }
}

// One neuron's weights as a column of the input-major matrix in global memory, or a
// copy of it in shared memory.
struct GCol {
    float* w;
    int n_out, j;
    __device__ __forceinline__ float ld(int i) const { return w[i * n_out + j]; }
    __device__ __forceinline__ void st(int i, float v) const { w[i * n_out + j] = v; }
};
struct SCol {
    float* c;
    __device__ __forceinline__ float ld(int i) const { return c[i]; }
    __device__ __forceinline__ void st(int i, float v) const { c[i] = v; }
};

// sum of the neuron's weights from the inputs spiking at step t, ascending, as the CPU model does
template <class C>
__device__ __forceinline__ float drive(const C& c, const Sched& s, int t) {
    float d = 0.0f;
    const int end = s.st[t + 1];
#pragma unroll 4
    for (int a = s.st[t]; a < end; a++) d += c.ld(s.act[a]);
    return d;
}

// STDP of one neuron's weights for a post spike at step t: every pre spike within
// the window moves its weight, in time order per input. Loads go out G at a time.
template <class C>
__device__ void stdp_fire(const C& c, int t, const Sched& s, int nlit, const float* table,
                          const float* f, const int* ic)
{
    const int steps = ic[STEPS], tb = ic[T_BACK], tf = ic[T_FORE];
    const int lo = max(t - tb, 1), hi = min(t + tf, steps);
    const float sigma = f[SIGMA], w_max = f[W_MAX], w_min = f[W_MIN];
    for (int l0 = 0; l0 < nlit; l0 += G) {
        float wv[G];
#pragma unroll
        for (int u = 0; u < G; u++)
            if (l0 + u < nlit) wv[u] = c.ld(s.lit[l0 + u]);
#pragma unroll
        for (int u = 0; u < G; u++) {
            if (l0 + u >= nlit) continue;
            const int l = l0 + u, i = s.lit[l], pr = s.per[i];
            const unsigned cc = s.litc[l];
            const int m1 = __umulhi(hi, cc);
            float x = wv[u];
            for (int m = __umulhi(lo - 1, cc) + 1; m <= m1; m++) {
                const float dw = table[m * pr + tb - t];
                x = dw > 0.0f ? x + sigma * dw * (w_max - x) : x + sigma * dw * (x - w_min);
            }
            c.st(i, x);
        }
    }
}

// The same STDP on a shared-memory column, with the neuron threads sharing the inputs.
__device__ void stdp_par(float* col, int t, const Sched& s, int nlit, const float* table,
                         const float* f, const int* ic, int tid, int NN)
{
    const int steps = ic[STEPS], tb = ic[T_BACK], tf = ic[T_FORE];
    const int lo = max(t - tb, 1), hi = min(t + tf, steps);
    const float sigma = f[SIGMA], w_max = f[W_MAX], w_min = f[W_MIN];
    for (int l = tid; l < nlit; l += NN) {
        const int i = s.lit[l], pr = s.per[i];
        const unsigned cc = s.litc[l];
        const int m1 = __umulhi(hi, cc);
        float x = col[i];
        for (int m = __umulhi(lo - 1, cc) + 1; m <= m1; m++) {
            const float dw = table[m * pr + tb - t];
            x = dw > 0.0f ? x + sigma * dw * (w_max - x) : x + sigma * dw * (x - w_min);
        }
        col[i] = x;
    }
}

// Runs one neuron from step tstar + 1 on; returns its spike count.
template <class C>
__device__ int run_after(const C& c, const Sched& S, int nlit, int tstar, float p, int t_rest,
                         int nspk, float thr, int j, int learn, const float* table, const float* f,
                         const int* ic, float* rec_pot, unsigned char* rec_fired)
{
    const int n_out = ic[N_OUT], steps = ic[STEPS], t_ref = ic[T_REF];
    const float rest = f[REST], floor_ = f[FLOOR], leak = f[LEAK];
    for (int t = tstar + 1; t <= steps; t++) {
        bool fired = false;
        if (t > t_rest) {
            p += drive(c, S, t);
            if (p > rest) p = fmaxf(p - leak, rest);
            if (p < floor_) p = floor_;
            if (p >= thr) {
                fired = true;
                p = rest;
                t_rest = t + t_ref;
                nspk++;
                if (learn) stdp_fire(c, t, S, nlit, table, f, ic);
            }
        }
        if (rec_pot) { rec_pot[(size_t)(t - 1) * n_out + j] = p; rec_fired[(size_t)(t - 1) * n_out + j] = fired; }
    }
    return nspk;
}

// Slice of steps starting at t0: the longest run (at most 16 steps) whose spikes fit `cap`
// gathered weights; a single step with more than `cap` spikes comes back as is.
__device__ __forceinline__ void slice_end(const Sched& S, int t0, int& t1, int& E) {
    t1 = S.nx[t0];
    E = S.st[t1] - S.st[t0];
}

// Loads the weights of the first E spikes (1 <= E <= 32) of the slice at t0 in one batch;
// the rest of the batch repeats the last one.
__device__ __forceinline__ void gather(float* rg, const float* w, const Sched& S, int t0, int E,
                                       int n_out, int j) {
    const int e0 = S.st[t0];
#pragma unroll
    for (int u = 0; u < 32; u++) rg[u] = w[S.act[e0 + min(u, E - 1)] * n_out + j];
}

__device__ __forceinline__ void keep(float* stage, const float* rg, int E, int NN, int j) {
#pragma unroll
    for (int u = 0; u < 32; u++)
        if (u < E) stage[u * NN + j] = rg[u];
}

// rec_*: per-step record of one presentation for the live view (null when unused)
extern "C" __global__ void __launch_bounds__(MAX_THREADS) present(
    const unsigned char* pixels, const int* order, int n_img, int learn,
    const float* fcfg, const int* icfg,
    float* w,      // input-major: w[i * n_out + j]
    float* theta,  // homeostatic threshold scale per neuron
    int* winners, unsigned short* spikes, float* bases,
    unsigned char* rec_act, float* rec_pot, unsigned char* rec_fired)
{
    const int n_in = icfg[N_IN], n_out = icfg[N_OUT], W = icfg[WIDTH], H = icfg[HEIGHT];
    const int steps = icfg[STEPS], t_ref = icfg[T_REF], tb = icfg[T_BACK], tf = icfg[T_FORE];
    const int max_spikes = icfg[MAX_SPIKES];
    const int NN = (n_out + 31) & ~31, NP = blockDim.x - NN, tid = threadIdx.x;
    const int CAP = min(32, max(8, 4096 / NN));  // weights a neuron gathers per slice of steps
    const int count = blockIdx.x < n_img ? (n_img - blockIdx.x + gridDim.x - 1) / gridDim.x : 0;
    if (count == 0) return;

    // shared memory: two schedules, then the producers' pixel plane, the gather stage,
    // the winner's column and drives, and the STDP table
    extern __shared__ __align__(16) unsigned char sm[];
    Sched buf[2];
    unsigned char* at = sm;
    for (int b = 0; b < 2; b++) {
        buf[b].per = (unsigned short*)at;  at += (n_in * 2 + 3) & ~3;
        buf[b].lit = (unsigned short*)at;  at += (n_in * 2 + 3) & ~3;
        buf[b].litc = (unsigned*)at;       at += n_in * 4;
        buf[b].st = (int*)at;              at += (steps + 2) * 4;
        buf[b].nx = (unsigned short*)at;   at += ((steps + 2) * 2 + 3) & ~3;
        buf[b].act = (unsigned short*)at;  at += (max_spikes * 2 + 3) & ~3;
    }
    float* vpad = (float*)at;   at += (W + 4) * (H + 4) * 4;
    float* stage = (float*)at;  at += CAP * NN * 4;
    float* col = (float*)at;    at += n_in * 4;
    float* dwin = (float*)at;   at += (steps + 2) * 4;
    float* table = (float*)at;
    __shared__ int s_nlit[2], s_max[2], s_tstar[2], s_fire, wcp[32];
    __shared__ unsigned long long s_key[2];
    __shared__ float s_len;

    for (int q = tid; q < tb + tf + 1; q += blockDim.x) table[q] = fcfg[TABLE + q];

    const bool consumer = tid < NN;
    const int pt = tid - NN;
#ifdef PROFILE
    long long mark = clock64(), pc[8] = {0};
    int last_win = -1;
#endif

    if (!consumer) {
        const int img = order ? order[blockIdx.x] : blockIdx.x;
        build(pixels + (size_t)img * n_in, buf[0], vpad, &s_nlit[0], &s_max[0], wcp, fcfg, icfg,
              pt, NP, CAP, rec_act);
        if (pt == 0) { s_tstar[0] = 0x7fffffff; s_key[0] = 0; }
    }
    __syncthreads();

    for (int m = 0; m < count; m++) {
        const int k = blockIdx.x + m * gridDim.x, b = m & 1;
        if (!consumer) {
            if (m + 1 < count) {
                const int kn = k + gridDim.x, img = order ? order[kn] : kn;
                build(pixels + (size_t)img * n_in, buf[b ^ 1], vpad, &s_nlit[b ^ 1], &s_max[b ^ 1],
                      wcp, fcfg, icfg, pt, NP, CAP, nullptr);
                if (pt == 0) { s_tstar[b ^ 1] = 0x7fffffff; s_key[b ^ 1] = 0; }
            }
            MARK(5)
        } else {
            const Sched& S = buf[b];
            const int j = tid, nlit = s_nlit[b];
            const bool live = j < n_out;
            const GCol gcol = {w, n_out, j};
            const float rest = fcfg[REST], floor_ = fcfg[FLOOR], leak = fcfg[LEAK];
            // a blank image gives no drive; keep the threshold above rest
            const float base = fmaxf((float)s_max[b] / fcfg[THRESHOLD_DIV], 1.0f);
            const float thr = live ? base * (1.0f + theta[j]) : 0.0f;
            float p = rest;
            int t_rest = -1, nspk = 0, tj = steps + 1;
            float pj = 0.0f;

            // Until the first spike nobody depends on anybody. Slices of steps: gather the
            // weights of all their spikes in one batch of loads (the next slice's loads fly
            // while this one integrates). Stop once the slice starts after the earliest
            // crossing seen; going on past it only wastes steps.
            if (live) {
                int t0 = 1, t1, E;
                slice_end(S, t0, t1, E);
                float rg[32];
                if (E > 0 && E <= CAP) {
                    gather(rg, w, S, t0, E, n_out, j);
                    keep(stage, rg, E, NN, j);
                }
                while (tj > steps) {
                    int n1 = t1, nE = 0;
                    const bool more = t1 <= steps;
                    if (more) {
                        slice_end(S, t1, n1, nE);
                        if (nE > 0 && nE <= CAP) gather(rg, w, S, t1, nE, n_out, j);
                    }
                    const int e0 = S.st[t0];
                    int e = 0, end = 0;
                    for (int t = t0; t < t1; t++) {
                        float d = 0.0f;
                        if (E <= CAP) {
                            end = S.st[t + 1] - e0;
#pragma unroll 4
                            for (; e < end; e++) d += stage[e * NN + j];
                        } else {
                            d = drive(gcol, S, t);
                        }
                        p += d;
                        if (p > rest) p = fmaxf(p - leak, rest);
                        if (p < floor_) p = floor_;
                        if (rec_pot) { rec_pot[(size_t)(t - 1) * n_out + j] = p; rec_fired[(size_t)(t - 1) * n_out + j] = 0; }
                        if (p >= thr) { tj = t; pj = p; atomicMin(&s_tstar[b], t); break; }
                    }
                    if (!more || t1 > *(volatile int*)&s_tstar[b]) break;
                    if (nE > 0 && nE <= CAP) keep(stage, rg, nE, NN, j);
                    t0 = t1; t1 = n1; E = nE;
                }
            }
            bar(NBAR_C, NN);
            MARK(0)
            const int tstar = s_tstar[b];
            int win = -1;
            if (tstar <= steps) {
                // first neuron furthest past its threshold wins, the last of equal ones
                if (live && tj == tstar)
                    atomicMax(&s_key[b], ((unsigned long long)__float_as_uint(pj - thr) << 32) | (unsigned)j);
                bar(NBAR_C, NN);
                win = (int)(s_key[b] & 0xffffffffu);
                // the winner works on a shared-memory copy of its weights from here on
                for (int i = tid; i < n_in; i += NN) col[i] = w[i * n_out + win];
                bar(NBAR_C, NN);
                MARK(1)

                // the winner fires at tstar; its weights only change at its own spikes, so
                // the drives of all later steps are computed together between spikes
                int t_cur = tstar;
                t_rest = tstar + t_ref;
                nspk = 1;
                p = rest;
                if (learn) { stdp_par(col, tstar, S, nlit, table, fcfg, icfg, tid, NN); bar(NBAR_C, NN); }
                if (j == win && rec_pot) { rec_pot[(size_t)(tstar - 1) * n_out + j] = rest; rec_fired[(size_t)(tstar - 1) * n_out + j] = 1; }
                const SCol sc = {col};
                for (bool first = true;; first = false) {
                    // the neuron is silent for t_ref steps after a spike: nothing to sum there
                    const int t_open = t_cur + t_ref + 1;
                    if (first || learn) {
                        for (int t = t_open + tid; t <= steps; t += NN) dwin[t] = drive(sc, S, t);
                        bar(NBAR_C, NN);
                    }
                    if (j == win) {
                        int ft = steps + 1;
                        if (rec_pot)
                            for (int t = t_cur + 1; t < t_open && t <= steps; t++) {
                                rec_pot[(size_t)(t - 1) * n_out + j] = rest;
                                rec_fired[(size_t)(t - 1) * n_out + j] = 0;
                            }
                        for (int t = t_open; t <= steps && ft > steps; t += 8) {
                            float dd[8];
#pragma unroll
                            for (int u = 0; u < 8; u++) dd[u] = dwin[min(t + u, steps + 1)];
#pragma unroll
                            for (int u = 0; u < 8; u++) {
                                const int tt = t + u;
                                if (tt > steps || ft <= steps) break;
                                bool fired = false;
                                if (tt > t_rest) {
                                    p += dd[u];
                                    if (p > rest) p = fmaxf(p - leak, rest);
                                    if (p < floor_) p = floor_;
                                    if (p >= thr) { fired = true; p = rest; t_rest = tt + t_ref; nspk++; }
                                }
                                if (rec_pot) { rec_pot[(size_t)(tt - 1) * n_out + j] = p; rec_fired[(size_t)(tt - 1) * n_out + j] = fired; }
                                // a spike changes the weights: hand over to the others
                                if (fired && learn) ft = tt;
                            }
                        }
                        s_fire = ft;
                    }
                    bar(NBAR_C, NN);
                    const int ft = s_fire;
                    if (ft > steps) break;
                    stdp_par(col, ft, S, nlit, table, fcfg, icfg, tid, NN);
                    bar(NBAR_C, NN);
                    t_cur = ft;
                }

                MARK(7)
                // the inhibited neurons start from the floor; with weights >= 0 one that cannot
                // collect enough drive from the spikes still to come never reaches threshold
                if (live && j != win) {
                    bool skip = !rec_pot && fcfg[W_MIN] >= 0.0f;
                    if (skip) {
                        float bound = 0.0f;
#pragma unroll 16
                        for (int l = 0; l < nlit; l++) {
                            const unsigned cc = S.litc[l];
                            bound += gcol.ld(S.lit[l]) * (float)(__umulhi(steps, cc) - __umulhi(tstar, cc));
                        }
                        skip = floor_ + bound * 1.001f < thr;
                    }
                    if (rec_pot) { rec_pot[(size_t)(tstar - 1) * n_out + j] = floor_; rec_fired[(size_t)(tstar - 1) * n_out + j] = 0; }
                    nspk = 0;
                    if (!skip)
                        nspk = run_after(gcol, S, nlit, tstar, floor_, -1, 0, thr, j, learn, table,
                                         fcfg, icfg, rec_pot, rec_fired);
                }
            }
            MARK(2)
            if (live) spikes[(size_t)k * n_out + j] = (unsigned short)nspk;
            if (tid == 0) { winners[k] = win; bases[k] = base; }

            if (learn) {
                if (win >= 0) {
                    // depress synapses from inputs that stayed silent, then rescale the row
                    for (int i = tid; i < n_in; i += NN)
                        if (!S.per[i]) col[i] = fmaxf(col[i] - fcfg[SILENT_PENALTY], fcfg[W_MIN]);
                    bar(NBAR_C, NN);
                    if (tid == 0) {
                        float len = 0.0f;
#pragma unroll 8
                        for (int i = 0; i < n_in; i++) len += col[i] * col[i];
                        s_len = sqrtf(len);
                    }
                    bar(NBAR_C, NN);
                    const float norm = fcfg[W_NORM], len = s_len;
                    for (int i = tid; i < n_in; i += NN) {
                        float wt = col[i];
                        if (norm > 0.0f && len > 0.0f) wt = fminf(wt * (norm / len), fcfg[W_MAX]);
                        w[i * n_out + win] = wt;
                    }
                }
                // drive every neuron toward an equal share of wins
                if (live) {
                    const float share = 1.0f / (float)n_out;
                    const float won = (win == j) ? 1.0f : 0.0f;
                    theta[j] = fmaxf(theta[j] + fcfg[THETA_RATE] * (won - share), -0.9f);
                }
            }
            MARK(3)
#ifdef PROFILE
            last_win = win;
#endif
        }
        __syncthreads();
#ifdef PROFILE
        if (consumer) MARK(4) else MARK(6)
        if (blockIdx.x == 0 && ((consumer && tid == last_win) || (!consumer && pt == 0)))
            for (int q = 0; q < 8; q++) atomicAdd(&prof[q], (unsigned long long)pc[q]);
        for (int q = 0; q < 8; q++) pc[q] = 0;
#endif
    }
}
