// Time loops of the network, one thread per neuron and sample. Layouts:
// drive (batch, steps, n), state (steps, batch, n), readout potential
// (steps + 1, batch, n). Compiled with --fmad=false, so the arithmetic matches
// the NumPy reference in net.py operation for operation.

extern "C" __global__ void lif_forward(const float* drive, float* mem_out, float* syn_out,
                                       float* spk_out, int batch, int steps, int n,
                                       float alpha, float beta) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch * n) return;
    int b = i / n, j = i % n;
    float mem = 0.0f, syn = 0.0f;
    for (int t = 0; t < steps; t++) {
        size_t o = ((size_t)t * batch + b) * n + j;
        float s = mem - 1.0f > 0.0f ? 1.0f : 0.0f;
        mem_out[o] = mem;
        syn_out[o] = syn;
        spk_out[o] = s;
        float d = drive[((size_t)b * steps + t) * n + j];
        mem = (beta * mem + syn) * (1.0f - s);
        syn = alpha * syn + d;
    }
}

// the hidden layer backward with the reset out of the gradient
extern "C" __global__ void lif_backward(const float* g_spk, const float* mem, const float* spk,
                                        float* g_drive, int batch, int steps, int n,
                                        float alpha, float beta, float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch * n) return;
    int b = i / n, j = i % n;
    float g_mem = 0.0f, g_syn = 0.0f;
    for (int t = steps - 1; t >= 0; t--) {
        size_t o = ((size_t)t * batch + b) * n + j;
        float keep = 1.0f - spk[o];
        g_drive[o] = g_syn;
        float a = scale * fabsf(mem[o] - 1.0f) + 1.0f;
        float slope = 1.0f / (a * a);
        float g_mem_t = g_spk[o] * slope + g_mem * beta * keep;
        g_syn = g_mem * keep + alpha * g_syn;
        g_mem = g_mem_t;
    }
}

// out[t+1] = beta out[t] + flt[t], flt[t+1] = alpha flt[t] + h2[t]
extern "C" __global__ void readout_forward(const float* h2, float* outs, int batch, int steps,
                                           int n, float alpha, float beta) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch * n) return;
    float flt = 0.0f, out = 0.0f;
    outs[i] = 0.0f;
    for (int t = 0; t < steps; t++) {
        out = beta * out + flt;
        flt = alpha * flt + h2[(size_t)t * batch * n + i];
        outs[(size_t)(t + 1) * batch * n + i] = out;
    }
}

extern "C" __global__ void readout_backward(const float* g_out, float* g_h2, int batch,
                                            int steps, int n, float alpha, float beta) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch * n) return;
    size_t stride = (size_t)batch * n;
    float gu = g_out[steps * stride + i], gf = 0.0f;
    for (int t = steps - 1; t >= 0; t--) {
        g_h2[t * stride + i] = gf;
        float gu_t = g_out[t * stride + i] + beta * gu;
        gf = gu + alpha * gf;
        gu = gu_t;
    }
}

// one step of a recurrent layer: the spikes of this step and the next state
extern "C" __global__ void lif_step(float* mem, float* syn, const float* drive_t,
                                    const float* rec_in, float* mem_out, float* syn_out,
                                    float* spk_out, int count, float alpha, float beta) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    float m = mem[i], y = syn[i];
    float s = m - 1.0f > 0.0f ? 1.0f : 0.0f;
    mem_out[i] = m;
    syn_out[i] = y;
    spk_out[i] = s;
    mem[i] = (beta * m + y) * (1.0f - s);
    syn[i] = alpha * y + (drive_t[i] + rec_in[i]);
}

// one step back through a recurrent layer; g_s already holds the gradient
// reaching this step's spikes from the readout and the recurrence
extern "C" __global__ void lif_step_back(float* g_mem, float* g_syn, const float* g_s,
                                         const float* mem, const float* spk, float* g_drive_t,
                                         int count, float alpha, float beta, float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    float keep = 1.0f - spk[i], gm = g_mem[i], gy = g_syn[i];
    g_drive_t[i] = gy;
    float a = scale * fabsf(mem[i] - 1.0f) + 1.0f;
    float slope = 1.0f / (a * a);
    g_mem[i] = g_s[i] * slope + gm * beta * keep;
    g_syn[i] = gm * keep + alpha * gy;
}
