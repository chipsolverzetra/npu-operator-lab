// npu-operator-lab — from-scratch NPU operator implementations in C++17.
//
// Each operator is implemented twice behind a C ABI:
//   *_naive  — straightforward reference-style implementation
//   *_opt    — optimized: fewer memory passes, fusion, cache blocking,
//              ILP-friendly accumulation, restrict-qualified pointers so
//              -O3 -march=native can auto-vectorize cleanly.
//
// Tensor layout: row-major, float32 compute. Shapes are (B, N) for
// per-row operators. Quantized paths use int8 with scale + zero-point.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

extern "C" {

// ---------------------------------------------------------------------------
// LayerNorm: y = (x - mean) / sqrt(var + eps) * gamma + beta   (per row)
// ---------------------------------------------------------------------------
void layernorm_naive(const float* x, const float* gamma, const float* beta,
                     float* y, int B, int N, float eps) {
    for (int b = 0; b < B; ++b) {
        const float* xr = x + (size_t)b * N;
        float* yr = y + (size_t)b * N;
        double sum = 0.0;                                  // pass 1: mean
        for (int i = 0; i < N; ++i) sum += xr[i];
        double mean = sum / N;
        double var = 0.0;                                  // pass 2: variance
        for (int i = 0; i < N; ++i) {
            double d = (double)xr[i] - mean;
            var += d * d;
        }
        var /= N;
        float inv = 1.0f / std::sqrt((float)(var + eps));
        for (int i = 0; i < N; ++i)                        // pass 3: norm+affine
            yr[i] = (float)(((double)xr[i] - mean) * inv) * gamma[i] + beta[i];
    }
}

// Optimized: single statistics pass (sum + sum-of-squares), then one fused
// normalize+affine pass. y = x*a + c with a = inv*gamma, c = beta - mean*inv*gamma.
void layernorm_opt(const float* __restrict__ x, const float* __restrict__ gamma,
                   const float* __restrict__ beta, float* __restrict__ y,
                   int B, int N, float eps) {
    for (int b = 0; b < B; ++b) {
        const float* __restrict__ xr = x + (size_t)b * N;
        float* __restrict__ yr = y + (size_t)b * N;
        double sum = 0.0, sumsq = 0.0;                     // single pass stats
        for (int i = 0; i < N; ++i) {
            double v = xr[i];
            sum += v;
            sumsq += v * v;
        }
        double mean = sum / N;
        double var = sumsq / N - mean * mean;
        if (var < 0.0) var = 0.0;                          // numerical guard
        float inv = 1.0f / std::sqrt((float)(var + eps));
        float m = (float)(-mean * inv);
        for (int i = 0; i < N; ++i) {                      // fused affine
            float g = gamma[i];
            yr[i] = xr[i] * (inv * g) + (m * g + beta[i]);
        }
    }
}

// ---------------------------------------------------------------------------
// RMSNorm: y = x / sqrt(mean(x^2) + eps) * gamma   (per row)
// ---------------------------------------------------------------------------
void rmsnorm_naive(const float* x, const float* gamma, float* y,
                   int B, int N, float eps) {
    for (int b = 0; b < B; ++b) {
        const float* xr = x + (size_t)b * N;
        float* yr = y + (size_t)b * N;
        double sumsq = 0.0;                                // pass 1
        for (int i = 0; i < N; ++i) sumsq += (double)xr[i] * xr[i];
        float inv = 1.0f / std::sqrt((float)(sumsq / N + eps));
        for (int i = 0; i < N; ++i)                        // pass 2
            yr[i] = xr[i] * inv * gamma[i];
    }
}

void rmsnorm_opt(const float* __restrict__ x, const float* __restrict__ gamma,
                 float* __restrict__ y, int B, int N, float eps) {
    for (int b = 0; b < B; ++b) {
        const float* __restrict__ xr = x + (size_t)b * N;
        float* __restrict__ yr = y + (size_t)b * N;
        double sumsq = 0.0;
        for (int i = 0; i < N; ++i) {
            double v = xr[i];
            sumsq += v * v;
        }
        float inv = 1.0f / std::sqrt((float)(sumsq / N + eps));
        for (int i = 0; i < N; ++i) yr[i] = (xr[i] * inv) * gamma[i];
    }
}

// ---------------------------------------------------------------------------
// Softmax (per row), numerically stable.
// naive: 3 passes (max / exp+sum / normalize) + temp buffer.
// opt:   online softmax — single rescaling pass (FlashAttention-style),
//        then one normalize pass. 2 passes, no temp buffer.
// ---------------------------------------------------------------------------
void softmax_naive(const float* x, float* y, float* tmp, int B, int N) {
    for (int b = 0; b < B; ++b) {
        const float* xr = x + (size_t)b * N;
        float* yr = y + (size_t)b * N;
        float mx = xr[0];                                   // pass 1: max
        for (int i = 1; i < N; ++i) mx = std::max(mx, xr[i]);
        double sum = 0.0;                                   // pass 2: exp
        for (int i = 0; i < N; ++i) {
            tmp[i] = std::exp(xr[i] - mx);
            sum += tmp[i];
        }
        float inv = (float)(1.0 / sum);
        for (int i = 0; i < N; ++i) yr[i] = tmp[i] * inv;   // pass 3
    }
}

void softmax_opt(const float* __restrict__ x, float* __restrict__ y,
                 int B, int N) {
    for (int b = 0; b < B; ++b) {
        const float* __restrict__ xr = x + (size_t)b * N;
        float* __restrict__ yr = y + (size_t)b * N;
        // Online pass: maintain running (max, scaled sum). Mathematically
        // identical to max-subtraction, but streams the row once.
        float m = -INFINITY, s = 0.0f;
        for (int i = 0; i < N; ++i) {
            float m_new = m > xr[i] ? m : xr[i];
            s = s * std::exp(m - m_new) + std::exp(xr[i] - m_new);
            m = m_new;
        }
        float inv = 1.0f / s;
        for (int i = 0; i < N; ++i) yr[i] = std::exp(xr[i] - m) * inv;
    }
}

// ---------------------------------------------------------------------------
// Elementwise fusion demo: y = relu(a + b)
// separate: two passes + extra temp traffic. fused: one pass.
// ---------------------------------------------------------------------------
void add_relu_separate(const float* a, const float* b, float* tmp, float* y, int n) {
    for (int i = 0; i < n; ++i) tmp[i] = a[i] + b[i];       // pass 1
    for (int i = 0; i < n; ++i) {                          // pass 2
        float v = tmp[i];
        y[i] = v > 0.0f ? v : 0.0f;
    }
}

void add_relu_fused(const float* __restrict__ a, const float* __restrict__ b,
                    float* __restrict__ y, int n) {
    for (int i = 0; i < n; ++i) {                          // single pass
        float v = a[i] + b[i];
        y[i] = v > 0.0f ? v : 0.0f;
    }
}

// ---------------------------------------------------------------------------
// 2D transpose: dst[c*R + r] = src[r*C + c]
// naive: strided writes thrash the cache. blocked: 32x32 tiles stay in L1.
// ---------------------------------------------------------------------------
void transpose_naive(const float* src, float* dst, int R, int C) {
    for (int r = 0; r < R; ++r)
        for (int c = 0; c < C; ++c)
            dst[(size_t)c * R + r] = src[(size_t)r * C + c];
}

void transpose_blocked(const float* __restrict__ src, float* __restrict__ dst,
                       int R, int C) {
    constexpr int T = 32;  // 32x32 float tile = 4KB, fits comfortably in L1
    for (int r0 = 0; r0 < R; r0 += T) {
        for (int c0 = 0; c0 < C; c0 += T) {
            int r1 = std::min(r0 + T, R);
            int c1 = std::min(c0 + T, C);
            for (int r = r0; r < r1; ++r) {
                const float* srow = src + (size_t)r * C;
                for (int c = c0; c < c1; ++c)
                    dst[(size_t)c * R + r] = srow[c];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Row reduction: y[b] = sum_i x[b, i]
// opt uses 4 independent accumulators to break the FP dependency chain (ILP).
// ---------------------------------------------------------------------------
void reduce_sum_rows_naive(const float* x, float* y, int B, int N) {
    for (int b = 0; b < B; ++b) {
        const float* xr = x + (size_t)b * N;
        float s = 0.0f;
        for (int i = 0; i < N; ++i) s += xr[i];
        y[b] = s;
    }
}

void reduce_sum_rows_opt(const float* __restrict__ x, float* __restrict__ y,
                         int B, int N) {
    for (int b = 0; b < B; ++b) {
        const float* __restrict__ xr = x + (size_t)b * N;
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        int i = 0;
        int n4 = N & ~3;
        for (; i < n4; i += 4) {
            s0 += xr[i];
            s1 += xr[i + 1];
            s2 += xr[i + 2];
            s3 += xr[i + 3];
        }
        float s = (s0 + s1) + (s2 + s3);
        for (; i < N; ++i) s += xr[i];
        y[b] = s;
    }
}

// ---------------------------------------------------------------------------
// INT8 quantization: q = clamp(round(x / scale) + zero_point, -128, 127)
// Dequantization:   x = (q - zero_point) * scale
// opt versions are unrolled x8 to expose ILP + clean vectorization.
// ---------------------------------------------------------------------------
void int8_quantize_naive(const float* x, int8_t* y, int n, float scale,
                         int zero_point) {
    float inv = 1.0f / scale;
    for (int i = 0; i < n; ++i) {
        int q = (int)std::nearbyint(x[i] * inv) + zero_point;
        q = q < -128 ? -128 : (q > 127 ? 127 : q);
        y[i] = (int8_t)q;
    }
}

void int8_quantize_opt(const float* __restrict__ x, int8_t* __restrict__ y,
                       int n, float scale, int zero_point) {
    float inv = 1.0f / scale;
    int i = 0;
    int n8 = n & ~7;
    for (; i < n8; i += 8) {
        for (int k = 0; k < 8; ++k) {
            int q = (int)std::nearbyint(x[i + k] * inv) + zero_point;
            q = q < -128 ? -128 : (q > 127 ? 127 : q);
            y[i + k] = (int8_t)q;
        }
    }
    for (; i < n; ++i) {
        int q = (int)std::nearbyint(x[i] * inv) + zero_point;
        q = q < -128 ? -128 : (q > 127 ? 127 : q);
        y[i] = (int8_t)q;
    }
}

void int8_dequantize_naive(const int8_t* x, float* y, int n, float scale,
                           int zero_point) {
    for (int i = 0; i < n; ++i) y[i] = ((float)x[i] - (float)zero_point) * scale;
}

void int8_dequantize_opt(const int8_t* __restrict__ x, float* __restrict__ y,
                         int n, float scale, int zero_point) {
    float zp = (float)zero_point;
    int i = 0;
    int n8 = n & ~7;
    for (; i < n8; i += 8) {
        for (int k = 0; k < 8; ++k)
            y[i + k] = ((float)x[i + k] - zp) * scale;
    }
    for (; i < n; ++i) y[i] = ((float)x[i] - zp) * scale;
}

// ---------------------------------------------------------------------------
// BF16 converters (portable scalar, round-to-nearest-even). Mixed-precision
// helper: lets the Python side demonstrate fp32<->bf16 roundtrips.
// ---------------------------------------------------------------------------
static inline uint16_t fp32_to_bf16_bits(float f) {
    uint32_t u;
    __builtin_memcpy(&u, &f, sizeof(u));
    // Round to nearest even on the low 16 bits.
    uint32_t bias = 0x7FFF + ((u >> 16) & 1u);
    u += bias;
    return (uint16_t)(u >> 16);
}

void fp32_to_bf16(const float* x, uint16_t* y, int n) {
    for (int i = 0; i < n; ++i) y[i] = fp32_to_bf16_bits(x[i]);
}

void bf16_to_fp32(const uint16_t* x, float* y, int n) {
    for (int i = 0; i < n; ++i) {
        uint32_t u = (uint32_t)x[i] << 16;
        float f;
        __builtin_memcpy(&f, &u, sizeof(f));
        y[i] = f;
    }
}

// ---------------------------------------------------------------------------
// STREAM-triad-like microbenchmark: c[i] = a[i] + s * b[i].
// Used by bench.py to measure achievable single-thread DRAM bandwidth.
// ---------------------------------------------------------------------------
void stream_triad(const float* __restrict__ a, const float* __restrict__ b,
                  float* __restrict__ c, float s, size_t n) {
    for (size_t i = 0; i < n; ++i) c[i] = a[i] + s * b[i];
}

}  // extern "C"
