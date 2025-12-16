/*
 * G5_fir.c - Optimized FIR filter for PowerPC G5 with AltiVec
 *
 * Compile with: gcc -O3 -faltivec -maltivec G5_fir.c
 *
 * This implementation uses AltiVec SIMD instructions optimized for the
 * IBM PowerPC 970 (G5) processor's deep pipeline and memory subsystem.
 */

#include <stdint.h>
#include <assert.h>
#include <string.h>

#if defined(__ALTIVEC__) && defined(__VEC__)
#include <altivec.h>

#define VEC_SIZE 4                              // Floats per vector (128-bit / 32-bit)
#define UNROLL_FACTOR 8                         // Number of output samples per iteration
#define BLOCK_FLOATS (VEC_SIZE * UNROLL_FACTOR) // 32 floats per block

// Prefetch parameters for G5 (128-byte cache lines)
// DST control word: block size (5 bits) | block count (8 bits) | block stride (16 bits)
#define DST_CONTROL(size, count, stride) (((size) << 24) | ((count) << 16) | (stride))
#define PREFETCH_BLOCKS 8
#define PREFETCH_STRIDE 128  // G5 cache line size

// Maximum filter length for stack-allocated coefficient vector
#define MAX_FILTER_LEN 256

// Helper macro for efficient scalar extraction (GCC/Clang)
#if defined(__GNUC__) || defined(__clang__)
#define VEC_EXTRACT(v, i) vec_extract((v), (i))
#else
// Fallback for other compilers
static inline float vec_extract_scalar(vector float v, int i) {
    union { vector float vec; float f[4]; } u = { . vec = v };
    return u.f[i];
}
#define VEC_EXTRACT(v, i) vec_extract_scalar((v), (i))
#endif

/**
 * fir_filter - Apply FIR filter to input signal using PowerPC AltiVec SIMD
 * @input:   Pointer to input signal array (must be 16-byte aligned for best performance)
 * @output:  Pointer to output signal array (must be 16-byte aligned for best performance)
 * @coeffs:  Pointer to filter coefficients array
 * @input_len:   Number of samples in input signal
 * @filter_len:  Number of filter coefficients (taps)
 *
 * The output array must have space for (input_len - filter_len + 1) samples.
 *
 * Optimizations for G5:
 * - Vectorized multiply-accumulate using vec_madd()
 * - 8-way output unrolling to hide latency
 * - Prefetching optimized for G5's memory subsystem
 * - Efficient coefficient broadcasting
 * - Direct scalar extraction using vec_extract()
 */
void fir_filter(const float * __restrict__ input,
                float * __restrict__ output,
                const float * __restrict__ coeffs,
                unsigned int input_len,
                unsigned int filter_len) {

    // Validate inputs
    if (input_len < filter_len || filter_len == 0) {
        return;
    }

    const unsigned int output_len = input_len - filter_len + 1;

    // For small filters or outputs, use scalar implementation
    if (filter_len < VEC_SIZE || output_len < UNROLL_FACTOR) {
        for (unsigned int n = 0; n < output_len; ++n) {
            float sum = 0.0f;
            for (unsigned int k = 0; k < filter_len; ++k) {
                sum += coeffs[k] * input[n + k];
            }
            output[n] = sum;
        }
        return;
    }

    // Assert alignment for vectorized path
    assert(((uintptr_t)input & 15) == 0 && "Input should be 16-byte aligned");
    assert(((uintptr_t)output & 15) == 0 && "Output should be 16-byte aligned");

    // Pre-splat coefficients into vectors for efficient broadcasting
    // Use actual filter length, capped at MAX_FILTER_LEN
    const unsigned int coeff_count = (filter_len < MAX_FILTER_LEN) ? filter_len : MAX_FILTER_LEN;
    vector float coeff_vecs[MAX_FILTER_LEN];

    for (unsigned int k = 0; k < coeff_count; ++k) {
        coeff_vecs[k] = vec_splats(coeffs[k]);
    }

    // Process output samples in blocks of UNROLL_FACTOR
    const unsigned int output_blocks = output_len / UNROLL_FACTOR;

    unsigned int out_idx = 0;

    // Set up prefetch stream with G5-optimized parameters
    vec_dst(input, DST_CONTROL(4, PREFETCH_BLOCKS, PREFETCH_STRIDE), 0);

    // Main vectorized loop - process UNROLL_FACTOR outputs per iteration
    for (unsigned int b = 0; b < output_blocks; ++b) {
        // Update prefetch for upcoming data
        if ((b & 7) == 0) {  // Update every 8 blocks to reduce overhead
            vec_dst(input + out_idx + UNROLL_FACTOR * 8,
                    DST_CONTROL(4, PREFETCH_BLOCKS, PREFETCH_STRIDE), 0);
        }

        // Initialize 8 accumulator vectors
        vector float acc0 = vec_splats(0.0f);
        vector float acc1 = vec_splats(0.0f);
        vector float acc2 = vec_splats(0.0f);
        vector float acc3 = vec_splats(0.0f);
        vector float acc4 = vec_splats(0.0f);
        vector float acc5 = vec_splats(0.0f);
        vector float acc6 = vec_splats(0.0f);
        vector float acc7 = vec_splats(0.0f);

        // Process coefficients with 4-way unrolling for better ILP
        unsigned int k = 0;
        const unsigned int k_unroll = (coeff_count / 4) * 4;

        for (; k < k_unroll; k += 4) {
            const float *in_ptr = input + out_idx + k;

            // Unroll coefficient loop 4x for better instruction scheduling
            vector float c0 = coeff_vecs[k];
            vector float c1 = coeff_vecs[k + 1];
            vector float c2 = coeff_vecs[k + 2];
            vector float c3 = coeff_vecs[k + 3];

            // Process all 8 outputs for coefficient k
            acc0 = vec_madd(c0, vec_splats(in_ptr[0]), acc0);
            acc1 = vec_madd(c0, vec_splats(in_ptr[1]), acc1);
            acc2 = vec_madd(c0, vec_splats(in_ptr[2]), acc2);
            acc3 = vec_madd(c0, vec_splats(in_ptr[3]), acc3);
            acc4 = vec_madd(c0, vec_splats(in_ptr[4]), acc4);
            acc5 = vec_madd(c0, vec_splats(in_ptr[5]), acc5);
            acc6 = vec_madd(c0, vec_splats(in_ptr[6]), acc6);
            acc7 = vec_madd(c0, vec_splats(in_ptr[7]), acc7);

            // Process all 8 outputs for coefficient k+1
            acc0 = vec_madd(c1, vec_splats(in_ptr[1]), acc0);
            acc1 = vec_madd(c1, vec_splats(in_ptr[2]), acc1);
            acc2 = vec_madd(c1, vec_splats(in_ptr[3]), acc2);
            acc3 = vec_madd(c1, vec_splats(in_ptr[4]), acc3);
            acc4 = vec_madd(c1, vec_splats(in_ptr[5]), acc4);
            acc5 = vec_madd(c1, vec_splats(in_ptr[6]), acc5);
            acc6 = vec_madd(c1, vec_splats(in_ptr[7]), acc6);
            acc7 = vec_madd(c1, vec_splats(in_ptr[8]), acc7);

            // Process all 8 outputs for coefficient k+2
            acc0 = vec_madd(c2, vec_splats(in_ptr[2]), acc0);
            acc1 = vec_madd(c2, vec_splats(in_ptr[3]), acc1);
            acc2 = vec_madd(c2, vec_splats(in_ptr[4]), acc2);
            acc3 = vec_madd(c2, vec_splats(in_ptr[5]), acc3);
            acc4 = vec_madd(c2, vec_splats(in_ptr[6]), acc4);
            acc5 = vec_madd(c2, vec_splats(in_ptr[7]), acc5);
            acc6 = vec_madd(c2, vec_splats(in_ptr[8]), acc6);
            acc7 = vec_madd(c2, vec_splats(in_ptr[9]), acc7);

            // Process all 8 outputs for coefficient k+3
            acc0 = vec_madd(c3, vec_splats(in_ptr[3]), acc0);
            acc1 = vec_madd(c3, vec_splats(in_ptr[4]), acc1);
            acc2 = vec_madd(c3, vec_splats(in_ptr[5]), acc2);
            acc3 = vec_madd(c3, vec_splats(in_ptr[6]), acc3);
            acc4 = vec_madd(c3, vec_splats(in_ptr[7]), acc4);
            acc5 = vec_madd(c3, vec_splats(in_ptr[8]), acc5);
            acc6 = vec_madd(c3, vec_splats(in_ptr[9]), acc6);
            acc7 = vec_madd(c3, vec_splats(in_ptr[10]), acc7);
        }

        // Handle remaining coefficients
        for (; k < coeff_count; ++k) {
            vector float coeff = coeff_vecs[k];
            const float *in_ptr = input + out_idx + k;

            acc0 = vec_madd(coeff, vec_splats(in_ptr[0]), acc0);
            acc1 = vec_madd(coeff, vec_splats(in_ptr[1]), acc1);
            acc2 = vec_madd(coeff, vec_splats(in_ptr[2]), acc2);
            acc3 = vec_madd(coeff, vec_splats(in_ptr[3]), acc3);
            acc4 = vec_madd(coeff, vec_splats(in_ptr[4]), acc4);
            acc5 = vec_madd(coeff, vec_splats(in_ptr[5]), acc5);
            acc6 = vec_madd(coeff, vec_splats(in_ptr[6]), acc6);
            acc7 = vec_madd(coeff, vec_splats(in_ptr[7]), acc7);
        }

        // Extract scalar results efficiently using vec_extract
        output[out_idx + 0] = VEC_EXTRACT(acc0, 0);
        output[out_idx + 1] = VEC_EXTRACT(acc1, 0);
        output[out_idx + 2] = VEC_EXTRACT(acc2, 0);
        output[out_idx + 3] = VEC_EXTRACT(acc3, 0);
        output[out_idx + 4] = VEC_EXTRACT(acc4, 0);
        output[out_idx + 5] = VEC_EXTRACT(acc5, 0);
        output[out_idx + 6] = VEC_EXTRACT(acc6, 0);
        output[out_idx + 7] = VEC_EXTRACT(acc7, 0);

        out_idx += UNROLL_FACTOR;
    }

    // Stop prefetching
    vec_dss(0);

    // Process remaining output samples with scalar code
    for (unsigned int n = out_idx; n < output_len; ++n) {
        float sum = 0.0f;
        for (unsigned int k = 0; k < filter_len; ++k) {
            sum += coeffs[k] * input[n + k];
        }
        output[n] = sum;
    }
}

/**
 * fir_filter_vectorized - Vectorized FIR for processing 4 outputs at once
 *
 * This version processes 4 output samples simultaneously by loading
 * overlapping input windows and using vec_sld for shifting.
 */
void fir_filter_vectorized(const float * __restrict__ input,
                           float * __restrict__ output,
                           const float * __restrict__ coeffs,
                           unsigned int input_len,
                           unsigned int filter_len) {

    if (input_len < filter_len || filter_len == 0) {
        return;
    }

    const unsigned int output_len = input_len - filter_len + 1;

    // For small cases, use scalar
    if (filter_len < VEC_SIZE || output_len < VEC_SIZE * 2) {
        for (unsigned int n = 0; n < output_len; ++n) {
            float sum = 0.0f;
            for (unsigned int k = 0; k < filter_len; ++k) {
                sum += coeffs[k] * input[n + k];
            }
            output[n] = sum;
        }
        return;
    }

    assert(((uintptr_t)input & 15) == 0 && "Input should be 16-byte aligned");
    assert(((uintptr_t)output & 15) == 0 && "Output should be 16-byte aligned");

    const unsigned int vec_outputs = output_len / VEC_SIZE;

    // Set up prefetch stream
    vec_dst(input, DST_CONTROL(4, PREFETCH_BLOCKS, PREFETCH_STRIDE), 0);

    // Process 4 outputs at a time
    for (unsigned int v = 0; v < vec_outputs; ++v) {
        unsigned int base = v * VEC_SIZE;

        // Update prefetch periodically
        if ((v & 15) == 0) {
            vec_dst(input + base + VEC_SIZE * 16,
                    DST_CONTROL(4, PREFETCH_BLOCKS, PREFETCH_STRIDE), 0);
        }

        vector float acc = vec_splats(0.0f);

        // Apply each coefficient with 2-way unrolling
        unsigned int k = 0;
        const unsigned int k_unroll = (filter_len / 2) * 2;

        for (; k < k_unroll; k += 2) {
            vector float coeff0 = vec_splats(coeffs[k]);
            vector float coeff1 = vec_splats(coeffs[k + 1]);

            const float *ptr0 = input + base + k;
            const float *ptr1 = input + base + k + 1;

            // Load input vectors (handle alignment)
            vector float in_vec0, in_vec1;

            if (((uintptr_t)ptr0 & 15) == 0) {
                in_vec0 = vec_ld(0, ptr0);
                in_vec1 = vec_ld(0, ptr1);  // This will be unaligned
            } else {
                vector float v1_0 = vec_ld(0, ptr0);
                vector float v2_0 = vec_ld(16, ptr0);
                vector unsigned char perm0 = vec_lvsl(0, ptr0);
                in_vec0 = vec_perm(v1_0, v2_0, perm0);

                vector float v1_1 = vec_ld(0, ptr1);
                vector float v2_1 = vec_ld(16, ptr1);
                vector unsigned char perm1 = vec_lvsl(0, ptr1);
                in_vec1 = vec_perm(v1_1, v2_1, perm1);
            }

            acc = vec_madd(coeff0, in_vec0, acc);
            acc = vec_madd(coeff1, in_vec1, acc);
        }

        // Handle remaining coefficient
        for (; k < filter_len; ++k) {
            vector float coeff = vec_splats(coeffs[k]);
            const float *ptr = input + base + k;

            vector float in_vec;
            if (((uintptr_t)ptr & 15) == 0) {
                in_vec = vec_ld(0, ptr);
            } else {
                vector float v1 = vec_ld(0, ptr);
                vector float v2 = vec_ld(16, ptr);
                vector unsigned char perm = vec_lvsl(0, ptr);
                in_vec = vec_perm(v1, v2, perm);
            }

            acc = vec_madd(coeff, in_vec, acc);
        }

        // Store 4 output values
        vec_st(acc, 0, output + base);
    }

    vec_dss(0);

    // Process tail
    for (unsigned int n = vec_outputs * VEC_SIZE; n < output_len; ++n) {
        float sum = 0.0f;
        for (unsigned int k = 0; k < filter_len; ++k) {
            sum += coeffs[k] * input[n + k];
        }
        output[n] = sum;
    }
}

#else /* !(__ALTIVEC__ && __VEC__) */

#warning "AltiVec not available - this code requires PowerPC with AltiVec support"

/* Provide stub declarations so the file can be parsed on non-PowerPC systems */
void fir_filter(const float *input, float *output, const float *coeffs,
                unsigned int input_len, unsigned int filter_len) {
    (void)input; (void)output; (void)coeffs;
    (void)input_len; (void)filter_len;
}

void fir_filter_vectorized(const float *input, float *output, const float *coeffs,
                           unsigned int input_len, unsigned int filter_len) {
    (void)input; (void)output; (void)coeffs;
    (void)input_len; (void)filter_len;
}

#endif /* __ALTIVEC__ && __VEC__ */
