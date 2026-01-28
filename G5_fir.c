/*
 * G5_fir.c - Optimized FIR filter for PowerPC G5 with AltiVec
 *
 * Compile with: gcc -O3 -faltivec -maltivec G5_fir.c
 *
 * This implementation uses AltiVec SIMD instructions optimized for the
 * IBM PowerPC 970 (G5) processor's deep pipeline and memory subsystem.
 */

#include <string.h>
#include "altivec_common.h"

#if defined(__ALTIVEC__) && defined(__VEC__)

#define UNROLL_FACTOR 8                         // Number of output samples per iteration
#define BLOCK_FLOATS (VEC_SIZE * UNROLL_FACTOR) // 32 floats per block

/*
 * Maximum filter length for stack-allocated coefficient vector
 * 128 vectors * 16 bytes = 2KB stack usage (safe for most applications)
 * Filters longer than this fall back to scalar processing
 */
#define MAX_FILTER_LEN 128

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

    // For small filters, large filters (stack safety), or small outputs, use scalar
    if (filter_len < VEC_SIZE || filter_len > MAX_FILTER_LEN || output_len < UNROLL_FACTOR) {
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
    ASSERT_ALIGNED(input, "Input");
    ASSERT_ALIGNED(output, "Output");

    // Pre-splat coefficients into vectors for efficient broadcasting
    // Note: filter_len <= MAX_FILTER_LEN is guaranteed by the early return above
    vector float coeff_vecs[MAX_FILTER_LEN];

    for (unsigned int k = 0; k < filter_len; ++k) {
        coeff_vecs[k] = vec_splats(coeffs[k]);
    }

    // Process output samples in blocks of UNROLL_FACTOR
    const unsigned int output_blocks = output_len / UNROLL_FACTOR;

    unsigned int out_idx = 0;

    // Set up prefetch stream with G5-optimized parameters
    vec_dst(input, PREFETCH_CONTROL_SEQ, 0);

    // Main vectorized loop - process UNROLL_FACTOR outputs per iteration
    for (unsigned int b = 0; b < output_blocks; ++b) {
        // Update prefetch periodically for upcoming data
        if ((b & (PREFETCH_UPDATE_INTERVAL - 1)) == 0) {
            vec_dst(input + out_idx + UNROLL_FACTOR * PREFETCH_UPDATE_INTERVAL,
                    PREFETCH_CONTROL_SEQ, 0);
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
        const unsigned int k_unroll = (filter_len / 4) * 4;

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
        for (; k < filter_len; ++k) {
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

        // Extract scalar results using shared extraction function
        output[out_idx + 0] = vec_extract_first(acc0);
        output[out_idx + 1] = vec_extract_first(acc1);
        output[out_idx + 2] = vec_extract_first(acc2);
        output[out_idx + 3] = vec_extract_first(acc3);
        output[out_idx + 4] = vec_extract_first(acc4);
        output[out_idx + 5] = vec_extract_first(acc5);
        output[out_idx + 6] = vec_extract_first(acc6);
        output[out_idx + 7] = vec_extract_first(acc7);

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

    ASSERT_ALIGNED(input, "Input");
    ASSERT_ALIGNED(output, "Output");

    const unsigned int vec_outputs = output_len / VEC_SIZE;

    // Set up prefetch stream
    vec_dst(input, PREFETCH_CONTROL_SEQ, 0);

    // Process 4 outputs at a time
    for (unsigned int v = 0; v < vec_outputs; ++v) {
        unsigned int base = v * VEC_SIZE;

        // Update prefetch periodically
        if ((v & (PREFETCH_UPDATE_INTERVAL * 2 - 1)) == 0) {
            vec_dst(input + base + VEC_SIZE * PREFETCH_UPDATE_INTERVAL * 2,
                    PREFETCH_CONTROL_SEQ, 0);
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
                // ptr1 is ptr0+4 bytes, so it's unaligned - use permute method
                vector float v1_1 = vec_ld(0, ptr1);
                vector float v2_1 = vec_ld(16, ptr1);
                vector unsigned char perm1 = vec_lvsl(0, ptr1);
                in_vec1 = vec_perm(v1_1, v2_1, perm1);
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

#ifdef TEST_FIR_FILTER
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

// Scalar reference implementation for verification
static void fir_filter_scalar(const float *input, float *output,
                              const float *coeffs,
                              unsigned int input_len, unsigned int filter_len) {
    if (input_len < filter_len || filter_len == 0) return;
    unsigned int output_len = input_len - filter_len + 1;
    for (unsigned int n = 0; n < output_len; ++n) {
        float sum = 0.0f;
        for (unsigned int k = 0; k < filter_len; ++k) {
            sum += coeffs[k] * input[n + k];
        }
        output[n] = sum;
    }
}

static float max_error(const float *a, const float *b, unsigned int n) {
    float max_err = 0.0f;
    for (unsigned int i = 0; i < n; ++i) {
        float err = fabsf(a[i] - b[i]);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

int main() {
    printf("=== G5 AltiVec FIR Filter Test Suite ===\n\n");

    // Test configurations: {input_len, filter_len}
    const unsigned int configs[][2] = {
        {16, 4},      // Small
        {32, 8},      // Matches unroll factor
        {64, 16},     // Medium
        {100, 7},     // Odd sizes
        {256, 32},    // Larger
        {1024, 64},   // Performance test
        {1024, 128},  // Max filter length
        {1024, 200},  // Exceeds MAX_FILTER_LEN (uses scalar fallback)
    };
    const int num_configs = sizeof(configs) / sizeof(configs[0]);

    int all_passed = 1;

    printf("--- Correctness Tests (fir_filter) ---\n");

    for (int c = 0; c < num_configs; ++c) {
        unsigned int input_len = configs[c][0];
        unsigned int filter_len = configs[c][1];
        unsigned int output_len = input_len - filter_len + 1;

        // Allocate aligned memory
        float *input, *coeffs, *output_vec, *output_ref;
        if (posix_memalign((void**)&input, 16, input_len * sizeof(float)) != 0 ||
            posix_memalign((void**)&coeffs, 16, filter_len * sizeof(float)) != 0 ||
            posix_memalign((void**)&output_vec, 16, output_len * sizeof(float)) != 0 ||
            posix_memalign((void**)&output_ref, 16, output_len * sizeof(float)) != 0) {
            fprintf(stderr, "Memory allocation failed\n");
            return 1;
        }

        // Initialize with test pattern
        for (unsigned int i = 0; i < input_len; ++i) {
            input[i] = (float)((i % 17) - 8) * 0.1f;
        }
        for (unsigned int k = 0; k < filter_len; ++k) {
            coeffs[k] = (float)((k % 5) - 2) * 0.2f;
        }

        // Compute reference
        fir_filter_scalar(input, output_ref, coeffs, input_len, filter_len);

        // Compute vectorized
        fir_filter(input, output_vec, coeffs, input_len, filter_len);

        // Check results
        float err = max_error(output_vec, output_ref, output_len);
        int passed = (err < 1e-5f);

        printf("input=%4u, filter=%3u, output=%4u: max_err=%e %s\n",
               input_len, filter_len, output_len, err, passed ? "[PASS]" : "[FAIL]");

        if (!passed) all_passed = 0;

        free(input);
        free(coeffs);
        free(output_vec);
        free(output_ref);
    }

    printf("\n--- Correctness Tests (fir_filter_vectorized) ---\n");

    for (int c = 0; c < num_configs; ++c) {
        unsigned int input_len = configs[c][0];
        unsigned int filter_len = configs[c][1];
        unsigned int output_len = input_len - filter_len + 1;

        float *input, *coeffs, *output_vec, *output_ref;
        if (posix_memalign((void**)&input, 16, input_len * sizeof(float)) != 0 ||
            posix_memalign((void**)&coeffs, 16, filter_len * sizeof(float)) != 0 ||
            posix_memalign((void**)&output_vec, 16, output_len * sizeof(float)) != 0 ||
            posix_memalign((void**)&output_ref, 16, output_len * sizeof(float)) != 0) {
            fprintf(stderr, "Memory allocation failed\n");
            return 1;
        }

        for (unsigned int i = 0; i < input_len; ++i) {
            input[i] = (float)((i % 17) - 8) * 0.1f;
        }
        for (unsigned int k = 0; k < filter_len; ++k) {
            coeffs[k] = (float)((k % 5) - 2) * 0.2f;
        }

        fir_filter_scalar(input, output_ref, coeffs, input_len, filter_len);
        fir_filter_vectorized(input, output_vec, coeffs, input_len, filter_len);

        float err = max_error(output_vec, output_ref, output_len);
        int passed = (err < 1e-5f);

        printf("input=%4u, filter=%3u, output=%4u: max_err=%e %s\n",
               input_len, filter_len, output_len, err, passed ? "[PASS]" : "[FAIL]");

        if (!passed) all_passed = 0;

        free(input);
        free(coeffs);
        free(output_vec);
        free(output_ref);
    }

    // Performance benchmark
    printf("\n--- Performance Benchmark ---\n");

    const unsigned int perf_input = 10000;
    const unsigned int perf_filter = 64;
    const unsigned int perf_output = perf_input - perf_filter + 1;
    const int iterations = 1000;

    float *input, *coeffs, *output;
    if (posix_memalign((void**)&input, 16, perf_input * sizeof(float)) != 0 ||
        posix_memalign((void**)&coeffs, 16, perf_filter * sizeof(float)) != 0 ||
        posix_memalign((void**)&output, 16, perf_output * sizeof(float)) != 0) {
        fprintf(stderr, "Memory allocation failed for benchmark\n");
        return 1;
    }

    for (unsigned int i = 0; i < perf_input; ++i) input[i] = 1.0f;
    for (unsigned int k = 0; k < perf_filter; ++k) coeffs[k] = 1.0f / perf_filter;

    // Benchmark scalar
    clock_t start = clock();
    for (int iter = 0; iter < iterations; ++iter) {
        fir_filter_scalar(input, output, coeffs, perf_input, perf_filter);
    }
    clock_t end = clock();
    double scalar_time = ((double)(end - start)) / CLOCKS_PER_SEC;

    // Benchmark fir_filter
    start = clock();
    for (int iter = 0; iter < iterations; ++iter) {
        fir_filter(input, output, coeffs, perf_input, perf_filter);
    }
    end = clock();
    double vec_time = ((double)(end - start)) / CLOCKS_PER_SEC;

    // Benchmark fir_filter_vectorized
    start = clock();
    for (int iter = 0; iter < iterations; ++iter) {
        fir_filter_vectorized(input, output, coeffs, perf_input, perf_filter);
    }
    end = clock();
    double vec2_time = ((double)(end - start)) / CLOCKS_PER_SEC;

    printf("Input: %u samples, Filter: %u taps (%d iterations)\n\n",
           perf_input, perf_filter, iterations);
    printf("Scalar:              %.3f sec\n", scalar_time);
    printf("fir_filter:          %.3f sec (%.2fx speedup)\n",
           vec_time, scalar_time / vec_time);
    printf("fir_filter_vectorized: %.3f sec (%.2fx speedup)\n",
           vec2_time, scalar_time / vec2_time);

    free(input);
    free(coeffs);
    free(output);

    printf("\n=== %s ===\n", all_passed ? "ALL TESTS PASSED" : "SOME TESTS FAILED");

    return all_passed ? 0 : 1;
}
#endif /* TEST_FIR_FILTER */

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
