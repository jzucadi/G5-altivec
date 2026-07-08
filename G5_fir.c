/*
 * G5_fir.c - Optimized FIR filter for PowerPC G5 with AltiVec
 *
 * Compile with: powerpc-linux-gnu-gcc -O3 -maltivec -mabi=altivec G5_fir.c
 *
 * This implementation uses AltiVec SIMD instructions optimized for the
 * IBM PowerPC 970 (G5) processor's deep pipeline and memory subsystem.
 */

#include "altivec_common.h"

#if defined(__ALTIVEC__) && defined(__VEC__)

/*
 * fir_scalar - Scalar FIR over the output range [start, end)
 *
 * Shared by the small-size fallback, the vectorized tail, and the test
 * suite's reference computation.
 */
static void fir_scalar(const float *input, float *output, const float *coeffs,
                       unsigned int start, unsigned int end,
                       unsigned int filter_len) {
    for (unsigned int n = start; n < end; ++n) {
        float sum = 0.0f;
        for (unsigned int k = 0; k < filter_len; ++k) {
            sum += coeffs[k] * input[n + k];
        }
        output[n] = sum;
    }
}

/**
 * fir_filter - Apply FIR filter to input signal using PowerPC AltiVec SIMD
 * @input:   Pointer to input signal array (must be 16-byte aligned)
 * @output:  Pointer to output signal array (must be 16-byte aligned)
 * @coeffs:  Pointer to filter coefficients array
 * @input_len:   Number of samples in input signal
 * @filter_len:  Number of filter coefficients (taps)
 *
 * The output array must have space for (input_len - filter_len + 1) samples.
 *
 * Processes 4 output samples simultaneously: each coefficient is splatted
 * across a vector and multiplied against a 4-sample input window. The input
 * windows are generally unaligned, so they are loaded with the
 * vec_lvsl/vec_perm idiom.
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

    // For small cases, use scalar
    if (filter_len < VEC_SIZE || output_len < VEC_SIZE * 2) {
        fir_scalar(input, output, coeffs, 0, output_len, filter_len);
        return;
    }

    ASSERT_ALIGNED(input, "Input");
    ASSERT_ALIGNED(output, "Output");

    // The unaligned-load idiom below (vec_ld(0, p) + vec_ld(16, p) + vec_perm)
    // can touch up to 15 bytes past the last input element it needs, so the
    // final block of outputs could read past the end of the input array.
    // Peel one vector block off and let the scalar tail loop handle it: the
    // last vectorized block then starts at base <= output_len - 8, keeping
    // every 32-byte load window inside input[0 .. input_len-1].
    // (output_len >= VEC_SIZE * 2 is guaranteed by the early return above.)
    const unsigned int vec_outputs = (output_len - VEC_SIZE) / VEC_SIZE;

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

    // Process tail (including the block peeled off above)
    fir_scalar(input, output, coeffs, vec_outputs * VEC_SIZE, output_len,
               filter_len);
}

#ifdef TEST_FIR_FILTER
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
    printf("=== G5 AltiVec FIR Filter Test Suite ===\n\n");

    // Test configurations: {input_len, filter_len}
    const unsigned int configs[][2] = {
        {16, 4},      // Small
        {32, 8},      // Matches vector width * 2
        {64, 16},     // Medium
        {100, 7},     // Odd sizes
        {256, 32},    // Larger
        {1024, 64},   // Performance test
        {1024, 128},  // Long filter
        {1024, 200},  // Longer filter
    };
    const int num_configs = sizeof(configs) / sizeof(configs[0]);

    int all_passed = 1;

    printf("--- Correctness Tests ---\n");

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

        // Compute reference and vectorized results
        fir_scalar(input, output_ref, coeffs, 0, output_len, filter_len);
        fir_filter(input, output_vec, coeffs, input_len, filter_len);

        // Check results
        float err = max_error(output_vec, output_ref, output_len);
        int passed = (err < 1e-5f);

        printf("input=%4u, filter=%3u, output=%4u: max_err=%e %s\n",
               input_len, filter_len, output_len, err, passed ? "[PASS]" : "[FAIL]");

        if (!passed) {
            all_passed = 0;
            // Dump mismatching elements to localize the failure
            for (unsigned int i = 0; i < output_len; ++i) {
                float d = fabsf(output_vec[i] - output_ref[i]);
                if (d > 1e-5f) {
                    printf("    [%3u] vec=%g ref=%g\n",
                           i, output_vec[i], output_ref[i]);
                }
            }
        }

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
        fir_scalar(input, output, coeffs, 0, perf_output, perf_filter);
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

    printf("Input: %u samples, Filter: %u taps (%d iterations)\n\n",
           perf_input, perf_filter, iterations);
    printf("Scalar:     %.3f sec\n", scalar_time);
    printf("fir_filter: %.3f sec (%.2fx speedup)\n",
           vec_time, scalar_time / vec_time);

    free(input);
    free(coeffs);
    free(output);

    printf("\n=== %s ===\n", all_passed ? "ALL TESTS PASSED" : "SOME TESTS FAILED");

    return all_passed ? 0 : 1;
}
#endif /* TEST_FIR_FILTER */

#else /* !(__ALTIVEC__ && __VEC__) */

#warning "AltiVec not available - this code requires PowerPC with AltiVec support"

/* Provide stub declaration so the file can be parsed on non-PowerPC systems */
void fir_filter(const float *input, float *output, const float *coeffs,
                unsigned int input_len, unsigned int filter_len) {
    (void)input; (void)output; (void)coeffs;
    (void)input_len; (void)filter_len;
}

#endif /* __ALTIVEC__ && __VEC__ */
