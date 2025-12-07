/*
 * G5_fir.c - Optimized FIR filter for PowerPC G5 with AltiVec
 * 
 * Compile with: gcc -O3 -faltivec -maltivec G5_fir. c
 * 
 * This implementation uses AltiVec SIMD instructions optimized for the
 * IBM PowerPC 970 (G5) processor's deep pipeline and memory subsystem.
 */

#include <altivec.h>
#include <stdint.h>
#include <assert.h>
#include <string.h>

#define VEC_SIZE 4                              // Floats per vector (128-bit / 32-bit)
#define UNROLL_FACTOR 8                         // Number of output samples per iteration
#define BLOCK_FLOATS (VEC_SIZE * UNROLL_FACTOR) // 32 floats per block

// Prefetch distance for G5 (in cache lines, G5 has 128-byte cache lines)
#define PREFETCH_DISTANCE 256

// Maximum filter length for stack-allocated coefficient vector
#define MAX_FILTER_LEN 256

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
 * - Prefetching for G5's memory subsystem
 * - Efficient coefficient broadcasting
 * - Sliding window using vec_sld()
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
            float sum = 0. 0f;
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
    vector float coeff_vecs[MAX_FILTER_LEN];
    for (unsigned int k = 0; k < filter_len && k < MAX_FILTER_LEN; ++k) {
        coeff_vecs[k] = vec_splats(coeffs[k]);
    }
    
    const unsigned int vec_filter_len = (filter_len + VEC_SIZE - 1) / VEC_SIZE;
    
    // Process output samples in blocks of UNROLL_FACTOR
    const unsigned int output_blocks = output_len / UNROLL_FACTOR;
    const unsigned int output_tail = output_len % UNROLL_FACTOR;
    
    unsigned int out_idx = 0;
    
    // Main vectorized loop - process UNROLL_FACTOR outputs per iteration
    for (unsigned int b = 0; b < output_blocks; ++b) {
        // Prefetch upcoming input data
        vec_dst(input + out_idx + UNROLL_FACTOR, PREFETCH_DISTANCE, 0);
        
        // Initialize 8 accumulator vectors
        vector float acc0 = vec_splats(0.0f);
        vector float acc1 = vec_splats(0.0f);
        vector float acc2 = vec_splats(0. 0f);
        vector float acc3 = vec_splats(0.0f);
        vector float acc4 = vec_splats(0.0f);
        vector float acc5 = vec_splats(0. 0f);
        vector float acc6 = vec_splats(0.0f);
        vector float acc7 = vec_splats(0.0f);
        
        // For each coefficient, multiply with shifted input and accumulate
        for (unsigned int k = 0; k < filter_len; ++k) {
            vector float coeff = coeff_vecs[k];
            
            // Load input vectors for each output position
            // Each output[n] = sum(coeffs[k] * input[n+k])
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
        
        // Extract scalar results from accumulators
        union {
            vector float v;
            float f[VEC_SIZE];
        } extract;
        
        vec_st(acc0, 0, &extract.v); output[out_idx + 0] = extract.f[0];
        vec_st(acc1, 0, &extract. v); output[out_idx + 1] = extract.f[0];
        vec_st(acc2, 0, &extract.v); output[out_idx + 2] = extract. f[0];
        vec_st(acc3, 0, &extract.v); output[out_idx + 3] = extract.f[0];
        vec_st(acc4, 0, &extract.v); output[out_idx + 4] = extract.f[0];
        vec_st(acc5, 0, &extract. v); output[out_idx + 5] = extract.f[0];
        vec_st(acc6, 0, &extract.v); output[out_idx + 6] = extract.f[0];
        vec_st(acc7, 0, &extract.v); output[out_idx + 7] = extract. f[0];
        
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
    const unsigned int scalar_tail = output_len % VEC_SIZE;
    
    // Process 4 outputs at a time
    for (unsigned int v = 0; v < vec_outputs; ++v) {
        unsigned int base = v * VEC_SIZE;
        
        // Prefetch
        vec_dst(input + base + VEC_SIZE * 4, PREFETCH_DISTANCE, 0);
        
        vector float acc = vec_splats(0.0f);
        
        // Apply each coefficient
        for (unsigned int k = 0; k < filter_len; ++k) {
            vector float coeff = vec_splats(coeffs[k]);
            
            // Load 4 consecutive input values starting at input[base + k]
            // This gives us input[base+k], input[base+k+1], input[base+k+2], input[base+k+3]
            const float *ptr = input + base + k;
            
            // Handle potentially unaligned load
            vector float in_vec;
            if (((uintptr_t)ptr & 15) == 0) {
                in_vec = vec_ld(0, ptr);
            } else {
                // Unaligned load using vec_perm
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

// Reference scalar implementation for verification
void fir_reference(const float *input, float *output, const float *coeffs,
                   unsigned int input_len, unsigned int filter_len) {
    if (input_len < filter_len) return;
    unsigned int output_len = input_len - filter_len + 1;
    for (unsigned int n = 0; n < output_len; ++n) {
        float sum = 0.0f;
        for (unsigned int k = 0; k < filter_len; ++k) {
            sum += coeffs[k] * input[n + k];
        }
        output[n] = sum;
    }
}

int main() {
    printf("FIR Filter Test Suite for PowerPC G5 AltiVec\n");
    printf("=============================================\n\n");
    
    // Test configurations: {input_len, filter_len}
    const int tests[][2] = {
        {100, 5},      // Simple case
        {100, 16},     // Filter length = 4 vectors
        {1000, 32},    // Medium case
        {1000, 64},    // Longer filter
        {10000, 128},  // Large input
        {256, 7},      // Non-power-of-2 filter
        {33, 8},       // Small input, multiple of vec size filter
    };
    const int num_tests = sizeof(tests) / sizeof(tests[0]);
    
    printf("Correctness Tests:\n");
    printf("------------------\n");
    
    for (int t = 0; t < num_tests; ++t) {
        int input_len = tests[t][0];
        int filter_len = tests[t][1];
        int output_len = input_len - filter_len + 1;
        
        // Allocate aligned memory
        float *input, *output, *output_ref, *coeffs;
        posix_memalign((void**)&input, 16, input_len * sizeof(float));
        posix_memalign((void**)&output, 16, output_len * sizeof(float));
        posix_memalign((void**)&output_ref, 16, output_len * sizeof(float));
        posix_memalign((void**)&coeffs, 16, filter_len * sizeof(float));
        
        // Initialize input with sine wave
        for (int i = 0; i < input_len; ++i) {
            input[i] = sinf(2.0f * 3.14159f * i / 100.0f);
        }
        
        // Initialize coefficients (simple moving average)
        float coeff_sum = 0.0f;
        for (int k = 0; k < filter_len; ++k) {
            coeffs[k] = 1.0f / filter_len;
            coeff_sum += coeffs[k];
        }
        
        // Run reference
        fir_reference(input, output_ref, coeffs, input_len, filter_len);
        
        // Run optimized version
        fir_filter(input, output, coeffs, input_len, filter_len);
        
        // Compare results
        float max_error = 0.0f;
        for (int i = 0; i < output_len; ++i) {
            float err = fabsf(output[i] - output_ref[i]);
            if (err > max_error) max_error = err;
        }
        
        printf("input=%5d, filter=%3d, output=%5d: max_error=%.2e %s\n",
               input_len, filter_len, output_len, max_error,
               max_error < 1e-5 ? "✓" : "✗");
        
        // Also test vectorized version
        memset(output, 0, output_len * sizeof(float));
        fir_filter_vectorized(input, output, coeffs, input_len, filter_len);
        
        max_error = 0.0f;
        for (int i = 0; i < output_len; ++i) {
            float err = fabsf(output[i] - output_ref[i]);
            if (err > max_error) max_error = err;
        }
        
        printf("  (vectorized version):                   max_error=%.2e %s\n",
               max_error, max_error < 1e-5 ? "✓" : "✗");
        
        free(input);
        free(output);
        free(output_ref);
        free(coeffs);
    }
    
    // Performance test
    printf("\nPerformance Test:\n");
    printf("-----------------\n");
    
    const int perf_input_len = 100000;
    const int perf_filter_len = 64;
    const int perf_output_len = perf_input_len - perf_filter_len + 1;
    const int iterations = 100;
    
    float *perf_input, *perf_output, *perf_coeffs;
    posix_memalign((void**)&perf_input, 16, perf_input_len * sizeof(float));
    posix_memalign((void**)&perf_output, 16, perf_output_len * sizeof(float));
    posix_memalign((void**)&perf_coeffs, 16, perf_filter_len * sizeof(float));
    
    for (int i = 0; i < perf_input_len; ++i) {
        perf_input[i] = (float)rand() / RAND_MAX;
    }
    for (int k = 0; k < perf_filter_len; ++k) {
        perf_coeffs[k] = 1.0f / perf_filter_len;
    }
    
    // Time fir_filter
    clock_t start = clock();
    for (int iter = 0; iter < iterations; ++iter) {
        fir_filter(perf_input, perf_output, perf_coeffs, perf_input_len, perf_filter_len);
    }
    clock_t end = clock();
    double time1 = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    // Time fir_filter_vectorized
    start = clock();
    for (int iter = 0; iter < iterations; ++iter) {
        fir_filter_vectorized(perf_input, perf_output, perf_coeffs, perf_input_len, perf_filter_len);
    }
    end = clock();
    double time2 = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    // Time reference
    start = clock();
    for (int iter = 0; iter < iterations; ++iter) {
        fir_reference(perf_input, perf_output, perf_coeffs, perf_input_len, perf_filter_len);
    }
    end = clock();
    double time_ref = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    printf("Input: %d samples, Filter: %d taps, %d iterations\n\n",
           perf_input_len, perf_filter_len, iterations);
    printf("Reference (scalar):     %.3f sec\n", time_ref);
    printf("fir_filter (unrolled):  %.3f sec (%.1fx speedup)\n", time1, time_ref / time1);
    printf("fir_filter_vectorized:  %.3f sec (%.1fx speedup)\n", time2, time_ref / time2);
    
    // Calculate throughput
    double macs_per_run = (double)perf_output_len * perf_filter_len;
    double total_macs = macs_per_run * iterations;
    printf("\nThroughput (fir_filter_vectorized): %.2f GMAC/s\n",
           total_macs / time2 / 1e9);
    
    free(perf_input);
    free(perf_output);
    free(perf_coeffs);
    
    return 0;
}
#endif
