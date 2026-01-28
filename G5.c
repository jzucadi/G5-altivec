/*
 * G5.c - Optimized floating point summation for PowerPC G5 with AltiVec
 *
 * Compile with: gcc -O3 -faltivec -maltivec G5.c
 *
 * This implementation uses AltiVec SIMD instructions optimized for the
 * IBM PowerPC 970 (G5) processor's deep pipeline and memory subsystem.
 */

#include "altivec_common.h"

#if defined(__ALTIVEC__) && defined(__VEC__)
#define UNROLL_FACTOR 8                         // Number of vectors to unroll
#define BLOCK_FLOATS (VEC_SIZE * UNROLL_FACTOR) // 32 floats per block

/**
 * vec_sum - Sum an array of floats using PowerPC AltiVec SIMD instructions
 * @data: Pointer to float array (must be 16-byte aligned for best performance)
 * @N: Number of floats in array
 * 
 * Returns: Sum of all elements in the array
 * 
 * Optimizations for G5:
 * - 8-way loop unrolling to hide latency
 * - Prefetching for G5's memory subsystem
 * - Reduced register dependency chains
 * - Optimized tail processing
 * - Proper handling of small arrays
 */
float vec_sum(const float * __restrict__ data, unsigned int N) {
    // Handle empty array
    if (N == 0) return 0.0f;
    
    // Handle small arrays with scalar loop
    if (N < BLOCK_FLOATS) {
        float sum = 0.0f;
        for (unsigned int i = 0; i < N; ++i) {
            sum += data[i];
        }
        return sum;
    }
    
    // Assert alignment for vectorized path (can be removed in production)
    ASSERT_ALIGNED(data, "Data");
    
    // Initialize 8 accumulator vectors to zero
    vector float acc0 = vec_splats(0.0f);
    vector float acc1 = vec_splats(0.0f);
    vector float acc2 = vec_splats(0.0f);
    vector float acc3 = vec_splats(0.0f);
    vector float acc4 = vec_splats(0.0f);
    vector float acc5 = vec_splats(0.0f);
    vector float acc6 = vec_splats(0.0f);
    vector float acc7 = vec_splats(0.0f);
    
    const unsigned int blocks = N / BLOCK_FLOATS;
    const unsigned int tail = N % BLOCK_FLOATS;
    
    const float *ptr = data;
    
    // Set up initial prefetch stream
    vec_dst(data, PREFETCH_CONTROL_SEQ, 0);

    // Main vectorized loop - process 32 floats (8 vectors) per iteration
    for (unsigned int b = 0; b < blocks; ++b) {
        // Update prefetch periodically for upcoming data
        if ((b & (PREFETCH_UPDATE_INTERVAL - 1)) == 0 && b + PREFETCH_UPDATE_INTERVAL < blocks) {
            vec_dst(ptr + BLOCK_FLOATS * PREFETCH_UPDATE_INTERVAL, PREFETCH_CONTROL_SEQ, 0);
        }
        
        // Load and accumulate 8 vectors
        // Using separate accumulators reduces dependency chains
        acc0 = vec_add(acc0, vec_ld(0 * 16, ptr));
        acc1 = vec_add(acc1, vec_ld(1 * 16, ptr));
        acc2 = vec_add(acc2, vec_ld(2 * 16, ptr));
        acc3 = vec_add(acc3, vec_ld(3 * 16, ptr));
        acc4 = vec_add(acc4, vec_ld(4 * 16, ptr));
        acc5 = vec_add(acc5, vec_ld(5 * 16, ptr));
        acc6 = vec_add(acc6, vec_ld(6 * 16, ptr));
        acc7 = vec_add(acc7, vec_ld(7 * 16, ptr));
        
        ptr += BLOCK_FLOATS;
    }
    
    // Stop prefetching
    vec_dss(0);
    
    // Reduce 8 accumulators to 1 using binary tree reduction
    // This minimizes dependency chains on G5's deep pipeline
    acc0 = vec_add(acc0, acc1);
    acc2 = vec_add(acc2, acc3);
    acc4 = vec_add(acc4, acc5);
    acc6 = vec_add(acc6, acc7);
    
    acc0 = vec_add(acc0, acc2);
    acc4 = vec_add(acc4, acc6);
    
    vector float total = vec_add(acc0, acc4);
    
    // Process tail elements that form complete vectors
    const unsigned int tail_vectors = tail / VEC_SIZE;
    for (unsigned int i = 0; i < tail_vectors; ++i) {
        total = vec_add(total, vec_ld(i * 16, ptr));
    }
    ptr += tail_vectors * VEC_SIZE;
    
    // Horizontal reduction: sum all 4 lanes of the vector
    float result = horizontal_sum_scalar(total);
    
    // Process final scalar elements
    const unsigned int final_scalars = tail % VEC_SIZE;
    for (unsigned int i = 0; i < final_scalars; ++i) {
        result += ptr[i];
    }
    
    return result;
}

#ifdef TEST_VEC_SUM
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Test function to verify correctness
int main() {
    const int sizes[] = {0, 1, 15, 31, 32, 95, 96, 1000, 10000};
    const int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    
    for (int s = 0; s < num_sizes; ++s) {
        int N = sizes[s];
        
        // Allocate aligned memory
        float *data;
        if (posix_memalign((void**)&data, 16, N * sizeof(float)) != 0) {
            fprintf(stderr, "Failed to allocate aligned memory\n");
            return 1;
        }
        
        // Initialize with simple pattern
        float expected = 0.0f;
        for (int i = 0; i < N; ++i) {
            data[i] = (float)(i + 1);
            expected += data[i];
        }
        
        // Test the function
        float result = vec_sum(data, N);
        
        // Check result (with some tolerance for floating point errors)
        float error = result - expected;
        if (error < 0) error = -error;
        
        printf("N=%5d: result=%12.2f, expected=%12.2f, error=%e %s\n", 
               N, result, expected, error,
               (error < 1e-5 * expected || (expected == 0 && error == 0)) ? "✓" : "✗");
        
        free(data);
    }
    
    // Performance test for large array
    printf("\nPerformance test (N=1000000):\n");
    const int perf_N = 1000000;
    float *perf_data;
    if (posix_memalign((void**)&perf_data, 16, perf_N * sizeof(float)) != 0) {
        fprintf(stderr, "Failed to allocate aligned memory for performance test\n");
        return 1;
    }
    
    // Initialize
    for (int i = 0; i < perf_N; ++i) {
        perf_data[i] = 1.0f;
    }
    
    // Time the operation
    clock_t start = clock();
    float sum = 0;
    for (int iter = 0; iter < 1000; ++iter) {
        sum = vec_sum(perf_data, perf_N);
    }
    clock_t end = clock();
    
    double cpu_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("1000 iterations: %.3f seconds, sum=%.0f\n", cpu_time, sum);
    printf("Throughput: %.2f GB/s\n", 
           (1000.0 * perf_N * sizeof(float)) / (cpu_time * 1e9));
    
    free(perf_data);

    return 0;
}
#endif /* TEST_VEC_SUM */

#else /* !(__ALTIVEC__ && __VEC__) */

#warning "AltiVec not available - this code requires PowerPC with AltiVec support"

/* Provide stub declaration so the file can be parsed on non-PowerPC systems */
float vec_sum(const float *data, unsigned int N) {
    (void)data; (void)N;
    return 0.0f;
}

#endif /* __ALTIVEC__ && __VEC__ */
