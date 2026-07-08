/*
 * G5_gemv.c - Optimized matrix-vector multiplication for PowerPC G5 with AltiVec
 *
 * Compile with: powerpc-linux-gnu-gcc -O3 -maltivec -mabi=altivec G5_gemv.c
 *
 * This implementation uses AltiVec SIMD instructions optimized for the
 * IBM PowerPC 970 (G5) processor's deep pipeline and memory subsystem.
 *
 * Computes: y = A * x  (where A is M x N matrix, x is N-vector, y is M-vector)
 */

#include <string.h>
#include "altivec_common.h"

#if defined(__ALTIVEC__) && defined(__VEC__)

#define UNROLL_FACTOR 4                         // Rows processed per iteration

/*
 * Scalar implementations - used as the small-size / misaligned-lda fallback
 * and as the reference computation in the test suite.
 */
static void gemv_scalar(const float *A, const float *x, float *y,
                        unsigned int M, unsigned int N, unsigned int lda) {
    for (unsigned int i = 0; i < M; ++i) {
        float sum = 0.0f;
        const float *row = A + i * lda;
        for (unsigned int j = 0; j < N; ++j) {
            sum += row[j] * x[j];
        }
        y[i] = sum;
    }
}

static void gemv_transposed_scalar(const float *A, const float *x, float *y,
                                   unsigned int M, unsigned int N, unsigned int lda) {
    for (unsigned int j = 0; j < N; ++j) {
        y[j] = 0.0f;
    }
    for (unsigned int i = 0; i < M; ++i) {
        float xi = x[i];
        const float *row = A + i * lda;
        for (unsigned int j = 0; j < N; ++j) {
            y[j] += row[j] * xi;
        }
    }
}

/**
 * gemv - Matrix-vector multiplication using PowerPC AltiVec SIMD
 * @A:      Pointer to M x N matrix in row-major order (must be 16-byte aligned)
 * @x:      Pointer to input vector of length N (must be 16-byte aligned)
 * @y:      Pointer to output vector of length M (must be 16-byte aligned)
 * @M:      Number of rows in matrix A
 * @N:      Number of columns in matrix A (length of vector x)
 * @lda:    Leading dimension of A (stride between rows, typically N)
 *
 * Computes y = A * x
 *
 * Optimizations for G5:
 * - 4-way row unrolling to hide memory latency
 * - Dual accumulators per row to reduce dependency chains
 * - Prefetching optimized for G5's memory subsystem
 * - Binary tree reduction for horizontal sum
 * - Efficient handling of non-aligned tail columns
 */
void gemv(const float * __restrict__ A,
          const float * __restrict__ x,
          float * __restrict__ y,
          unsigned int M,
          unsigned int N,
          unsigned int lda) {

    // Handle edge cases
    if (M == 0 || N == 0) {
        return;
    }

    // For small matrices or a misaligned row stride, use scalar implementation.
    // Rows sit at A + i*lda, so they are only 16-byte aligned when lda is a
    // multiple of VEC_SIZE; vec_ld on an unaligned row silently loads from a
    // rounded-down address and returns wrong results.
    if (N < VEC_SIZE * 2 || M < UNROLL_FACTOR || lda % VEC_SIZE != 0) {
        gemv_scalar(A, x, y, M, N, lda);
        return;
    }

    // Assert alignment for vectorized path
    ASSERT_ALIGNED(A, "Matrix A");
    ASSERT_ALIGNED(x, "Vector x");
    ASSERT_ALIGNED(y, "Vector y");

    const unsigned int row_blocks = M / UNROLL_FACTOR;
    const unsigned int col_vecs = N / VEC_SIZE;
    const unsigned int col_tail = N % VEC_SIZE;

    // Set up prefetch stream for x vector
    vec_dst(x, PREFETCH_CONTROL_SEQ, 0);

    unsigned int row = 0;

    // Main loop: process UNROLL_FACTOR rows at a time
    for (unsigned int rb = 0; rb < row_blocks; ++rb) {
        const float *row0 = A + (row + 0) * lda;
        const float *row1 = A + (row + 1) * lda;
        const float *row2 = A + (row + 2) * lda;
        const float *row3 = A + (row + 3) * lda;

        // Set up prefetch for matrix rows (use streams 1 and 2)
        vec_dst(row0, PREFETCH_CONTROL_SEQ, 1);
        vec_dst(row1, PREFETCH_CONTROL_SEQ, 2);

        // Initialize 4 accumulator vectors (one per row), with 2 accumulators each
        // Using 2 accumulators per row reduces dependency chains
        vector float acc0a = vec_splats(0.0f);
        vector float acc0b = vec_splats(0.0f);
        vector float acc1a = vec_splats(0.0f);
        vector float acc1b = vec_splats(0.0f);
        vector float acc2a = vec_splats(0.0f);
        vector float acc2b = vec_splats(0.0f);
        vector float acc3a = vec_splats(0.0f);
        vector float acc3b = vec_splats(0.0f);

        // Process columns in pairs of vectors for better ILP
        unsigned int col = 0;
        const unsigned int col_unroll = (col_vecs / 2) * 2;

        for (; col < col_unroll; col += 2) {
            int byte_off0 = (int)(col * 16);
            int byte_off1 = (int)((col + 1) * 16);

            // Load 2 vectors from x
            vector float xv0 = vec_ld(byte_off0, x);
            vector float xv1 = vec_ld(byte_off1, x);

            // Load and multiply-accumulate for all 4 rows
            // Row 0
            acc0a = vec_madd(vec_ld(byte_off0, row0), xv0, acc0a);
            acc0b = vec_madd(vec_ld(byte_off1, row0), xv1, acc0b);

            // Row 1
            acc1a = vec_madd(vec_ld(byte_off0, row1), xv0, acc1a);
            acc1b = vec_madd(vec_ld(byte_off1, row1), xv1, acc1b);

            // Row 2
            acc2a = vec_madd(vec_ld(byte_off0, row2), xv0, acc2a);
            acc2b = vec_madd(vec_ld(byte_off1, row2), xv1, acc2b);

            // Row 3
            acc3a = vec_madd(vec_ld(byte_off0, row3), xv0, acc3a);
            acc3b = vec_madd(vec_ld(byte_off1, row3), xv1, acc3b);
        }

        // Handle remaining complete vector
        for (; col < col_vecs; ++col) {
            int byte_off = (int)(col * 16);
            vector float xv = vec_ld(byte_off, x);

            acc0a = vec_madd(vec_ld(byte_off, row0), xv, acc0a);
            acc1a = vec_madd(vec_ld(byte_off, row1), xv, acc1a);
            acc2a = vec_madd(vec_ld(byte_off, row2), xv, acc2a);
            acc3a = vec_madd(vec_ld(byte_off, row3), xv, acc3a);
        }

        // Combine the two accumulators for each row
        vector float sum0 = vec_add(acc0a, acc0b);
        vector float sum1 = vec_add(acc1a, acc1b);
        vector float sum2 = vec_add(acc2a, acc2b);
        vector float sum3 = vec_add(acc3a, acc3b);

        // Horizontal reduction and scalar extraction for each row
        float result0 = horizontal_sum_scalar(sum0);
        float result1 = horizontal_sum_scalar(sum1);
        float result2 = horizontal_sum_scalar(sum2);
        float result3 = horizontal_sum_scalar(sum3);

        // Handle tail columns with scalar code
        if (col_tail > 0) {
            unsigned int tail_start = col_vecs * VEC_SIZE;
            for (unsigned int j = tail_start; j < N; ++j) {
                result0 += row0[j] * x[j];
                result1 += row1[j] * x[j];
                result2 += row2[j] * x[j];
                result3 += row3[j] * x[j];
            }
        }

        // Store results
        y[row + 0] = result0;
        y[row + 1] = result1;
        y[row + 2] = result2;
        y[row + 3] = result3;

        row += UNROLL_FACTOR;
    }

    // Stop prefetch streams
    vec_dss(0);
    vec_dss(1);
    vec_dss(2);

    // Process remaining rows
    for (unsigned int i = row; i < M; ++i) {
        const float *row_ptr = A + i * lda;

        vector float acc = vec_splats(0.0f);

        // Vectorized portion
        for (unsigned int col = 0; col < col_vecs; ++col) {
            int byte_off = (int)(col * 16);
            vector float xv = vec_ld(byte_off, x);
            vector float av = vec_ld(byte_off, row_ptr);
            acc = vec_madd(av, xv, acc);
        }

        // Horizontal reduction and scalar extraction
        float result = horizontal_sum_scalar(acc);

        // Scalar tail
        unsigned int tail_start = col_vecs * VEC_SIZE;
        for (unsigned int j = tail_start; j < N; ++j) {
            result += row_ptr[j] * x[j];
        }

        y[i] = result;
    }
}

/**
 * gemv_transposed - Transposed matrix-vector multiplication (y = A^T * x)
 * @A:      Pointer to M x N matrix in row-major order (must be 16-byte aligned)
 * @x:      Pointer to input vector of length M (must be 16-byte aligned)
 * @y:      Pointer to output vector of length N (must be 16-byte aligned)
 * @M:      Number of rows in matrix A (length of vector x)
 * @N:      Number of columns in matrix A (length of vector y)
 * @lda:    Leading dimension of A (stride between rows, typically N)
 *
 * Computes y = A^T * x (equivalent to column-wise dot products)
 *
 * This version is memory-access optimized: instead of strided column access,
 * it accumulates partial results while streaming through rows.
 */
void gemv_transposed(const float * __restrict__ A,
                     const float * __restrict__ x,
                     float * __restrict__ y,
                     unsigned int M,
                     unsigned int N,
                     unsigned int lda) {

    if (M == 0 || N == 0) {
        return;
    }

    // For small cases or a misaligned row stride, use scalar
    // (also zero-initializes y)
    if (N < VEC_SIZE * 2 || M < 4 || lda % VEC_SIZE != 0) {
        gemv_transposed_scalar(A, x, y, M, N, lda);
        return;
    }

    ASSERT_ALIGNED(A, "Matrix A");
    ASSERT_ALIGNED(x, "Vector x");
    ASSERT_ALIGNED(y, "Vector y");

    // Initialize output to zero
    memset(y, 0, N * sizeof(float));

    const unsigned int col_vecs = N / VEC_SIZE;
    const unsigned int col_tail = N % VEC_SIZE;

    // Process rows, accumulating into y
    // Unroll by 2 rows for better memory bandwidth utilization
    unsigned int i = 0;
    const unsigned int row_unroll = (M / 2) * 2;

    vec_dst(A, PREFETCH_CONTROL_SEQ, 0);

    for (; i < row_unroll; i += 2) {
        const float *row0 = A + i * lda;
        const float *row1 = A + (i + 1) * lda;

        // Prefetch next rows
        if (i + 2 < M) {
            vec_dst(A + (i + 2) * lda, PREFETCH_CONTROL_SEQ, 0);
        }

        vector float x0 = vec_splats(x[i]);
        vector float x1 = vec_splats(x[i + 1]);

        // Process columns in vectors
        for (unsigned int col = 0; col < col_vecs; ++col) {
            int byte_off = (int)(col * 16);

            vector float yv = vec_ld(byte_off, y);
            vector float a0 = vec_ld(byte_off, row0);
            vector float a1 = vec_ld(byte_off, row1);

            yv = vec_madd(a0, x0, yv);
            yv = vec_madd(a1, x1, yv);

            vec_st(yv, byte_off, y);
        }

        // Scalar tail
        if (col_tail > 0) {
            float x0s = x[i];
            float x1s = x[i + 1];
            unsigned int tail_start = col_vecs * VEC_SIZE;
            for (unsigned int k = tail_start; k < N; ++k) {
                y[k] += row0[k] * x0s + row1[k] * x1s;
            }
        }
    }

    vec_dss(0);

    // Handle remaining row
    for (; i < M; ++i) {
        const float *row_ptr = A + i * lda;
        float xi = x[i];
        vector float xv = vec_splats(xi);

        for (unsigned int col = 0; col < col_vecs; ++col) {
            int byte_off = (int)(col * 16);
            vector float yv = vec_ld(byte_off, y);
            vector float av = vec_ld(byte_off, row_ptr);
            yv = vec_madd(av, xv, yv);
            vec_st(yv, byte_off, y);
        }

        unsigned int tail_start = col_vecs * VEC_SIZE;
        for (unsigned int k = tail_start; k < N; ++k) {
            y[k] += row_ptr[k] * xi;
        }
    }
}

#ifdef TEST_GEMV
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static float relative_error(const float *result, const float *expected, unsigned int n) {
    float max_rel = 0.0f;
    for (unsigned int i = 0; i < n; ++i) {
        float denom = fabsf(expected[i]);
        if (denom < 1e-10f) denom = 1e-10f;
        float rel = fabsf(result[i] - expected[i]) / denom;
        if (rel > max_rel) max_rel = rel;
    }
    return max_rel;
}

int main() {
    printf("=== G5 AltiVec GEMV Test Suite ===\n\n");

    // Test configurations: {M, N}
    const unsigned int configs[][2] = {
        {1, 1},       // Minimal
        {4, 4},       // Small square
        {8, 8},       // Matches unroll factor
        {16, 16},     // Medium square
        {7, 13},      // Odd sizes
        {64, 64},     // Larger square
        {128, 32},    // Tall matrix
        {32, 128},    // Wide matrix
        {256, 256},   // Large
        {512, 512},   // Performance test size
    };
    const int num_configs = sizeof(configs) / sizeof(configs[0]);

    printf("--- Correctness Tests (gemv and gemv_transposed) ---\n");

    int all_passed = 1;

    for (int c = 0; c < num_configs; ++c) {
        unsigned int M = configs[c][0];
        unsigned int N = configs[c][1];
        unsigned int maxdim = M > N ? M : N;

        // Allocate aligned memory; x and y are sized max(M, N) so the same
        // buffers serve both gemv (x: N, y: M) and gemv_transposed (x: M, y: N)
        float *A, *x, *y_vec, *y_ref;
        if (posix_memalign((void**)&A, 16, M * N * sizeof(float)) != 0 ||
            posix_memalign((void**)&x, 16, maxdim * sizeof(float)) != 0 ||
            posix_memalign((void**)&y_vec, 16, maxdim * sizeof(float)) != 0 ||
            posix_memalign((void**)&y_ref, 16, maxdim * sizeof(float)) != 0) {
            fprintf(stderr, "Memory allocation failed\n");
            return 1;
        }

        // Initialize with test pattern
        for (unsigned int i = 0; i < M * N; ++i) {
            A[i] = (float)((i % 17) - 8) * 0.1f;
        }
        for (unsigned int j = 0; j < maxdim; ++j) {
            x[j] = (float)((j % 11) - 5) * 0.1f;
        }

        // y = A * x
        gemv_scalar(A, x, y_ref, M, N, N);
        gemv(A, x, y_vec, M, N, N);

        float err = max_error(y_vec, y_ref, M);
        float rel_err = relative_error(y_vec, y_ref, M);
        int passed = (rel_err < 1e-5f) || (err < 1e-6f);

        printf("gemv            M=%3u, N=%3u: max_err=%e, rel_err=%e %s\n",
               M, N, err, rel_err, passed ? "[PASS]" : "[FAIL]");
        if (!passed) all_passed = 0;

        // y = A^T * x
        gemv_transposed_scalar(A, x, y_ref, M, N, N);
        gemv_transposed(A, x, y_vec, M, N, N);

        err = max_error(y_vec, y_ref, N);
        rel_err = relative_error(y_vec, y_ref, N);
        passed = (rel_err < 1e-5f) || (err < 1e-6f);

        printf("gemv_transposed M=%3u, N=%3u: max_err=%e, rel_err=%e %s\n",
               M, N, err, rel_err, passed ? "[PASS]" : "[FAIL]");
        if (!passed) all_passed = 0;

        free(A);
        free(x);
        free(y_vec);
        free(y_ref);
    }

    printf("\n--- Odd-lda Tests (lda not a multiple of %d) ---\n", VEC_SIZE);
    {
        // Rows at A + i*lda are unaligned when lda isn't a multiple of
        // VEC_SIZE; both functions must fall back to scalar and stay correct.
        const unsigned int M = 16, N = 16, lda = 17;
        float *A, *x, *y_vec, *y_ref;
        if (posix_memalign((void**)&A, 16, M * lda * sizeof(float)) != 0 ||
            posix_memalign((void**)&x, 16, N * sizeof(float)) != 0 ||
            posix_memalign((void**)&y_vec, 16, M * sizeof(float)) != 0 ||
            posix_memalign((void**)&y_ref, 16, M * sizeof(float)) != 0) {
            fprintf(stderr, "Memory allocation failed\n");
            return 1;
        }
        for (unsigned int i = 0; i < M * lda; ++i) A[i] = (float)((i % 17) - 8) * 0.1f;
        for (unsigned int j = 0; j < N; ++j) x[j] = (float)((j % 11) - 5) * 0.1f;

        gemv_scalar(A, x, y_ref, M, N, lda);
        gemv(A, x, y_vec, M, N, lda);
        float err = max_error(y_vec, y_ref, M);
        int passed = (err < 1e-5f);
        printf("gemv            M=%u, N=%u, lda=%u: max_err=%e %s\n",
               M, N, lda, err, passed ? "[PASS]" : "[FAIL]");
        if (!passed) all_passed = 0;

        gemv_transposed_scalar(A, x, y_ref, M, N, lda);
        gemv_transposed(A, x, y_vec, M, N, lda);
        err = max_error(y_vec, y_ref, N);
        passed = (err < 1e-5f);
        printf("gemv_transposed M=%u, N=%u, lda=%u: max_err=%e %s\n",
               M, N, lda, err, passed ? "[PASS]" : "[FAIL]");
        if (!passed) all_passed = 0;

        free(A); free(x); free(y_vec); free(y_ref);
    }

    // Performance benchmark
    printf("\n--- Performance Benchmark ---\n");

    const unsigned int perf_M = 512;
    const unsigned int perf_N = 512;
    const int iterations = 1000;

    float *A, *x, *y;
    if (posix_memalign((void**)&A, 16, perf_M * perf_N * sizeof(float)) != 0 ||
        posix_memalign((void**)&x, 16, perf_N * sizeof(float)) != 0 ||
        posix_memalign((void**)&y, 16, perf_M * sizeof(float)) != 0) {
        fprintf(stderr, "Memory allocation failed for benchmark\n");
        return 1;
    }

    // Initialize
    for (unsigned int i = 0; i < perf_M * perf_N; ++i) {
        A[i] = 1.0f;
    }
    for (unsigned int j = 0; j < perf_N; ++j) {
        x[j] = 1.0f;
    }

    // Benchmark scalar
    clock_t start = clock();
    for (int iter = 0; iter < iterations; ++iter) {
        gemv_scalar(A, x, y, perf_M, perf_N, perf_N);
    }
    clock_t end = clock();
    double scalar_time = ((double)(end - start)) / CLOCKS_PER_SEC;

    // Benchmark vectorized
    start = clock();
    for (int iter = 0; iter < iterations; ++iter) {
        gemv(A, x, y, perf_M, perf_N, perf_N);
    }
    end = clock();
    double vec_time = ((double)(end - start)) / CLOCKS_PER_SEC;

    // Benchmark transposed
    start = clock();
    for (int iter = 0; iter < iterations; ++iter) {
        gemv_transposed(A, x, y, perf_M, perf_N, perf_N);
    }
    end = clock();
    double trans_time = ((double)(end - start)) / CLOCKS_PER_SEC;

    // Calculate FLOPS (2 * M * N operations per GEMV: multiply + add)
    double flops_per_gemv = 2.0 * perf_M * perf_N;
    double total_flops = flops_per_gemv * iterations;

    printf("Matrix size: %u x %u (%d iterations)\n", perf_M, perf_N, iterations);
    printf("\n");
    printf("Scalar:         %.3f sec, %.2f MFLOPS\n",
           scalar_time, (total_flops / scalar_time) / 1e6);
    printf("AltiVec gemv:   %.3f sec, %.2f MFLOPS (%.2fx speedup)\n",
           vec_time, (total_flops / vec_time) / 1e6, scalar_time / vec_time);
    printf("AltiVec trans:  %.3f sec, %.2f MFLOPS (%.2fx speedup)\n",
           trans_time, (total_flops / trans_time) / 1e6, scalar_time / trans_time);

    // Memory bandwidth calculation
    // gemv reads: M*N (matrix) + N (vector) + writes M (result)
    double bytes_per_gemv = (perf_M * perf_N + perf_N + perf_M) * sizeof(float);
    printf("\nMemory bandwidth (gemv): %.2f GB/s\n",
           (bytes_per_gemv * iterations) / (vec_time * 1e9));

    free(A);
    free(x);
    free(y);

    printf("\n=== %s ===\n", all_passed ? "ALL TESTS PASSED" : "SOME TESTS FAILED");

    return all_passed ? 0 : 1;
}
#endif /* TEST_GEMV */

#else /* !(__ALTIVEC__ && __VEC__) */

#warning "AltiVec not available - this code requires PowerPC with AltiVec support"

/* Provide stub declarations so the file can be parsed on non-PowerPC systems */
void gemv(const float *A, const float *x, float *y,
          unsigned int M, unsigned int N, unsigned int lda) {
    (void)A; (void)x; (void)y; (void)M; (void)N; (void)lda;
}

void gemv_transposed(const float *A, const float *x, float *y,
                     unsigned int M, unsigned int N, unsigned int lda) {
    (void)A; (void)x; (void)y; (void)M; (void)N; (void)lda;
}

#endif /* __ALTIVEC__ && __VEC__ */
