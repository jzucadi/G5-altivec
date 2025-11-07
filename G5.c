#include <altivec.h>
#include <stdint.h>

// Altivec vector summation routine
// Requirements: data must be 16-byte aligned, N >= 96 floats

#define VEC_SIZE 4                              // Floats per vector (128-bit / 32-bit)
#define UNROLL_FACTOR 8                         // Number of vectors to unroll
#define BLOCK_FLOATS (VEC_SIZE * UNROLL_FACTOR) // 32 floats per block
#define BLOCK_BYTES (BLOCK_FLOATS * sizeof(float))

/**
 * vec_sum - Sum an array of floats using PowerPC Altivec SIMD instructions
 * @data: Pointer to 16-byte aligned float array
 * @N: Number of floats in array (must be >= 96)
 * 
 * Returns: Sum of all elements in the array
 * 
 * Optimizations:
 * - 8-way loop unrolling to hide latency
 * - Reduced register dependency chains
 * - Efficient horizontal reduction
 * - Minimal branching in main loop
 */
float vec_sum(float *data, unsigned int N) {
    // Early exit for invalid input
    if (N < 96) return 0.0f;

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
    
    float *ptr = data;

    // Main vectorized loop - process 32 floats (8 vectors) per iteration
    for (unsigned int b = 0; b < blocks; ++b) {
        // Load and accumulate 8 vectors
        // Using separate variables reduces register dependencies
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

    // Reduce 8 accumulators to 1 using binary tree reduction
    // This minimizes dependency chains
    acc0 = vec_add(acc0, acc1);
    acc2 = vec_add(acc2, acc3);
    acc4 = vec_add(acc4, acc5);
    acc6 = vec_add(acc6, acc7);
    
    acc0 = vec_add(acc0, acc2);
    acc4 = vec_add(acc4, acc6);
    
    vector float total = vec_add(acc0, acc4);

    // Horizontal reduction: sum all 4 lanes of the vector
    // Uses vec_sld (shift left double) for efficient lane extraction
    total = vec_add(total, vec_sld(total, total, 4));  // Add lanes 0+1, 2+3
    total = vec_add(total, vec_sld(total, total, 8));  // Add all 4 lanes

    // Extract scalar result from vector
    union {
        vector float v;
        float f[VEC_SIZE];
    } extract;
    
    vec_st(total, 0, &extract.v);
    float result = extract.f[0];

    // Process remaining elements (scalar tail loop)
    for (unsigned int i = 0; i < tail; ++i) {
        result += ptr[i];
    }

    return result;
}
