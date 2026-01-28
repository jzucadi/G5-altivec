/*
 * altivec_common.h - Common AltiVec utilities for PowerPC G5
 *
 * Shared definitions, macros, and inline functions for G5 AltiVec routines.
 * This header consolidates duplicated code across G5.c, G5_fir.c, and G5_gemv.c.
 */

#ifndef ALTIVEC_COMMON_H
#define ALTIVEC_COMMON_H

#include <stdint.h>
#include <assert.h>

#if defined(__ALTIVEC__) && defined(__VEC__)
#include <altivec.h>

/*
 * Vector size constant - floats per 128-bit AltiVec vector
 */
#define VEC_SIZE 4

/*
 * Prefetch parameters for G5 (128-byte cache lines)
 * DST control word format: block size (5 bits) | block count (8 bits) | block stride (16 bits)
 */
#define DST_CONTROL(size, count, stride) (((size) << 24) | ((count) << 16) | (stride))
#define G5_CACHE_LINE_SIZE 128
#define PREFETCH_BLOCKS 8
#define PREFETCH_STRIDE G5_CACHE_LINE_SIZE

/*
 * Alignment check macro
 * Usage: ASSERT_ALIGNED(ptr, "description")
 */
#define ASSERT_ALIGNED(ptr, msg) \
    assert(((uintptr_t)(ptr) & 15) == 0 && msg " should be 16-byte aligned")

/*
 * vec_extract_first - Extract the first scalar element from a vector
 * @v: The vector float to extract from
 *
 * Returns: The first (index 0) float element of the vector
 *
 * Uses vec_extract() on GCC/Clang for efficiency, falls back to
 * union-based extraction on other compilers.
 */
#if defined(__GNUC__) || defined(__clang__)

static inline float vec_extract_first(vector float v) {
    return vec_extract(v, 0);
}

static inline float vec_extract_at(vector float v, int i) {
    return vec_extract(v, i);
}

#else

static inline float vec_extract_first(vector float v) {
    union { vector float vec; float f[4]; } u;
    u.vec = v;
    return u.f[0];
}

static inline float vec_extract_at(vector float v, int i) {
    union { vector float vec; float f[4]; } u;
    u.vec = v;
    return u.f[i];
}

#endif /* __GNUC__ || __clang__ */

/*
 * horizontal_sum - Reduce a vector to a scalar sum using binary tree reduction
 * @v: The vector float to reduce
 *
 * Returns: A vector where all lanes contain the sum of the original 4 lanes
 *
 * This method minimizes dependency chains on G5's deep pipeline by using
 * vec_sld (shift left double) for efficient lane permutation.
 */
static inline vector float horizontal_sum(vector float v) {
    v = vec_add(v, vec_sld(v, v, 8));  /* Add lanes 0+2, 1+3 */
    v = vec_add(v, vec_sld(v, v, 4));  /* Final sum in all lanes */
    return v;
}

/*
 * horizontal_sum_scalar - Convenience function to get scalar result directly
 * @v: The vector float to reduce
 *
 * Returns: The sum of all 4 lanes as a scalar float
 */
static inline float horizontal_sum_scalar(vector float v) {
    return vec_extract_first(horizontal_sum(v));
}

#endif /* __ALTIVEC__ && __VEC__ */

#endif /* ALTIVEC_COMMON_H */
