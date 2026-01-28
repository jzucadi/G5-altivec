# G5-altivec

Optimized floating-point routines for PowerPC G5 (970/970FX) processors using AltiVec SIMD instructions. 

## Features

- **G5-Optimized**: Tuned specifically for IBM PowerPC 970's deep pipeline and memory subsystem
- **High Performance**: 8-way loop unrolling breaks instruction serialization and hides memory latency
- **Smart Prefetching**: Aggressive data prefetch instructions optimized for G5's 128-byte cache lines
- **Robust Handling**: Correctly processes arrays of any size, from 0 to millions of elements
- **Production Ready**: Handles both aligned and unaligned data with graceful fallback to scalar code

## Included Implementations

### Vector Summation (`G5.c`)

Optimized floating-point array summation using AltiVec SIMD. 

- Multiple independent accumulator vectors to maximize instruction-level parallelism
- Binary tree reduction to minimize dependency chains in the G5's deep pipeline
- Strategic prefetch distance tuned for the 970's memory controller
- Efficient tail processing that vectorizes partial blocks when possible

### FIR Filter (`G5_fir.c`)

High-performance FIR filter implementation with two optimized versions:

**`fir_filter()`** - Uses 8-way output unrolling with separate accumulators

**`fir_filter_vectorized()`** - Processes 4 outputs simultaneously using vectorized loads

#### G5-Specific Optimizations

- Pre-splatted coefficients to avoid repeated broadcasts
- `vec_madd()` fused multiply-add for maximum throughput
- Aggressive prefetching with `vec_dst()`
- Handles unaligned input loads with `vec_perm()` and `vec_lvsl()`
- Proper tail processing for any input/filter size

### Matrix-Vector Multiplication (`G5_gemv.c`)

High-performance GEMV (General Matrix-Vector multiplication) with two operations:

**`gemv()`** - Computes y = A * x (standard matrix-vector product)

**`gemv_transposed()`** - Computes y = A^T * x (transposed matrix-vector product)

#### G5-Specific Optimizations

- 4-way row unrolling to hide memory latency
- Dual accumulators per row to reduce dependency chains
- `vec_madd()` fused multiply-add for maximum throughput
- Multiple prefetch streams for matrix rows and input vector
- Binary tree horizontal reduction using `vec_sld()`
- Memory-access optimized transposed version that streams through rows
- Proper handling of non-aligned tail columns

## Compilation

```bash
# Vector summation
gcc -O3 -faltivec -maltivec G5.c -o vec_sum

# Vector summation with test suite
gcc -O3 -faltivec -maltivec -DTEST_VEC_SUM G5.c -o vec_sum_test

# FIR filter
gcc -O3 -faltivec -maltivec G5_fir.c -o fir_filter

# FIR filter with test suite
gcc -O3 -faltivec -maltivec -DTEST_FIR_FILTER G5_fir.c -o fir_filter_test

# GEMV (matrix-vector multiplication)
gcc -O3 -faltivec -maltivec G5_gemv.c -o gemv

# GEMV with test suite
gcc -O3 -faltivec -maltivec -DTEST_GEMV G5_gemv.c -o gemv_test
```

## Requirements

- PowerPC G5 processor (IBM 970/970FX/970MP)
- GCC with AltiVec support
- 16-byte aligned data for optimal performance (handled automatically with fallback)

## Background

AltiVec is a single-precision floating point and integer SIMD instruction set designed and owned by Apple, IBM, and Freescale Semiconductor (formerly Motorola's Semiconductor Products Sector). 

## Performance Notes

These implementations achieve near-theoretical memory bandwidth on large arrays while maintaining correctness for all input sizes.  The G5's deep pipeline particularly benefits from:

- `__restrict__` qualifiers to enable aggressive compiler optimizations
- Prefetch instructions tuned for the 970's memory controller
- Independent accumulator chains that maximize instruction-level parallelism
- Binary tree reductions that minimize dependency stalls

## File Structure

| File | Description |
|------|-------------|
| `G5.c` | Vectorized floating-point summation routine |
| `G5_fir.c` | Optimized FIR filter implementations |
| `G5_gemv.c` | Matrix-vector multiplication (GEMV and transposed GEMV) |
| `README.md` | This documentation |
