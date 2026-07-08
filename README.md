# G5-altivec

Optimized single-precision floating-point routines for PowerPC G5 (970/970FX) processors using AltiVec SIMD instructions.

## Included Implementations

### Vector Summation (`G5.c`)

**`vec_sum(data, N)`** — sums a float array.

- 8 independent accumulator vectors (32 floats per iteration) to maximize instruction-level parallelism
- Binary tree reduction to minimize dependency chains in the G5's deep pipeline
- Prefetching (`vec_dst`) tuned for the 970's 128-byte cache lines
- Vectorized tail processing; arrays smaller than one 32-float block use scalar code

### FIR Filter (`G5_fir.c`)

**`fir_filter(input, output, coeffs, input_len, filter_len)`** — computes `output[n] = Σ coeffs[k] · input[n+k]`, producing `input_len − filter_len + 1` output samples.

- Processes 4 output samples per iteration using `vec_madd()` fused multiply-add
- Handles the unaligned input windows inherent to FIR with the `vec_lvsl()`/`vec_perm()` idiom
- The final partial block is always computed in scalar code so vector loads never read past the end of the input array
- Filters shorter than 4 taps or outputs shorter than 8 samples use scalar code

### Matrix-Vector Multiplication (`G5_gemv.c`)

**`gemv(A, x, y, M, N, lda)`** — computes y = A·x for a row-major M×N matrix with row stride `lda`.

**`gemv_transposed(A, x, y, M, N, lda)`** — computes y = Aᵀ·x. Memory-access optimized: streams through rows accumulating into `y` instead of making strided column accesses.

- 4-way row unrolling with dual accumulators per row to hide memory latency
- Multiple prefetch streams for matrix rows and the input vector
- Binary tree horizontal reduction using `vec_sld()`
- Scalar fallback for small matrices and whenever `lda` is not a multiple of 4 (rows would be misaligned for vector loads)

## Alignment Requirements

The vectorized paths require **16-byte aligned base pointers** (`input`/`output` for FIR; `A`, `x`, `y` for GEMV). Allocate with `posix_memalign(&p, 16, bytes)` or equivalent.

Alignment is checked with `assert()`, so debug builds abort on misaligned input. In release builds (`-DNDEBUG`) misaligned pointers produce **silently wrong results**, because AltiVec loads round addresses down to the previous 16-byte boundary. Small inputs — and any `lda` that is not a multiple of 4 in the GEMV routines — automatically take the scalar path, which has no alignment requirement.

## Building

With a PowerPC cross-toolchain (matches CI):

```bash
# Library object files
powerpc-linux-gnu-gcc -O3 -c -maltivec -mabi=altivec G5.c      -o vec_sum.o
powerpc-linux-gnu-gcc -O3 -c -maltivec -mabi=altivec G5_fir.c  -o fir_filter.o
powerpc-linux-gnu-gcc -O3 -c -maltivec -mabi=altivec G5_gemv.c -o gemv.o

# Test executables (each file carries its own test suite behind a -D flag)
powerpc-linux-gnu-gcc -O3 -static -maltivec -mabi=altivec -DTEST_VEC_SUM    G5.c      -o vec_sum_test    -lm
powerpc-linux-gnu-gcc -O3 -static -maltivec -mabi=altivec -DTEST_FIR_FILTER G5_fir.c  -o fir_filter_test -lm
powerpc-linux-gnu-gcc -O3 -static -maltivec -mabi=altivec -DTEST_GEMV       G5_gemv.c -o gemv_test       -lm

# Run without PowerPC hardware (G4 is the closest AltiVec-capable CPU model
# in 32-bit qemu user mode; it shares the G5's AltiVec ISA)
qemu-ppc -cpu G4 ./vec_sum_test
qemu-ppc -cpu G4 ./fir_filter_test
qemu-ppc -cpu G4 ./gemv_test
```

Native compilation on a G5 works with `gcc -O3 -maltivec -mabi=altivec`. On non-AltiVec platforms the files compile to no-op stubs (with a `#warning`) so they can at least be parsed and linked.

## Testing

Each test suite compares the vectorized kernels against plain scalar reference implementations across a range of sizes (including odd sizes and unroll-boundary cases), then runs a benchmark. The GEMV suite also includes an odd-`lda` regression test verifying the scalar fallback for misaligned row strides.

CI (GitHub Actions) cross-compiles everything for PowerPC and runs all three test suites under `qemu-ppc` on every push and pull request; test failures fail the build.

## Background

AltiVec is a single-precision floating point and integer SIMD instruction set designed and owned by Apple, IBM, and Freescale Semiconductor (formerly Motorola's Semiconductor Products Sector).

## File Structure

| File | Description |
|------|-------------|
| `altivec_common.h` | Shared AltiVec utilities (prefetch constants, alignment check, horizontal sum, test helpers) |
| `G5.c` | Vectorized floating-point summation (`vec_sum`) |
| `G5_fir.c` | Vectorized FIR filter (`fir_filter`) |
| `G5_gemv.c` | Matrix-vector multiplication (`gemv`, `gemv_transposed`) |
| `.github/workflows/c-cpp.yml` | CI: PowerPC cross-build + qemu test run |
