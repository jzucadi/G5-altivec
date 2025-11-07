# G5-altivec

Optimized floating-point summation routines for PowerPC G5 (970/970FX) processors using AltiVec SIMD instructions.

## Features

- **G5-Optimized**: Tuned specifically for IBM PowerPC 970's deep pipeline and memory subsystem
- **High Performance**: 8-way loop unrolling breaks instruction serialization and hides memory latency
- **Smart Prefetching**: Aggressive data prefetch instructions optimized for G5's 128-byte cache lines
- **Robust Handling**: Correctly processes arrays of any size, from 0 to millions of elements
- **Production Ready**: Handles both aligned and unaligned data with graceful fallback to scalar code

## Optimizations

The implementation uses several G5-specific techniques:
- Multiple independent accumulator vectors to maximize instruction-level parallelism
- Binary tree reduction to minimize dependency chains in the G5's deep pipeline
- Strategic prefetch distance tuned for the 970's memory controller
- Efficient tail processing that vectorizes partial blocks when possible
- `__restrict__` qualifiers to enable aggressive compiler optimizations

## Compilation
```bash
# Basic compilation
gcc -O3 -faltivec -maltivec G5.c -o vec_sum

# With test suite
gcc -O3 -faltivec -maltivec -DTEST_VEC_SUM G5.c -o vec_sum_test
```

## Requirements

- PowerPC G5 processor (IBM 970/970FX/970MP)
- GCC with AltiVec support
- 16-byte aligned data for optimal performance (handled automatically with fallback)

## Background

AltiVec is a single-precision floating point and integer SIMD instruction set designed and owned by Apple, IBM, and Freescale Semiconductor (formerly Motorola's Semiconductor Products Sector) — the AIM alliance. It is implemented on versions of the PowerPC processor architecture, including Motorola's G4, IBM's G5 and POWER6 processors, and P.A. Semi's PWRficient PA6T. AltiVec is a trademark owned solely by Freescale, so the system is also referred to as Velocity Engine by Apple and VMX (Vector Multimedia Extension) by IBM and P.A. Semi.

## Performance Notes

This implementation achieves near-theoretical memory bandwidth on large arrays while maintaining correctness for all input sizes. The G5's deep pipeline particularly benefits from the prefetch instructions and reduced dependency chains compared to simpler implementations.
