# G5-altivec
 Best scalar floating point routines on PowerPC G5 Processor

Compile in gcc using -faltivec
 
Code depends on an algebraic substitution and re-expression of the tight loop written, to break the instruction serialization.
 
A general library routine would/could simply use a branch test at the top to skip the whole vector/unrolled loop to get down to the scalar tail-work-loop at the bottom if called with small N.

And a general libary routine would deal with unaligned data too ... via permutes in the loop with a permute constant built via vec_lvsl. This adds considerable complexity and costs some performance, particularly on G4, and to a lesser degree on G4+

AltiVec is a single-precision floating point and integer SIMD instruction set designed and owned by Apple, IBM, and Freescale Semiconductor (formerly Motorola's Semiconductor Products Sector) — the AIM alliance. It is implemented on versions of the PowerPC processor architecture, including Motorola's G4, IBM's G5 and POWER6 processors, and P.A. Semi's PWRficient PA6T. AltiVec is a trademark owned solely by Freescale, so the system is also referred to as Velocity Engine by Apple and VMX (Vector Multimedia Extension) by IBM and P.A. Semi.
