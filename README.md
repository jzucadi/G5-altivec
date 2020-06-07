# G5-altivec
 Best scalar floating point routines on G5
 
Code depends on an algebraic substitution and re-expression of the tight loop written, to break the instruction serialization.
 
Edited as commented in italics above, also fixed comment error about minimum loop-work for the loop shown. A general library routine would/could simply use a branch test at the top to skip the whole vector/unrolled loop to get down to the scalar tail-work-loop at the bottom if called with small N.


And a general libary routine would deal with unaligned data too ... via permutes in the loop with a permute constant built via vec_lvsl. This adds considerable complexity and costs some performance, particularly on G4, and to a lesser degree on G4+
