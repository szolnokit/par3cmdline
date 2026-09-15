#ifndef PAR3_SIMD_H
#define PAR3_SIMD_H

// Runtime CPU feature level for SIMD dispatch.
// 0 = scalar only, 1 = SSSE3 (PSHUFB), 2 = AVX2
int par3_simd_level(void);

#endif
