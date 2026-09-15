// Runtime CPU feature detection for SIMD dispatch (x86 / x64).

#include "simd.h"

#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
#define PAR3_X86 1
#endif

#ifdef PAR3_X86

#ifdef _MSC_VER
#include <intrin.h>
static void par3_cpuidex(int info[4], int leaf, int subleaf)
{
	__cpuidex(info, leaf, subleaf);
}
static unsigned long long par3_xgetbv0(void)
{
	return _xgetbv(0);
}
#else
#include <cpuid.h>
static void par3_cpuidex(int info[4], int leaf, int subleaf)
{
	unsigned int a, b, c, d;
	__cpuid_count(leaf, subleaf, a, b, c, d);
	info[0] = (int)a;
	info[1] = (int)b;
	info[2] = (int)c;
	info[3] = (int)d;
}
static unsigned long long par3_xgetbv0(void)
{
	unsigned int eax, edx;
	__asm__ volatile ("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
	return ((unsigned long long)edx << 32) | eax;
}
#endif

#endif	// PAR3_X86

int par3_simd_level(void)
{
	static int level = -1;	// benign race: the computed value is identical

	if (level >= 0)
		return level;

	level = 0;
#ifdef PAR3_X86
	{
		int info[4];
		int max_leaf;

		par3_cpuidex(info, 0, 0);
		max_leaf = info[0];
		if (max_leaf >= 1){
			int ssse3, osxsave, avx;

			par3_cpuidex(info, 1, 0);
			ssse3 = (info[2] >> 9) & 1;
			osxsave = (info[2] >> 27) & 1;
			avx = (info[2] >> 28) & 1;
			if (ssse3)
				level = 1;

			// AVX2 also needs OS support for YMM state (XCR0 bits 1 and 2).
			if ( (osxsave != 0) && (avx != 0) && (max_leaf >= 7)
					&& ((par3_xgetbv0() & 6) == 6) ){
				par3_cpuidex(info, 7, 0);
				if ((info[1] >> 5) & 1)
					level = 2;
			}
		}
	}
#endif
	return level;
}
