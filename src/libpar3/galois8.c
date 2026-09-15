// This is based on source code of Jerasure (v1.2), and modified for 8-bit Galois Field.

/* Galois.c
 * James S. Plank
 * April, 2007

Galois.tar - Fast Galois Field Arithmetic Library in C/C++
Copright (C) 2007 James S. Plank

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA

James S. Plank
Department of Electrical Engineering and Computer Science
University of Tennessee
Knoxville, TN 37996
plank@cs.utk.edu

 */

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

#include "simd.h"

#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
#define PAR3_X86 1
#include <immintrin.h>
#endif

#ifdef PAR3_X86
#if defined(__GNUC__) && !defined(__clang__)
#define TARGET_SSSE3 __attribute__((target("ssse3")))
#define TARGET_AVX2 __attribute__((target("avx2")))
#elif defined(__clang__)
#define TARGET_SSSE3 __attribute__((target("ssse3")))
#define TARGET_AVX2 __attribute__((target("avx2")))
#else
#define TARGET_SSSE3
#define TARGET_AVX2
#endif

/*
 * PSHUFB based GF(2^8) region multiply (classic 4-bit split tables):
 *   product(b) = TL[b & 15] ^ TH[b >> 4]
 * tbl[0..15] = TL, tbl[16..31] = TH.
 */
TARGET_SSSE3
static size_t gf8_region_mul_ssse3(const uint8_t *tbl, const uint8_t *src, uint8_t *dst, size_t nbytes, int add)
{
	__m128i tl = _mm_loadu_si128((const __m128i *)tbl);
	__m128i th = _mm_loadu_si128((const __m128i *)(tbl + 16));
	__m128i mask = _mm_set1_epi8(0x0F);
	size_t i;

	for (i = 0; i + 16 <= nbytes; i += 16){
		__m128i v = _mm_loadu_si128((const __m128i *)(src + i));
		__m128i lo = _mm_and_si128(v, mask);
		__m128i hi = _mm_and_si128(_mm_srli_epi64(v, 4), mask);
		__m128i p = _mm_xor_si128(_mm_shuffle_epi8(tl, lo), _mm_shuffle_epi8(th, hi));
		if (add)
			p = _mm_xor_si128(p, _mm_loadu_si128((const __m128i *)(dst + i)));
		_mm_storeu_si128((__m128i *)(dst + i), p);
	}
	return i;	// number of bytes processed
}

TARGET_AVX2
static size_t gf8_region_mul_avx2(const uint8_t *tbl, const uint8_t *src, uint8_t *dst, size_t nbytes, int add)
{
	__m256i tl = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)tbl));
	__m256i th = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)(tbl + 16)));
	__m256i mask = _mm256_set1_epi8(0x0F);
	size_t i;

	for (i = 0; i + 32 <= nbytes; i += 32){
		__m256i v = _mm256_loadu_si256((const __m256i *)(src + i));
		__m256i lo = _mm256_and_si256(v, mask);
		__m256i hi = _mm256_and_si256(_mm256_srli_epi64(v, 4), mask);
		__m256i p = _mm256_xor_si256(_mm256_shuffle_epi8(tl, lo), _mm256_shuffle_epi8(th, hi));
		if (add)
			p = _mm256_xor_si256(p, _mm256_loadu_si256((const __m256i *)(dst + i)));
		_mm256_storeu_si256((__m256i *)(dst + i), p);
	}
	return i;	// number of bytes processed
}
#endif	// PAR3_X86


// Create tables for 8-bit Galois Field
// Return main pointer of tables.
uint8_t * gf8_create_table(int prim_poly)
{
	int j, b;
	int x, y, logx, sum_j;
	uint8_t *galois_log_table, *galois_ilog_table, *galois_mult_table;

	// Allocate tables on memory
	// To fit CPU cache memory, table uses 8-bit integer.
	galois_log_table = malloc(sizeof(uint8_t) * 256 * (1 + 1 + 256));
	if (galois_log_table == NULL)
		return NULL;
	galois_ilog_table = galois_log_table + 256;
	galois_mult_table = galois_log_table + 256 * 2;

	// galois_log_table[0] is invalid, because power of 2 never becomes 0.
	galois_log_table[0] = prim_poly;	// Instead of invalid value, set generator polynomial.
	galois_ilog_table[255] = 1;	// 2 power 0 is 1. 2 power 255 is 1.

	b = 1;
	for (j = 0; j < 255; j++) {
		galois_log_table[b] = j;
		galois_ilog_table[j] = b;
		b = b << 1;
		if (b & 256)
			b = (b ^ prim_poly) & 255;
	}

	// Set multiply tables for x = 0
	j = 0;
	galois_mult_table[j] = 0;	// y = 0
	j++;
	for (y = 1; y < 256; y++){	// y > 0
		galois_mult_table[j] = 0;
		j++;
	}

	for (x = 1; x < 256; x++){	// x > 0
		galois_mult_table[j] = 0;	// y = 0
		j++;
		logx = galois_log_table[x];
		for (y = 1; y < 256; y++){	// y > 0
			sum_j = logx + galois_log_table[y];
			if (sum_j >= 255)
				sum_j -= 255;
			galois_mult_table[j] = galois_ilog_table[sum_j];
			j++;
		}
	}

	return galois_log_table;
}


// Return (x * y)
/*
// Normal slow version
int gf8_multiply(uint8_t *galois_log_table, int x, int y)
{
	int sum_j;
	int *galois_ilog_table;

	if (x == 0 || y == 0)
		return 0;
	galois_ilog_table = galois_log_table + 256;

	sum_j = galois_log_table[x] + galois_log_table[y];
	if (sum_j >= 255)
		sum_j -= 255;

	return galois_ilog_table[sum_j];
}
*/

// Using galois_mult_table
int gf8_multiply(uint8_t *galois_log_table, int x, int y)
{
	uint8_t *galois_mult_table;

	galois_mult_table = galois_log_table + 256 * 2;

	return galois_mult_table[(x << 8) | y];
}

// Return (x / y)
int gf8_divide(uint8_t *galois_log_table, int x, int y)
{
	int sum_j;
	uint8_t *galois_ilog_table;

	if (y == 0)
		return -1;	// Error: division by zero
	if (x == 0)
		return 0;
	galois_ilog_table = galois_log_table + 256;

	sum_j = galois_log_table[x] - galois_log_table[y];
	if (sum_j < 0)
		sum_j += 255;

	return galois_ilog_table[sum_j];
}

// Return (1 / y)
int gf8_reciprocal(uint8_t *galois_log_table, int y)
{
	uint8_t *galois_ilog_table;

	if (y == 0)
		return -1;	// Error: division by zero
	galois_ilog_table = galois_log_table + 256;

	return galois_ilog_table[ 255 - galois_log_table[y] ];
}


// Simplify and support size_t for 64-bit build
void gf8_region_multiply(uint8_t *galois_log_table,
						uint8_t *region,	/* Region to multiply */
						int multby,			/* Number to multiply by */
						size_t nbytes,		/* Number of bytes in region */
						uint8_t *r2,		/* If r2 != NULL, products go here */
						int add)
{
	size_t i;

	if (multby == 0) {
		if (add == 0){
			if (r2 == NULL)
				r2 = region;

			for (i = 0; i < nbytes; i++) {
				r2[i] = 0;
			}
		}

	} else if (multby == 1) {
		if (add == 0){
			if (r2 != NULL){
				for (i = 0; i < nbytes; i++) {
					r2[i] = region[i];
				}
			}
		} else {
			if (r2 != NULL){
				for (i = 0; i < nbytes; i++) {
					r2[i] ^= region[i];
				}
			} else {
				for (i = 0; i < nbytes; i++) {
					region[i] = 0;
				}
			}
		}

	} else {
		uint8_t prod;
		uint8_t *galois_mult_table;
		int add_mode;

		galois_mult_table = galois_log_table + 256 * 2;
		galois_mult_table += multby * 256;	// Shift mult_table offset by multby

		// Original semantics: r2 == NULL means multiply in place (overwrite).
		add_mode = (add != 0) && (r2 != NULL);
		if (r2 == NULL)
			r2 = region;

		i = 0;
#ifdef PAR3_X86
		if (nbytes >= 64){
			int simd = par3_simd_level();
			if (simd >= 1){
				uint8_t tbl[32];
				int k;
				for (k = 0; k < 16; k++){
					tbl[k]      = galois_mult_table[k];
					tbl[16 + k] = galois_mult_table[k << 4];
				}
				if (simd >= 2){
					i = gf8_region_mul_avx2(tbl, region, r2, nbytes, add_mode);
				} else {
					i = gf8_region_mul_ssse3(tbl, region, r2, nbytes, add_mode);
				}
			}
		}
#endif
		if (add_mode == 0) {
			for (; i < nbytes; i++) {
				prod = galois_mult_table[ region[i] ];
				r2[i] = prod;
			}
		} else {
			for (; i < nbytes; i++) {
				prod = galois_mult_table[ region[i] ];
				r2[i] ^= prod;
			}
		}
	}
}


// Create parity bytes in the region
void gf8_region_create_parity(int prim_poly, uint8_t *buf, size_t region_size)
{
	uint32_t sum, temp, mask;

	prim_poly &= 0xFF;	// reduce to 8-bit value

	// XOR all block data to 4 bytes
	sum = 0;
	while (region_size > 4){
		temp = *((uint32_t *)buf);

		// store highest bits of each 8-bit integer
		mask = (sum & 0x80808080) >> 7;	// 0x01010101 or 0x00000000

		// When SIMD is used, multiple of 2 is faster.
		// previous value multiply by 2
		//sum = (sum & 0x7F7F7F7F) << 1;

		// If multiple of 3 is good, it's possible by XOR to the original value.
		// previous value multiply by 3
		sum ^= (sum & 0x7F7F7F7F) << 1;

		// prim_poly may be 0x1D
		sum ^= mask * prim_poly;	// 0x1D1D1D1D or 0x00000000

	 	// add new 4 bytes
		sum ^= temp;

		region_size -= 4;
		buf += 4;
	}

	((uint32_t *)buf)[0] = sum;
}

// Check parity bytes in the region
int gf8_region_check_parity(int galois_poly, uint8_t *buf, size_t region_size)
{
	uint32_t sum, temp, mask;

	galois_poly &= 0xFF;	// reduce to 8-bit value

	// XOR all block data to 4 bytes
	sum = 0;
	while (region_size > 4){
		temp = *((uint32_t *)buf);

		// store highest bits of each 8-bit integer
		mask = (sum & 0x80808080) >> 7;	// 0x01010101 or 0x00000000

		// previous value multiply by 2
		//sum = (sum & 0x7F7F7F7F) << 1;

		// previous value multiply by 3
		sum ^= (sum & 0x7F7F7F7F) << 1;

		// galois_poly may be 0x1D
		sum ^= mask * galois_poly;	// 0x1D1D1D1D or 0x00000000

	 	// add new 4 bytes
		sum ^= temp;

		region_size -= 4;
		buf += 4;
	}

	// Parity is 4 bytes.
	if (((uint32_t *)buf)[0] != sum)
		return 1;

	return 0;
}

