#include "libpar3.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "galois.h"
#include "hash.h"
#include "reedsolomon.h"
#include "sparse.h"


// Create all recovery blocks from one input block.
void rs_create_one_all(PAR3_CTX *par3_ctx, int x_index)
{
	void *gf_table;
	uint8_t *work_buf, *buf_p;
	uint8_t gf_size;
	int first_num;
	int y_index, x_abs;
	int recovery_block_count;
	size_t region_size;

	recovery_block_count = (int)(par3_ctx->recovery_block_count);
	first_num = (int)(par3_ctx->first_recovery_block);
	gf_size = par3_ctx->gf_size;
	gf_table = par3_ctx->galois_table;
	work_buf = par3_ctx->work_buf;
	buf_p = par3_ctx->block_data;

	// For every recovery block
	region_size = (par3_ctx->block_size + 4 + 3) & ~3;
	if ( (par3_ctx->ecc_method & 2) && (x_index == 0) ){
		/* Sparse rows may start with zeros; clear all recovery buffers first. */
		memset(buf_p, 0, region_size * (size_t)recovery_block_count);
	}
	// Incremental backup: matrix elements use the absolute input block index.
	x_abs = x_index + (int)(par3_ctx->block_index_offset);

	// Each recovery block buffer is independent: parallelize over rows.
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(region_size >= 4096 || recovery_block_count >= 16)
#endif
	for (y_index = 0; y_index < recovery_block_count; y_index++){
		uint8_t *buf_y = buf_p + region_size * (size_t)y_index;
		int element, y_R;

		// Calculate Matrix elements
		if (par3_ctx->ecc_method & 2){	// Sparse Random Matrix
			element = sparse_matrix_element(par3_ctx, x_abs, y_index + first_num);
			if (element != 0){
				if (gf_size == 2)
					gf16_region_multiply(gf_table, work_buf, element, region_size, buf_y, 1);
				else
					gf8_region_multiply(gf_table, work_buf, element, region_size, buf_y, 1);
			}

		} else if (gf_size == 2){	// 16-bit Galois Field (Cauchy)
			y_R = 65535 - (y_index + first_num);
			element = gf16_reciprocal(gf_table, x_abs ^ y_R);	// inv( x_abs ^ y_R )

			// If x_index == 0, just put values.
			// If x_index > 0, add values on previous values.
			gf16_region_multiply(gf_table, work_buf, element, region_size, buf_y, x_index);

		} else {	// 8-bit Galois Field (Cauchy)
			y_R = 255 - (y_index + first_num);
			element = gf8_reciprocal(gf_table, x_abs ^ y_R);	// inv( x_abs ^ y_R )

			// If x_index == 0, just put values.
			// If x_index > 0, add values on previous values.
			gf8_region_multiply(gf_table, work_buf, element, region_size, buf_y, x_index);
		}
		//printf("x = %d, R = %d, y_R = %d, element = %d\n", x_abs, y_index + first_num, y_R, element);
	}
}

// Create all recovery blocks from all input blocks.
void rs_create_all(PAR3_CTX *par3_ctx, size_t region_size, uint64_t progress_total, uint64_t progress_step)
{
	void *gf_table;
	uint8_t *block_data, *recv_base;
	uint8_t gf_size;
	int first_num;
	int y_index;
	int block_count, recovery_block_count;
	int progress_old, show_progress;
	time_t time_old;

	block_count = (int)(par3_ctx->block_count);
	recovery_block_count = (int)(par3_ctx->recovery_block_count);
	first_num = (int)(par3_ctx->first_recovery_block);
	gf_size = par3_ctx->gf_size;
	gf_table = par3_ctx->galois_table;
	block_data = par3_ctx->block_data;
	recv_base = block_data + region_size * block_count;

	show_progress = ( (par3_ctx->noise_level >= 0) && (par3_ctx->noise_level <= 1) );
	progress_old = 0;
	time_old = time(NULL);

	// Each recovery block is an independent linear combination of the
	// input blocks: parallelize over recovery rows.
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
	for (y_index = 0; y_index < recovery_block_count; y_index++){
		uint8_t *input_p = block_data;
		uint8_t *recv_p = recv_base + region_size * (size_t)y_index;
		int x_index, y_R, element;

		if (par3_ctx->ecc_method & 2)
			memset(recv_p, 0, region_size);

		// For every input block
		for (x_index = 0; x_index < block_count; x_index++){
			// Incremental backup: matrix elements use the absolute input block index.
			int x_abs = x_index + (int)(par3_ctx->block_index_offset);

			// Calculate Matrix elements
			if (par3_ctx->ecc_method & 2){	// Sparse Random Matrix
				element = sparse_matrix_element(par3_ctx, x_abs, y_index + first_num);
				if (element != 0){
					if (gf_size == 2)
						gf16_region_multiply(gf_table, input_p, element, region_size, recv_p, 1);
					else
						gf8_region_multiply(gf_table, input_p, element, region_size, recv_p, 1);
				}

			} else if (par3_ctx->gf_size == 2){	// 16-bit Galois Field
				y_R = 65535 - (y_index + first_num);
				element = gf16_reciprocal(gf_table, x_abs ^ y_R);	// inv( x_abs ^ y_R )

				// If x_index == 0, just put values.
				// If x_index > 0, add values on previous values.
				gf16_region_multiply(gf_table, input_p, element, region_size, recv_p, x_index);

			} else {	// 8-bit Galois Field
				y_R = 255 - (y_index + first_num);
				element = gf8_reciprocal(gf_table, x_abs ^ y_R);	// inv( x_abs ^ y_R )

				// If x_index == 0, just put values.
				// If x_index > 0, add values on previous values.
				gf8_region_multiply(gf_table, input_p, element, region_size, recv_p, x_index);
			}
			//printf("x = %d, R = %d, y_R = %d, element = %d\n", x_abs, y_index + first_num, y_R, element);

			input_p += region_size;
		}

		// Print progress percent
		if (show_progress){
#ifdef _OPENMP
#pragma omp critical (rs_progress)
#endif
			{
				time_t time_now;
				int progress_now;

				progress_step += block_count;
				time_now = time(NULL);
				if (time_now != time_old){
					time_old = time_now;
					progress_now = (int)((progress_step * 1000) / progress_total);
					if (progress_now != progress_old){
						progress_old = progress_now;
						printf("%d.%d%%\r", progress_now / 10, progress_now % 10);	// 0.0% ~ 100.0%
					}
				}
			}
		}
	}
}


// Construct matrix for Cauchy Reed-Solomon, and solve linear equation.
int rs_compute_matrix(PAR3_CTX *par3_ctx, uint64_t lost_count)
{
	int ret, ranged;
	size_t alloc_size, region_size;

	// Only when it uses Reed-Solomon Erasure Codes (Cauchy or Sparse).
	if ((par3_ctx->ecc_method & 3) == 0)
		return RET_LOGIC_ERROR;

	if (par3_ctx->ecc_method & 2){
		ret = sparse_matrix_prepare(par3_ctx);
		if (ret != 0)
			return ret;
	}

	if (par3_ctx->gf_size == 2){	// 16-bit Galois Field
		par3_ctx->galois_table = gf16_create_table(par3_ctx->galois_poly);

	} else if (par3_ctx->gf_size == 1){	// 8-bit Galois Field
		par3_ctx->galois_table = gf8_create_table(par3_ctx->galois_poly);

	} else {
		printf("Galois Field (0x%X) isn't supported.\n", par3_ctx->galois_poly);
		return RET_LOGIC_ERROR;
	}
	if (par3_ctx->galois_table == NULL){
		printf("Failed to create tables for Galois Field (0x%X)\n", par3_ctx->galois_poly);
		return RET_MEMORY_ERROR;
	}

	// Is the matrix restricted to a range of input blocks? (incremental backup)
	ranged = 0;
	if ( (par3_ctx->matrix_first_block != 0)
			|| ( (par3_ctx->matrix_last_block != 0) && (par3_ctx->matrix_last_block < par3_ctx->block_count) ) )
		ranged = 1;

	// Set memory alignment of block data to be 4.
	// Increase at least 1 byte as checksum.
	region_size = (par3_ctx->block_size + 4 + 3) & ~3;

	// Try the peeling decoder for sparse codes first: it avoids the dense
	// input-space elimination entirely, which makes repair time proportional
	// to the damage instead of the whole block count.
	if ( (par3_ctx->ecc_method & 2) && (par3_ctx->gf_size == 2) ){
		ret = rs16_peel_plan(par3_ctx, (int)lost_count);
		if (ret != 0)
			return ret;	// unsolvable damage pattern (or out of memory)

		alloc_size = region_size * lost_count;
		if ( (par3_ctx->memory_limit == 0)
				|| (alloc_size + region_size * (size_t)(par3_ctx->peel_avail) <= par3_ctx->memory_limit) ){
			par3_ctx->block_data = malloc(alloc_size);
			if (par3_ctx->block_data != NULL){
				par3_ctx->peel_residual = malloc(region_size * (size_t)(par3_ctx->peel_avail));
				if (par3_ctx->peel_residual != NULL){
					memset(par3_ctx->peel_residual, 0, region_size * (size_t)(par3_ctx->peel_avail));
					par3_ctx->peel_region = region_size;
					par3_ctx->use_peel = 1;
					par3_ctx->ecc_method |= 0x8000;	// Keep all lost blocks on memory
					if (par3_ctx->noise_level >= 2){
						printf("\nAligned size of block data = %zu\n", region_size);
						printf("Keep all lost blocks and residuals on memory (%zu * %"PRIu64" + %zu * %d)\n",
								region_size, lost_count, region_size, par3_ctx->peel_avail);
					}
					return 0;
				}
				free(par3_ctx->block_data);
				par3_ctx->block_data = NULL;
			}
		}
		// Not enough memory for the residual buffers: fall back to the dense
		// matrix path (streaming or split method).
		rs_peel_free(par3_ctx);
	}

	// Make matrix
	if (par3_ctx->gf_size == 2){	// 16-bit Reed-Solomon Codes
		if ( (par3_ctx->ecc_method & 2) || (ranged != 0) )
			ret = rs16_gaussian_elimination(par3_ctx, (int)lost_count);
		else
			ret = rs16_invert_matrix_cauchy(par3_ctx, (int)lost_count);
		if (ret != 0)
			return ret;

	} else if (par3_ctx->gf_size == 1){	// 8-bit Reed-Solomon Codes
		ret = rs8_gaussian_elimination(par3_ctx, (int)lost_count);
		if (ret != 0)
			return ret;
	}

	// Limited memory usage
	alloc_size = region_size * lost_count;
	if ( (par3_ctx->memory_limit > 0) && (alloc_size > par3_ctx->memory_limit) )
		return 0;

	// Allocate memory to keep lost blocks
	par3_ctx->block_data = malloc(alloc_size);
	//par3_ctx->block_data = NULL;	// For testing another method
	if (par3_ctx->block_data != NULL){
		par3_ctx->ecc_method |= 0x8000;	// Keep all lost blocks on memory
		if (par3_ctx->noise_level >= 2){
			printf("\nAligned size of block data = %zu\n", region_size);
			printf("Keep all lost blocks on memory (%zu * %"PRIu64" = %zu)\n", region_size, lost_count, alloc_size);
		}
	}

	return 0;
}

// Recover all lost input blocks from one block.
void rs_recover_one_all(PAR3_CTX *par3_ctx, int x_index, int lost_count)
{
	void *gf_table, *matrix;
	uint8_t *work_buf, *buf_p;
	uint8_t gf_size;
	int y_index;
	size_t block_count;	// size_t: index math must not overflow with extended block count
	size_t region_size;

	block_count = (size_t)(par3_ctx->block_count);
	gf_size = par3_ctx->gf_size;
	gf_table = par3_ctx->galois_table;
	matrix = par3_ctx->matrix;
	work_buf = par3_ctx->work_buf;
	buf_p = par3_ctx->block_data;

	// For every lost block (each lost block buffer is independent)
	region_size = (par3_ctx->block_size + 4 + 3) & ~3;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(region_size >= 4096 || lost_count >= 16)
#endif
	for (y_index = 0; y_index < lost_count; y_index++){
		uint8_t *buf_y = buf_p + region_size * (size_t)y_index;
		int factor;

		if (gf_size == 2){
			factor = ((uint16_t *)matrix)[ block_count * y_index + x_index ];
			gf16_region_multiply(gf_table, work_buf, factor, region_size, buf_y, 1);
		} else {
			factor = ((uint8_t *)matrix)[ block_count * y_index + x_index ];
			gf8_region_multiply(gf_table, work_buf, factor, region_size, buf_y, 1);
		}
		//printf("%d-th lost block += input block[%d] * %2x\n", y_index, x_index, factor);
	}
}

// Recover all lost input blocks from all blocks.
void rs_recover_all(PAR3_CTX *par3_ctx, size_t region_size, int lost_count, uint64_t progress_total, uint64_t progress_step)
{
	void *gf_table, *matrix;
	uint8_t *block_data;
	uint8_t gf_size;
	int *lost_id;
	int y_index;
	size_t block_count;	// size_t: index math must not overflow with extended block count
	int progress_old, show_progress;
	time_t time_old;

	block_count = (size_t)(par3_ctx->block_count);
	gf_size = par3_ctx->gf_size;
	gf_table = par3_ctx->galois_table;
	matrix = par3_ctx->matrix;
	lost_id = par3_ctx->recv_id_list + lost_count;
	block_data = par3_ctx->block_data;

	show_progress = ( (par3_ctx->noise_level >= 0) && (par3_ctx->noise_level <= 1) );
	progress_old = 0;
	time_old = time(NULL);

	// Each lost block is recovered independently: parallelize over rows.
	// Readers skip every lost block slot, so concurrent writes into
	// distinct lost slots are safe.
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
	for (y_index = 0; y_index < lost_count; y_index++){
		uint8_t *buf_p = block_data + region_size * lost_id[y_index];
		uint8_t *input_p = block_data;
		int x_index, lost_index, factor;

		// For every available input block
		lost_index = 0;
		for (x_index = 0; x_index < block_count; x_index++){
			if ( (lost_index < lost_count) && (x_index == lost_id[lost_index]) ){
				lost_index++;
				input_p += region_size;
				continue;
			}

			if (gf_size == 2){
				factor = ((uint16_t *)matrix)[ block_count * y_index + x_index ];
				gf16_region_multiply(gf_table, input_p, factor, region_size, buf_p, 1);
			} else {
				factor = ((uint8_t *)matrix)[ block_count * y_index + x_index ];
				gf8_region_multiply(gf_table, input_p, factor, region_size, buf_p, 1);
			}

			input_p += region_size;
		}

		// For every using recovery block
		for (lost_index = 0; lost_index < lost_count; lost_index++){
			x_index = lost_id[lost_index];

			if (gf_size == 2){
				factor = ((uint16_t *)matrix)[ block_count * y_index + x_index ];
				gf16_region_multiply(gf_table, input_p, factor, region_size, buf_p, 1);
			} else {
				factor = ((uint8_t *)matrix)[ block_count * y_index + x_index ];
				gf8_region_multiply(gf_table, input_p, factor, region_size, buf_p, 1);
			}

			input_p += region_size;
		}

		// Print progress percent
		if (show_progress){
#ifdef _OPENMP
#pragma omp critical (rs_progress)
#endif
			{
				time_t time_now;
				int progress_now;

				progress_step += block_count;
				time_now = time(NULL);
				if (time_now != time_old){
					time_old = time_now;
					progress_now = (int)((progress_step * 1000) / progress_total);
					if (progress_now != progress_old){
						progress_old = progress_now;
						printf("%d.%d%%\r", progress_now / 10, progress_now % 10);	// 0.0% ~ 100.0%
					}
				}
			}
		}
	}
}

