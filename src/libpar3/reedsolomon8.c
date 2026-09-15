#include "libpar3.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "galois.h"
#include "sparse.h"


// Gaussian elimination for Sparse Random Matrix.
// Uses every available recovery block as a candidate row, so repair succeeds
// whenever any invertible subset exists (not only the first lost_count blocks).
static int rs8_gaussian_sparse(PAR3_CTX *par3_ctx, int lost_count)
{
	uint8_t *gf_table, *matrix;
	int x, y, y2, r;
	int *lost_id, *recv_id, *all_ids;
	size_t block_count;	// size_t: index math must not overflow with extended block count
	int avail;
	int pivot, factor, factor2;

	block_count = (size_t)(par3_ctx->block_count);
	gf_table = par3_ctx->galois_table;
	recv_id = par3_ctx->recv_id_list;
	lost_id = recv_id + lost_count;

	// Gather available recovery blocks for this Matrix Packet.
	// Spare rows beyond lost_count let elimination sidestep singular subsets.
	// The margin must scale with the matrix density: a lost column is touched
	// by a candidate row with probability nnz / max_recovery, so we need about
	// (8 * max_recovery / nnz) rows for every column to be covered well.
	{
		uint64_t margin = 64;
		if (par3_ctx->sparse_nnz > 0){
			uint64_t need = (8 * par3_ctx->sparse_max_recovery) / par3_ctx->sparse_nnz;
			if (need > margin)
				margin = need;
		}
		avail = (int)(par3_ctx->sparse_max_recovery);
		if ((uint64_t)avail > (uint64_t)lost_count + margin)
			avail = lost_count + (int)margin;
	}
	all_ids = malloc(sizeof(int) * avail);
	if (all_ids == NULL){
		printf("Failed to allocate memory for recovery id list\n");
		return RET_MEMORY_ERROR;
	}
	avail = sparse_gather_recovery_ids(par3_ctx, all_ids, avail);
	if (avail < lost_count){
		printf("Not enough recovery blocks for sparse repair (%d of %d).\n", avail, lost_count);
		free(all_ids);
		return RET_LOGIC_ERROR;
	}

	// Lost blocks outside the covered range cannot be repaired by this set.
	{
		uint64_t range_first = par3_ctx->matrix_first_block;
		uint64_t range_last = par3_ctx->matrix_last_block;
		if (range_last == 0)
			range_last = par3_ctx->block_count;
		for (y = 0; y < lost_count; y++){
			if ( ((uint64_t)lost_id[y] < range_first) || ((uint64_t)lost_id[y] >= range_last) ){
				printf("Lost block[%d] is outside the range covered by this PAR3 set (%"PRIu64" - %"PRIu64").\n",
						lost_id[y], range_first, range_last - 1);
				printf("Repair that block with the parent backup's PAR3 files.\n");
				free(all_ids);
				return RET_LOGIC_ERROR;
			}
		}
	}

	// Allocate matrix with a row per candidate recovery block.
	matrix = malloc((size_t)block_count * avail);
	if (matrix == NULL){
		printf("Failed to allocate memory for matrix\n");
		free(all_ids);
		return RET_MEMORY_ERROR;
	}
	par3_ctx->matrix = matrix;

	for (y = 0; y < avail; y++){
		for (x = 0; x < block_count; x++){
			matrix[block_count * y + x] = (uint8_t)sparse_matrix_element(par3_ctx, x, all_ids[y]);
		}
	}

	// Gauss-Jordan elimination with row pivoting over all candidate rows.
	for (y = 0; y < lost_count; y++){
		pivot = lost_id[y];

		// Find a row with non-zero value in the pivot column.
		r = -1;
		for (y2 = y; y2 < avail; y2++){
			if (matrix[block_count * y2 + pivot] != 0){
				r = y2;
				break;
			}
		}
		if (r < 0){
			printf("Failed to invert sparse matrix at lost block[%d].\n", pivot);
			printf("More recovery blocks are required for this damage pattern.\n");
			free(all_ids);
			return RET_LOGIC_ERROR;
		}
		if (r != y){	// Swap rows and their recovery ids.
			int tmp_id = all_ids[y];
			all_ids[y] = all_ids[r];
			all_ids[r] = tmp_id;
			for (x = 0; x < block_count; x++){
				uint8_t tmp = matrix[block_count * y + x];
				matrix[block_count * y + x] = matrix[block_count * r + x];
				matrix[block_count * r + x] = tmp;
			}
		}

		// Let pivot value be 1.
		factor = gf8_reciprocal(gf_table, matrix[block_count * y + pivot]);
		gf8_region_multiply(gf_table, matrix + block_count * y, factor, block_count, NULL, 0);

		// Erase values of same pivot on all other rows (selected and candidate).
		// Rows are independent of each other: parallelize.
#ifdef _OPENMP
#pragma omp parallel for private(factor2) if(block_count >= 256)
#endif
		for (y2 = 0; y2 < avail; y2++){
			if (y2 == y)
				continue;

			factor2 = matrix[block_count * y2 + pivot];
			gf8_region_multiply(gf_table, matrix + block_count * y, factor2, block_count, matrix + block_count * y2, 1);

			// After eliminating the pivot value, store "factor * factor2" on the pivot.
			matrix[block_count * y2 + pivot] = gf8_multiply(gf_table, factor, factor2);
		}

		// After eliminating the pivot column, store "factor" on the pivot.
		matrix[block_count * y + pivot] = factor;
	}

	// Publish which recovery blocks were selected.
	for (y = 0; y < lost_count; y++)
		recv_id[y] = all_ids[y];
	free(all_ids);

	if (par3_ctx->noise_level >= 3){
		printf("\n recovery matrix (%d * %d):\n", block_count, lost_count);
		for (y = 0; y < lost_count; y++){
			printf("recv%3d -> lost%3d =", recv_id[y], lost_id[y]);
			for (x = 0; x < block_count; x++){
				printf(" %2x", matrix[block_count * y + x]);
			}
			printf("\n");
		}
	}

	return 0;
}

// Gaussian elimination of matrix for Cauchy / Sparse Reed-Solomon
int rs8_gaussian_elimination(PAR3_CTX *par3_ctx, int lost_count)
{
	uint8_t *gf_table, *matrix;
	int x, y, y_R, y2;
	int *lost_id, *recv_id;
	size_t block_count;	// size_t: index math must not overflow with extended block count
	int pivot, factor, factor2;
	int range_first, range_last;

	if (lost_count == 0)
		return 0;

	if (par3_ctx->ecc_method & 2)	// Sparse Random Matrix
		return rs8_gaussian_sparse(par3_ctx, lost_count);

	block_count = (size_t)(par3_ctx->block_count);
	gf_table = par3_ctx->galois_table;
	recv_id = par3_ctx->recv_id_list;
	lost_id = recv_id + lost_count;

	// Allocate matrix on memory
	matrix = malloc(block_count * lost_count);
	if (matrix == NULL){
		printf("Failed to allocate memory for matrix\n");
		return RET_MEMORY_ERROR;
	}
	par3_ctx->matrix = matrix;

	// Range of input blocks covered by the Matrix Packet (incremental backup).
	// Blocks outside the range have zero matrix elements.
	range_first = 0;
	range_last = block_count;
	if (par3_ctx->matrix_last_block != 0){
		range_first = (int)(par3_ctx->matrix_first_block);
		range_last = (int)(par3_ctx->matrix_last_block);
		if (range_last > block_count)
			range_last = block_count;
	}
	for (y = 0; y < lost_count; y++){
		if ( (lost_id[y] < range_first) || (lost_id[y] >= range_last) ){
			printf("Lost block[%d] is outside the range covered by this PAR3 set (%d - %d).\n",
					lost_id[y], range_first, range_last - 1);
			printf("Repair that block with the parent backup's PAR3 files.\n");
			return RET_LOGIC_ERROR;
		}
	}

	// Set matrix elements
	for (y = 0; y < lost_count; y++){	// per each recovery block
		// These are elements of generator matrix.
		y_R = 255 - recv_id[y];	// y_R = MAX - y_index
		for (x = 0; x < block_count; x++){
			if ( (x < range_first) || (x >= range_last) ){
				matrix[block_count * y + x] = 0;	// outside the covered range
			} else {
				// inv( x_index ^ y_R )
				matrix[block_count * y + x] = gf8_reciprocal(gf_table, x ^ y_R);
			}
		}

		// No need to set values for recovery blocks,
		// because they will be put in positions of lost blocks.
	}

	if (par3_ctx->noise_level >= 3){
		printf("\n generator matrix (%d * %d):\n", block_count, lost_count);
		for (y = 0; y < lost_count; y++){
			printf("lost%3d <- recv%3d =", lost_id[y], recv_id[y]);
			for (x = 0; x < block_count; x++){
				printf(" %2x", matrix[block_count * y + x]);
			}
			printf("\n");
		}
	}

	// Gaussian elimination
	for (y = 0; y < lost_count; y++){
		// Let pivot value to be 1.
		pivot = lost_id[y];
		factor = matrix[block_count * y + pivot];
		if (factor == 0){
			printf("Failed to invert matrix\n");
			return RET_LOGIC_ERROR;
		}
		factor = gf8_reciprocal(gf_table, factor);
		gf8_region_multiply(gf_table, matrix + block_count * y, factor, block_count, NULL, 0);

		// Erase values of same pivot on other rows.
		// Rows are independent of each other: parallelize.
#ifdef _OPENMP
#pragma omp parallel for private(factor2) if(block_count >= 256)
#endif
		for (y2 = 0; y2 < lost_count; y2++){
			if (y2 == y)
				continue;

			factor2 = matrix[block_count * y2 + pivot];
			gf8_region_multiply(gf_table, matrix + block_count * y, factor2, block_count, matrix + block_count * y2, 1);

			// After eliminate the pivot value, store "factor * factor2" value on the pivot.
			matrix[block_count * y2 + pivot] = gf8_multiply(gf_table, factor, factor2);
		}

		// After eliminate the pivot columun, store "factor" value on the pivot.
		matrix[block_count * y + pivot] = factor;
	}

	if (par3_ctx->noise_level >= 3){
		printf("\n recovery matrix (%d * %d):\n", block_count, lost_count);
		for (y = 0; y < lost_count; y++){
			printf("recv%3d -> lost%3d =", recv_id[y], lost_id[y]);
			for (x = 0; x < block_count; x++){
				printf(" %2x", matrix[block_count * y + x]);
			}
			printf("\n");
		}
	}

	return 0;
}


/*

Fast inversion for Cauchy matrix
This method is based on sample code of persicum's RSC32.

The inverting theory may be described in these pages;

Cauchy matrix
https://en.wikipedia.org/wiki/Cauchy_matrix

Inverse of Cauchy Matrix
https://proofwiki.org/wiki/Inverse_of_Cauchy_Matrix

*/
int rs8_invert_matrix_cauchy(PAR3_CTX *par3_ctx, int lost_count)
{
	uint8_t *gf_table, *matrix;
	int *x, *y, *a, *b, *c, *d;
	int i, j, k;
	int *lost_id, *recv_id;
	size_t block_count;	// size_t: index math must not overflow

	if (lost_count == 0)
		return 0;

	block_count = (size_t)(par3_ctx->block_count);
	gf_table = par3_ctx->galois_table;
	recv_id = par3_ctx->recv_id_list;
	lost_id = recv_id + lost_count;

	// Allocate matrix on memory
	matrix = malloc(block_count * lost_count);
	if (matrix == NULL){
		printf("Failed to allocate memory for matrix\n");
		return RET_MEMORY_ERROR;
	}
	par3_ctx->matrix = matrix;

	// Allocate working buffer on memory
	a = calloc(block_count * 6, sizeof(int));
	if (a == NULL){
		printf("Failed to allocate memory for inversion\n");
		return RET_MEMORY_ERROR;
	}
	b = a + block_count;
	c = b + block_count;
	d = c + block_count;
	x = d + block_count;
	y = x + block_count;

	// Set index of lost input blocks
	for (i = 0; i < lost_count; i++){
		y[i] = lost_id[i];
	}
	// Set index of existing input blocks after
	k = 0;
	for (j = 0; j < block_count; j++){
		if (k < lost_count && j == lost_id[k]){
			k++;
			continue;
		}
		y[i] = j;
		i++;
	}

	// Set index of using recovery blocks
	for (i = 0; i < lost_count; i++){
		x[i] = 255 - recv_id[i];	// y_R = MAX - y_index
	}
	// Set index of existing input blocks after
	for (; i < block_count; i++){
		x[i] = y[i];
	}

	for (i = 0; i < block_count; i++){
		a[i] = 1;
		b[i] = 1;
		c[i] = 1;
		d[i] = 1;

		for (j = 0; j < lost_count; j++){
			if (i != j){
				a[i] = gf8_multiply(gf_table, a[i], x[i] ^ x[j]);
				b[i] = gf8_multiply(gf_table, b[i], y[i] ^ y[j]);
			}

			c[i] = gf8_multiply(gf_table, c[i], x[i] ^ y[j]);
			d[i] = gf8_multiply(gf_table, d[i], y[i] ^ x[j]);
		}
	}

/*
	if (par3_ctx->noise_level >= 3){
		printf("\n fast inversion (%d * 6):\n", block_count);
		printf("y =");
		for (i = 0; i < block_count; i++){
			printf(" %2x", y[i]);
		}
		printf("\n");
		printf("x =");
		for (i = 0; i < block_count; i++){
			printf(" %2x", x[i]);
		}
		printf("\n");
		printf("a =");
		for (i = 0; i < block_count; i++){
			printf(" %2x", a[i]);
		}
		printf("\n");
		printf("b =");
		for (i = 0; i < block_count; i++){
			printf(" %2x", b[i]);
		}
		printf("\n");
		printf("c =");
		for (i = 0; i < block_count; i++){
			printf(" %2x", c[i]);
		}
		printf("\n");
		printf("d =");
		for (i = 0; i < block_count; i++){
			printf(" %2x", d[i]);
		}
		printf("\n");
	}
*/

	for (i = 0; i < lost_count; i++){
		for (j = 0; j < block_count; j++){
			k = gf8_multiply(gf_table, a[j], b[i]);
			k = gf8_reciprocal(gf_table, gf8_multiply(gf_table, k, x[j] ^ y[i]));
			k = gf8_multiply(gf_table, gf8_multiply(gf_table, c[j], d[i]), k);
			matrix[ block_count * i + y[j] ] = k;
		}
	}

	if (par3_ctx->noise_level >= 3){
		printf("\n recovery matrix (%d * %d):\n", block_count, lost_count);
		for (i = 0; i < lost_count; i++){
			printf("recv%3d -> lost%3d =", recv_id[i], lost_id[i]);
			for (j = 0; j < block_count; j++){
				printf(" %2x", matrix[block_count * i + j]);
			}
			printf("\n");
		}
	}

	// Deallocate working buffer
	free(a);

	return 0;
}

