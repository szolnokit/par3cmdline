#include "sparse.h"

#include "galois.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _MSC_VER
#include <intrin.h>
#endif

/*
 * PCG-XSL-RR (128-bit state, 64-bit output) — Melissa E. O'Neill.
 * Spec: seed in low 64 bits of state; high 64 bits zeroed.
 */
typedef struct {
	uint64_t hi;
	uint64_t lo;
} pcg128_t;

static uint64_t rotr64(uint64_t x, unsigned r)
{
	return (x >> r) | (x << ((0u - r) & 63u));
}

/* 128-bit multiply: (ahi:alo) * (bhi:blo) -> low 128 bits */
static void mul128(uint64_t ahi, uint64_t alo, uint64_t bhi, uint64_t blo,
		uint64_t *rhi, uint64_t *rlo)
{
#ifdef _MSC_VER
	uint64_t p0_hi, p0 = _umul128(alo, blo, &p0_hi);
	uint64_t p1 = alo * bhi;
	uint64_t p2 = ahi * blo;
	*rlo = p0;
	*rhi = p0_hi + p1 + p2;
#else
	__uint128_t a = ((__uint128_t)ahi << 64) | alo;
	__uint128_t b = ((__uint128_t)bhi << 64) | blo;
	__uint128_t p = a * b;
	*rlo = (uint64_t)p;
	*rhi = (uint64_t)(p >> 64);
#endif
}

/* Multiplier and increment from pcg_variants.h (oneseq / default) */
#define PCG_MUL_HI 2549297995355413924ULL
#define PCG_MUL_LO 4865540595714420101ULL
#define PCG_INC_HI 6364136223846793005ULL
#define PCG_INC_LO 1442695040888963407ULL

static void pcg128_step(pcg128_t *rng)
{
	uint64_t mhi, mlo;
	mul128(rng->hi, rng->lo, PCG_MUL_HI, PCG_MUL_LO, &mhi, &mlo);
	/* add increment */
	{
		uint64_t lo = mlo + PCG_INC_LO;
		uint64_t c = (lo < mlo) ? 1 : 0;
		rng->lo = lo;
		rng->hi = mhi + PCG_INC_HI + c;
	}
}

static uint64_t pcg128_next(pcg128_t *rng)
{
	pcg128_step(rng);
	/* XSL RR: rotate64(high ^ low, high >> 58) */
	return rotr64(rng->hi ^ rng->lo, (unsigned)(rng->hi >> 58));
}

static void pcg128_seed(pcg128_t *rng, uint64_t seed)
{
	rng->hi = 0;
	rng->lo = seed;
	/* Advance once so zero seed is not a fixed point of the first output. */
	pcg128_step(rng);
}

static int random_gf_nonzero(pcg128_t *rng, int gf_bytes)
{
	uint64_t mask = (1ULL << (gf_bytes * 8)) - 1ULL;
	for (;;){
		int x = (int)(pcg128_next(rng) & mask);
		if (x != 0)
			return x;
	}
}

/*
 * Compact storage of the sparse generator matrix.
 * Per input block (row): exactly nnz entries of {column index, GF value}.
 * Memory: block_count * nnz * 6 bytes (instead of dense block_count * cols).
 */
typedef struct {
	uint32_t col;
	uint16_t val;
} SPARSE_ENTRY;

void sparse_matrix_free(PAR3_CTX *par3_ctx)
{
	if (par3_ctx->sparse_table){
		free(par3_ctx->sparse_table);
		par3_ctx->sparse_table = NULL;
	}
}

int sparse_matrix_prepare(PAR3_CTX *par3_ctx)
{
	uint8_t *pkt;
	uint16_t *row_buf;	/* reusable dense row (cols elements) */
	uint64_t pkt_size, first, last, max_rec, nnz, seed;
	uint64_t rows, cols, r, c;
	size_t alloc;
	SPARSE_ENTRY *table;
	pcg128_t rng;
	int gf_bytes;

	sparse_matrix_free(par3_ctx);

	if (par3_ctx->matrix_packet == NULL || par3_ctx->matrix_packet_size < 88)
		return RET_LOGIC_ERROR;

	pkt = par3_ctx->matrix_packet + par3_ctx->matrix_packet_offset;
	if ( (memcmp(pkt + 40, "PAR SPA\0", 8) != 0)
			&& (memcmp(pkt + 40, "PAR SPX\0", 8) != 0) ){
		/* During create, only one matrix packet exists at offset 0. */
		pkt = par3_ctx->matrix_packet;
		if ( (memcmp(pkt + 40, "PAR SPA\0", 8) != 0)
				&& (memcmp(pkt + 40, "PAR SPX\0", 8) != 0) )
			return RET_LOGIC_ERROR;
	}
	// The SPX packet type marks the extended block count variant.
	if (memcmp(pkt + 40, "PAR SPX\0", 8) == 0)
		par3_ctx->ecc_extended = 1;

	/* Validate the packet's own size before reading body fields. */
	memcpy(&pkt_size, pkt + 24, 8);
	if (pkt_size < 88){
		printf("Sparse Matrix Packet is too small (%"PRIu64" bytes).\n", pkt_size);
		return RET_LOGIC_ERROR;
	}

	memcpy(&first, pkt + 48, 8);
	memcpy(&last, pkt + 56, 8);
	memcpy(&max_rec, pkt + 64, 8);
	memcpy(&nnz, pkt + 72, 8);
	memcpy(&seed, pkt + 80, 8);

	// first == last == 0 means "every input block".
	// A real range appears with incremental backup (child's own blocks only).
	if (first == 0 && last == 0)
		last = par3_ctx->block_count;
	if (last <= first){
		printf("Sparse Matrix: invalid input block range (%"PRIu64" - %"PRIu64").\n", first, last);
		return RET_LOGIC_ERROR;
	}
	{
		// Extended block count (PAR SPX) lifts the field-size bound;
		// keep a sane implementation limit of 2^24.
		uint64_t limit = (par3_ctx->ecc_extended != 0) ? ((uint64_t)1 << 24) : 65536;

		if (max_rec == 0 || max_rec > limit){
			printf("Sparse Matrix: invalid max recovery count (%"PRIu64").\n", max_rec);
			return RET_LOGIC_ERROR;
		}
		if (nnz == 0 || nnz > max_rec){
			printf("Sparse Matrix: invalid non-zero count (%"PRIu64" of %"PRIu64").\n", nnz, max_rec);
			return RET_LOGIC_ERROR;
		}

		rows = last - first;
		cols = max_rec;
		if (rows == 0 || rows > limit)
			return RET_LOGIC_ERROR;
	}

	// Publish the range for matrix element lookups and decode dispatch.
	par3_ctx->matrix_first_block = first;
	par3_ctx->matrix_last_block = last;

	gf_bytes = par3_ctx->gf_size;
	if (gf_bytes != 1 && gf_bytes != 2){
		printf("Sparse Matrix: Galois field size %u is not supported.\n", par3_ctx->gf_size);
		return RET_LOGIC_ERROR;
	}

	alloc = (size_t)rows * (size_t)nnz * sizeof(SPARSE_ENTRY);
	table = malloc(alloc);
	if (table == NULL){
		perror("Failed to allocate sparse matrix");
		return RET_MEMORY_ERROR;
	}
	row_buf = calloc((size_t)cols, sizeof(uint16_t));
	if (row_buf == NULL){
		perror("Failed to allocate sparse row buffer");
		free(table);
		return RET_MEMORY_ERROR;
	}

	par3_ctx->sparse_table = table;
	par3_ctx->sparse_max_recovery = max_rec;
	par3_ctx->sparse_nnz = nnz;
	par3_ctx->sparse_seed = seed;

	/*
	 * Spec: for each input row (low to high):
	 *  - Place C-X zeros then X random non-zero GF values (left to right).
	 *  - Inside-out Fisher–Yates shuffle from index C-X .. C-1.
	 */
	pcg128_seed(&rng, seed);
	for (r = 0; r < rows; r++){
		uint64_t start = cols - nnz;
		uint64_t i, j, k;

		memset(row_buf, 0, (size_t)cols * sizeof(uint16_t));
		for (c = start; c < cols; c++)
			row_buf[c] = (uint16_t)random_gf_nonzero(&rng, gf_bytes);

		/* Inside-out Fisher–Yates for the non-zero region */
		for (i = start; i < cols; i++){
			uint16_t tmp;
			j = pcg128_next(&rng) % (i + 1);	/* unsigned modulus: 0..i */
			tmp = row_buf[i];
			row_buf[i] = row_buf[j];
			row_buf[j] = tmp;
		}

		/* Extract non-zero entries into compact storage */
		k = 0;
		for (c = 0; c < cols && k < nnz; c++){
			if (row_buf[c] != 0){
				table[r * nnz + k].col = (uint32_t)c;
				table[r * nnz + k].val = row_buf[c];
				k++;
			}
		}
		/* Pad (cannot happen: exactly nnz non-zeros survive the shuffle) */
		for (; k < nnz; k++){
			table[r * nnz + k].col = 0xFFFFFFFF;
			table[r * nnz + k].val = 0;
		}
	}
	free(row_buf);

	if (par3_ctx->noise_level >= 1){
		printf("Sparse matrix: rows=%"PRIu64" cols=%"PRIu64" nnz=%"PRIu64" seed=0x%"PRIx64"\n",
				rows, cols, nnz, seed);
	}
	return 0;
}

// x_index is the ABSOLUTE input block index (matters for incremental backup).
int sparse_matrix_element(PAR3_CTX *par3_ctx, int x_index, int recovery_index)
{
	SPARSE_ENTRY *row;
	uint64_t nnz, k, first, last;

	if (par3_ctx->sparse_table == NULL)
		return 0;
	first = par3_ctx->matrix_first_block;
	last = par3_ctx->matrix_last_block;
	if (last == 0)
		last = par3_ctx->block_count;
	if (x_index < 0 || (uint64_t)x_index < first || (uint64_t)x_index >= last)
		return 0;	// outside the covered range: matrix element is zero
	if (recovery_index < 0 || (uint64_t)recovery_index >= par3_ctx->sparse_max_recovery)
		return 0;

	nnz = par3_ctx->sparse_nnz;
	row = (SPARSE_ENTRY *)par3_ctx->sparse_table + ((uint64_t)x_index - first) * nnz;
	for (k = 0; k < nnz; k++){
		if (row[k].col == (uint32_t)recovery_index)
			return row[k].val;
	}
	return 0;
}

// Collect indexes of all available recovery blocks that belong to the
// currently selected Matrix Packet. Writes up to max_count entries into list.
// Returns the number of entries written.
int sparse_gather_recovery_ids(PAR3_CTX *par3_ctx, int *list, int max_count)
{
	uint8_t *packet_checksum;
	PAR3_PKT_CTX *packet_list;
	uint64_t count, index;
	int id;

	if (par3_ctx->matrix_packet == NULL || par3_ctx->recv_packet_list == NULL)
		return 0;

	packet_checksum = par3_ctx->matrix_packet + par3_ctx->matrix_packet_offset + 8;
	packet_list = par3_ctx->recv_packet_list;
	count = par3_ctx->recv_packet_count;
	id = 0;
	for (index = 0; index < count && id < max_count; index++){
		if (memcmp(packet_list[index].matrix, packet_checksum, 16) == 0){
			list[id] = (int)(packet_list[index].index);
			id++;
		}
	}
	return id;
}


/* ---- Peeling decoder --------------------------------------------------------
 *
 * The sparse generator matrix makes most damage patterns solvable without a
 * dense input-space elimination. Write each candidate recovery block's
 * residual as
 *
 *     r_j = recovery_j XOR sum(available inputs x: S[j][x] * input_x)
 *         = sum(lost inputs i: S[j][i] * lost_i)
 *
 * The residuals are accumulated while streaming the input data once (cost:
 * data * nnz, instead of data * lost_count of the dense path). The small
 * symbolic system S (candidate rows x lost columns, at most nnz entries per
 * column) is then solved by "peeling": a row that touches exactly one lost
 * block yields it directly, substitution removes it from other rows, and the
 * reaction usually solves everything. Whatever remains (the "core") is
 * handled by dense Gauss-Jordan over the residuals -- a lost-space matrix,
 * tiny compared to the input-space matrix of the dense path.
 *
 * The plan phase below is purely symbolic (no block data), so an unsolvable
 * damage pattern is detected before any temporary file is created.
 */

// Rows kept per lost column when choosing candidate recovery rows.
#define PEEL_ROWS_PER_COLUMN 6

void rs_peel_free(PAR3_CTX *par3_ctx)
{
	free(par3_ctx->peel_row_ids);		par3_ctx->peel_row_ids = NULL;
	free(par3_ctx->peel_rowmap);		par3_ctx->peel_rowmap = NULL;
	free(par3_ctx->peel_col_off);		par3_ctx->peel_col_off = NULL;
	free(par3_ctx->peel_col_row);		par3_ctx->peel_col_row = NULL;
	free(par3_ctx->peel_col_val);		par3_ctx->peel_col_val = NULL;
	free(par3_ctx->peel_op_col);		par3_ctx->peel_op_col = NULL;
	free(par3_ctx->peel_op_row);		par3_ctx->peel_op_row = NULL;
	free(par3_ctx->peel_core_cols);		par3_ctx->peel_core_cols = NULL;
	free(par3_ctx->peel_core_rows);		par3_ctx->peel_core_rows = NULL;
	free(par3_ctx->peel_core_inv);		par3_ctx->peel_core_inv = NULL;
	free(par3_ctx->peel_residual);		par3_ctx->peel_residual = NULL;
	par3_ctx->peel_avail = 0;
	par3_ctx->peel_op_count = 0;
	par3_ctx->peel_core_count = 0;
	par3_ctx->peel_core_rows_count = 0;
	par3_ctx->peel_region = 0;
	par3_ctx->use_peel = 0;
}

int rs16_peel_plan(PAR3_CTX *par3_ctx, int lost_count)
{
	uint16_t *gf_table;
	SPARSE_ENTRY *table;
	int *lost_id;
	int avail, avail_limit, i, j, k, e, entry_count;
	int ret = RET_MEMORY_ERROR;
	uint64_t first, last, nnz, max_rec;
	int *row_ids = NULL, *rowmap = NULL;
	int *col_off = NULL, *col_row = NULL;
	uint16_t *col_val = NULL;
	int *row_off = NULL, *row_col = NULL, *row_deg = NULL, *row_fill = NULL;
	uint16_t *row_val = NULL;
	int *queue = NULL, *op_col = NULL, *op_row = NULL, *colpos = NULL;
	uint8_t *solved = NULL;
	uint16_t *aug = NULL;
	int q_head, q_tail, solved_count;

	if ( (par3_ctx->sparse_table == NULL) || (par3_ctx->gf_size != 2) )
		return RET_LOGIC_ERROR;
	gf_table = par3_ctx->galois_table;
	if (gf_table == NULL)
		return RET_LOGIC_ERROR;

	table = (SPARSE_ENTRY *)par3_ctx->sparse_table;
	nnz = par3_ctx->sparse_nnz;
	max_rec = par3_ctx->sparse_max_recovery;
	if (nnz == 0 || max_rec == 0)
		return RET_LOGIC_ERROR;
	lost_id = par3_ctx->recv_id_list + lost_count;

	first = par3_ctx->matrix_first_block;
	last = par3_ctx->matrix_last_block;
	if (last == 0)
		last = par3_ctx->block_count;

	// Lost blocks outside the covered range cannot be repaired by this set.
	for (i = 0; i < lost_count; i++){
		if ( ((uint64_t)lost_id[i] < first) || ((uint64_t)lost_id[i] >= last) ){
			printf("Lost block[%d] is outside the range covered by this PAR3 set (%"PRIu64" - %"PRIu64").\n",
					lost_id[i], first, last - 1);
			printf("Repair that block with the parent backup's PAR3 files.\n");
			return RET_LOGIC_ERROR;
		}
	}

	// Candidate recovery rows. Only rows that touch a lost column carry a
	// non-zero residual, so those are the only useful equations. Every lost
	// column keeps up to PEEL_ROWS_PER_COLUMN of its available rows: enough
	// for the peel reaction to start with high probability, while the
	// residual memory stays proportional to the damage (lost_count * rows per
	// column * block size) instead of the whole recovery set.
	{
		int *all_ids, *col_cnt;
		uint8_t *avail_map;
		int all_cnt;
		uint64_t limit;

		all_ids = malloc(sizeof(int) * max_rec);
		avail_map = calloc((size_t)max_rec, 1);
		col_cnt = calloc(lost_count, sizeof(int));
		rowmap = malloc(sizeof(int) * max_rec);
		if (all_ids == NULL || avail_map == NULL || col_cnt == NULL || rowmap == NULL){
			free(all_ids);
			free(avail_map);
			free(col_cnt);
			goto fail;
		}
		all_cnt = sparse_gather_recovery_ids(par3_ctx, all_ids, (int)max_rec);
		for (j = 0; j < all_cnt; j++){
			if ((uint64_t)all_ids[j] < max_rec)
				avail_map[all_ids[j]] = 1;
		}
		free(all_ids);
		for (i = 0; i < (int)max_rec; i++)
			rowmap[i] = -1;
		if (all_cnt < lost_count){
			printf("Not enough recovery blocks for sparse repair (%d of %d).\n", all_cnt, lost_count);
			free(avail_map);
			free(col_cnt);
			ret = RET_LOGIC_ERROR;
			goto fail;
		}

		// Upper bound on selected rows: every row touching a lost column.
		limit = (uint64_t)lost_count * nnz;
		if (limit > max_rec)
			limit = max_rec;
		avail_limit = (int)limit;
		row_ids = malloc(sizeof(int) * (avail_limit > 0 ? avail_limit : 1));
		if (row_ids == NULL){
			free(avail_map);
			free(col_cnt);
			goto fail;
		}
		avail = 0;

		// Pass 1: give every lost column up to PEEL_ROWS_PER_COLUMN rows.
		for (i = 0; i < lost_count; i++){
			SPARSE_ENTRY *row = table + ((uint64_t)lost_id[i] - first) * nnz;

			for (k = 0; k < (int)nnz; k++){
				uint32_t c = row[k].col;
				if ( (c >= max_rec) || (avail_map[c] == 0) )
					continue;
				if (rowmap[c] >= 0)
					col_cnt[i]++;	// already selected through another column
			}
			for (k = 0; (k < (int)nnz) && (col_cnt[i] < PEEL_ROWS_PER_COLUMN); k++){
				uint32_t c = row[k].col;
				if ( (c >= max_rec) || (avail_map[c] == 0) || (rowmap[c] >= 0) )
					continue;
				rowmap[c] = avail;
				row_ids[avail++] = (int)c;
				col_cnt[i]++;
			}
		}

		// Pass 2: heavy damage packs many lost columns into the same rows;
		// make sure at least lost_count + 64 rows are in play when they exist.
		for (i = 0; (i < lost_count) && (avail < lost_count + 64); i++){
			SPARSE_ENTRY *row = table + ((uint64_t)lost_id[i] - first) * nnz;

			for (k = 0; (k < (int)nnz) && (avail < lost_count + 64); k++){
				uint32_t c = row[k].col;
				if ( (c >= max_rec) || (avail_map[c] == 0) || (rowmap[c] >= 0) )
					continue;
				rowmap[c] = avail;
				row_ids[avail++] = (int)c;
			}
		}
		free(avail_map);
		free(col_cnt);

		if (avail < lost_count){
			printf("Not enough recovery blocks touch the lost blocks (%d rows for %d blocks).\n", avail, lost_count);
			printf("More recovery blocks are required for this damage pattern.\n");
			ret = RET_LOGIC_ERROR;
			goto fail;
		}
	}

	// Symbolic system S: CSR by lost column (at most nnz entries per column).
	col_off = malloc(sizeof(int) * (lost_count + 1));
	col_row = malloc(sizeof(int) * (size_t)lost_count * nnz);
	col_val = malloc(sizeof(uint16_t) * (size_t)lost_count * nnz);
	row_deg = calloc(avail, sizeof(int));
	if (col_off == NULL || col_row == NULL || col_val == NULL || row_deg == NULL)
		goto fail;
	entry_count = 0;
	for (i = 0; i < lost_count; i++){
		SPARSE_ENTRY *row = table + ((uint64_t)lost_id[i] - first) * nnz;
		col_off[i] = entry_count;
		for (k = 0; k < (int)nnz; k++){
			if (row[k].col >= max_rec)
				continue;
			j = rowmap[row[k].col];
			if (j < 0)
				continue;
			col_row[entry_count] = j;
			col_val[entry_count] = row[k].val;
			entry_count++;
			row_deg[j]++;
		}
	}
	col_off[lost_count] = entry_count;

	// Transpose: CSR by candidate row (needed to find the single unsolved
	// column when a row reaches degree 1).
	row_off = malloc(sizeof(int) * (avail + 1));
	row_col = malloc(sizeof(int) * entry_count);
	row_val = malloc(sizeof(uint16_t) * entry_count);
	row_fill = malloc(sizeof(int) * avail);
	if (row_off == NULL || row_col == NULL || row_val == NULL || row_fill == NULL)
		goto fail;
	row_off[0] = 0;
	for (j = 0; j < avail; j++)
		row_off[j + 1] = row_off[j] + row_deg[j];
	memcpy(row_fill, row_off, sizeof(int) * avail);
	for (i = 0; i < lost_count; i++){
		for (e = col_off[i]; e < col_off[i + 1]; e++){
			j = col_row[e];
			row_col[row_fill[j]] = i;
			row_val[row_fill[j]] = col_val[e];
			row_fill[j]++;
		}
	}

	// Peel: rows of degree 1 solve their column; substitution propagates.
	queue = malloc(sizeof(int) * avail);
	op_col = malloc(sizeof(int) * lost_count);
	op_row = malloc(sizeof(int) * lost_count);
	solved = calloc(lost_count, 1);
	if (queue == NULL || op_col == NULL || op_row == NULL || solved == NULL)
		goto fail;
	q_head = 0;
	q_tail = 0;
	for (j = 0; j < avail; j++){
		if (row_deg[j] == 1)
			queue[q_tail++] = j;
	}
	solved_count = 0;
	while (q_head < q_tail){
		int col = -1;

		j = queue[q_head++];
		if (row_deg[j] != 1)
			continue;	// stale entry: the row changed since it was queued
		for (e = row_off[j]; e < row_off[j + 1]; e++){
			if (solved[row_col[e]] == 0){
				col = row_col[e];
				break;
			}
		}
		if (col < 0)
			continue;	// cannot happen: degree says one unsolved column exists
		op_col[solved_count] = col;
		op_row[solved_count] = j;
		solved_count++;
		solved[col] = 1;
		for (e = col_off[col]; e < col_off[col + 1]; e++){
			int j2 = col_row[e];
			row_deg[j2]--;
			if (row_deg[j2] == 1){
				if (q_tail >= avail){	// queue is bounded by rows; recycle space
					memmove(queue, queue + q_head, sizeof(int) * (q_tail - q_head));
					q_tail -= q_head;
					q_head = 0;
				}
				if (q_tail < avail)
					queue[q_tail++] = j2;
			}
		}
	}

	par3_ctx->peel_op_count = solved_count;
	par3_ctx->peel_core_count = 0;
	par3_ctx->peel_core_rows_count = 0;

	// Core: remaining columns solved by dense Gauss-Jordan (lost-space size).
	if (solved_count < lost_count){
		int c_cnt = lost_count - solved_count;
		int r_cnt = 0;
		int p, width;

		colpos = malloc(sizeof(int) * lost_count);
		if (colpos == NULL)
			goto fail;
		par3_ctx->peel_core_cols = malloc(sizeof(int) * c_cnt);
		if (par3_ctx->peel_core_cols == NULL)
			goto fail;
		k = 0;
		for (i = 0; i < lost_count; i++){
			colpos[i] = -1;
			if (solved[i] == 0){
				colpos[i] = k;
				par3_ctx->peel_core_cols[k] = i;
				k++;
			}
		}
		// Rows that still touch an unsolved column.
		for (j = 0; j < avail; j++){
			if (row_deg[j] >= 1)
				r_cnt++;
		}
		if (r_cnt < c_cnt){
			printf("Failed to solve sparse system: %d unsolved blocks, %d usable rows.\n", c_cnt, r_cnt);
			printf("More recovery blocks are required for this damage pattern.\n");
			ret = RET_LOGIC_ERROR;
			goto fail;
		}
		par3_ctx->peel_core_rows = malloc(sizeof(int) * r_cnt);
		if (par3_ctx->peel_core_rows == NULL)
			goto fail;
		k = 0;
		for (j = 0; j < avail; j++){
			if (row_deg[j] >= 1)
				par3_ctx->peel_core_rows[k++] = j;
		}

		// Augmented matrix [A | I] with SIMD region operations on the rows.
		width = c_cnt + r_cnt;
		aug = calloc((size_t)r_cnt * width, sizeof(uint16_t));
		if (aug == NULL)
			goto fail;
		for (k = 0; k < r_cnt; k++){
			j = par3_ctx->peel_core_rows[k];
			for (e = row_off[j]; e < row_off[j + 1]; e++){
				i = row_col[e];
				if (solved[i] == 0)
					aug[(size_t)k * width + colpos[i]] = row_val[e];
			}
			aug[(size_t)k * width + c_cnt + k] = 1;
		}
		for (p = 0; p < c_cnt; p++){
			int r = -1, factor;

			for (k = p; k < r_cnt; k++){
				if (aug[(size_t)k * width + p] != 0){
					r = k;
					break;
				}
			}
			if (r < 0){
				printf("Failed to invert the sparse core at column %d.\n", p);
				printf("More recovery blocks are required for this damage pattern.\n");
				ret = RET_LOGIC_ERROR;
				goto fail;
			}
			if (r != p){
				// Swap only the augmented rows. The identity part keeps the
				// original row order, so peel_core_rows must stay as built.
				for (k = 0; k < width; k++){
					uint16_t tmp = aug[(size_t)p * width + k];
					aug[(size_t)p * width + k] = aug[(size_t)r * width + k];
					aug[(size_t)r * width + k] = tmp;
				}
			}
			factor = gf16_reciprocal(gf_table, aug[(size_t)p * width + p]);
			gf16_region_multiply(gf_table, (uint8_t *)(aug + (size_t)p * width), factor, (size_t)width * 2, NULL, 0);
#ifdef _OPENMP
#pragma omp parallel for if(width >= 256)
#endif
			for (k = 0; k < r_cnt; k++){
				int factor2;

				if (k == p)
					continue;
				factor2 = aug[(size_t)k * width + p];
				if (factor2 == 0)
					continue;
				gf16_region_multiply(gf_table, (uint8_t *)(aug + (size_t)p * width), factor2, (size_t)width * 2, (uint8_t *)(aug + (size_t)k * width), 1);
				aug[(size_t)k * width + p] = 0;
			}
		}
		// Row p now expresses core column p as a combination of core residuals.
		par3_ctx->peel_core_inv = malloc(sizeof(uint16_t) * (size_t)c_cnt * r_cnt);
		if (par3_ctx->peel_core_inv == NULL)
			goto fail;
		for (p = 0; p < c_cnt; p++)
			memcpy(par3_ctx->peel_core_inv + (size_t)p * r_cnt, aug + (size_t)p * width + c_cnt, sizeof(uint16_t) * r_cnt);
		par3_ctx->peel_core_count = c_cnt;
		par3_ctx->peel_core_rows_count = r_cnt;
		free(aug);
		aug = NULL;
		free(colpos);
		colpos = NULL;
	}

	// Publish the plan.
	par3_ctx->peel_row_ids = row_ids;
	par3_ctx->peel_rowmap = rowmap;
	par3_ctx->peel_col_off = col_off;
	par3_ctx->peel_col_row = col_row;
	par3_ctx->peel_col_val = col_val;
	par3_ctx->peel_op_col = op_col;
	par3_ctx->peel_op_row = op_row;
	par3_ctx->peel_avail = avail;

	free(row_off);
	free(row_col);
	free(row_val);
	free(row_fill);
	free(row_deg);
	free(queue);
	free(solved);

	if (par3_ctx->noise_level >= 1){
		printf("Peeling decoder: %d candidate rows, %d peeled, %d in dense core\n",
				avail, par3_ctx->peel_op_count, par3_ctx->peel_core_count);
	}
	return 0;

fail:
	if (ret == RET_MEMORY_ERROR)
		printf("Failed to allocate memory for peeling decoder\n");
	free(row_ids);
	free(rowmap);
	free(col_off);
	free(col_row);
	free(col_val);
	free(row_off);
	free(row_col);
	free(row_val);
	free(row_fill);
	free(row_deg);
	free(queue);
	free(op_col);
	free(op_row);
	free(solved);
	free(colpos);
	free(aug);
	free(par3_ctx->peel_core_cols);
	par3_ctx->peel_core_cols = NULL;
	free(par3_ctx->peel_core_rows);
	par3_ctx->peel_core_rows = NULL;
	free(par3_ctx->peel_core_inv);
	par3_ctx->peel_core_inv = NULL;
	return ret;
}

// Accumulate one available input block (par3_ctx->work_buf) into the
// residuals of the candidate rows it touches (at most nnz region operations).
void rs_peel_input(PAR3_CTX *par3_ctx, int block_index)
{
	SPARSE_ENTRY *row;
	uint16_t *gf_table;
	uint8_t *work_buf, *residual;
	size_t region_size;
	uint64_t first, last, max_rec;
	int k, nnz;

	first = par3_ctx->matrix_first_block;
	last = par3_ctx->matrix_last_block;
	if (last == 0)
		last = par3_ctx->block_count;
	if ( ((uint64_t)block_index < first) || ((uint64_t)block_index >= last) )
		return;	// outside the covered range: all matrix elements are zero

	gf_table = par3_ctx->galois_table;
	work_buf = par3_ctx->work_buf;
	residual = par3_ctx->peel_residual;
	region_size = par3_ctx->peel_region;
	max_rec = par3_ctx->sparse_max_recovery;
	nnz = (int)(par3_ctx->sparse_nnz);
	row = (SPARSE_ENTRY *)par3_ctx->sparse_table + ((uint64_t)block_index - first) * (uint64_t)nnz;

	// Distinct entries touch distinct rows: safe to run in parallel.
#ifdef _OPENMP
#pragma omp parallel for if(region_size >= 16384)
#endif
	for (k = 0; k < nnz; k++){
		int j;

		if (row[k].col >= max_rec)
			continue;
		j = par3_ctx->peel_rowmap[row[k].col];
		if (j < 0)
			continue;
		gf16_region_multiply(gf_table, work_buf, row[k].val, region_size, residual + region_size * (size_t)j, 1);
	}
}

// XOR one read recovery block (par3_ctx->work_buf) into its residual row.
void rs_peel_recovery(PAR3_CTX *par3_ctx, int row_index)
{
	uint32_t *src, *dst;
	size_t count, k;

	src = (uint32_t *)par3_ctx->work_buf;
	dst = (uint32_t *)(par3_ctx->peel_residual + par3_ctx->peel_region * (size_t)row_index);
	count = par3_ctx->peel_region / 4;	// region size is 4-byte aligned
	for (k = 0; k < count; k++)
		dst[k] ^= src[k];
}

// Execute the peel order, then apply the core inverse. Recovered blocks are
// accumulated into par3_ctx->block_data (zero-filled lost_count slots).
int rs_peel_solve(PAR3_CTX *par3_ctx, int lost_count)
{
	uint16_t *gf_table;
	uint8_t *block_data, *residual;
	size_t region_size;
	int *col_off, *col_row;
	uint16_t *col_val;
	int t, e, p;

	(void)lost_count;
	gf_table = par3_ctx->galois_table;
	block_data = par3_ctx->block_data;
	residual = par3_ctx->peel_residual;
	region_size = par3_ctx->peel_region;
	col_off = par3_ctx->peel_col_off;
	col_row = par3_ctx->peel_col_row;
	col_val = par3_ctx->peel_col_val;
	if (gf_table == NULL || block_data == NULL || residual == NULL)
		return RET_LOGIC_ERROR;

	// Peeled blocks, in the order the plan discovered them.
	for (t = 0; t < par3_ctx->peel_op_count; t++){
		int i = par3_ctx->peel_op_col[t];
		int j = par3_ctx->peel_op_row[t];
		int val = 0, factor;
		uint8_t *buf_i = block_data + region_size * (size_t)i;

		for (e = col_off[i]; e < col_off[i + 1]; e++){
			if (col_row[e] == j){
				val = col_val[e];
				break;
			}
		}
		if (val == 0)
			return RET_LOGIC_ERROR;	// cannot happen: entry existed in the plan
		factor = gf16_reciprocal(gf_table, val);
		// The solved block (slot was zero-filled, so add equals overwrite).
		gf16_region_multiply(gf_table, residual + region_size * (size_t)j, factor, region_size, buf_i, 1);
		// Substitute into every other row that touches this column.
		for (e = col_off[i]; e < col_off[i + 1]; e++){
			int j2 = col_row[e];

			if (j2 == j)
				continue;
			gf16_region_multiply(gf_table, buf_i, col_val[e], region_size, residual + region_size * (size_t)j2, 1);
		}
	}

	// Core blocks: dense combination of the core residuals.
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) if(par3_ctx->peel_core_count >= 2)
#endif
	for (p = 0; p < par3_ctx->peel_core_count; p++){
		uint8_t *buf_p = block_data + region_size * (size_t)(par3_ctx->peel_core_cols[p]);
		uint16_t *weights = par3_ctx->peel_core_inv + (size_t)p * par3_ctx->peel_core_rows_count;
		int k;

		for (k = 0; k < par3_ctx->peel_core_rows_count; k++){
			if (weights[k] == 0)
				continue;
			gf16_region_multiply(gf_table, residual + region_size * (size_t)(par3_ctx->peel_core_rows[k]), weights[k], region_size, buf_p, 1);
		}
	}

	return 0;
}
