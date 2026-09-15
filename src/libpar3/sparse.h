#ifndef __SPARSE_H__
#define __SPARSE_H__

#include "libpar3.h"

#include <stdint.h>

// Build or rebuild the sparse generator matrix from ctx / Matrix Packet fields.
int sparse_matrix_prepare(PAR3_CTX *par3_ctx);

// Element at input block x, recovery block index r (absolute index, not offset by first_recovery).
int sparse_matrix_element(PAR3_CTX *par3_ctx, int x_index, int recovery_index);

// Collect all recovery block indexes for the selected Matrix Packet (returns count).
int sparse_gather_recovery_ids(PAR3_CTX *par3_ctx, int *list, int max_count);

void sparse_matrix_free(PAR3_CTX *par3_ctx);

// ---- Peeling decoder for sparse codes (16-bit Galois field) ----------------
// Plan phase (symbolic, no block data): choose candidate recovery rows, decide
// the peel order and invert the small dense core. Returns 0 when the damage
// pattern is solvable.
int rs16_peel_plan(PAR3_CTX *par3_ctx, int lost_count);

// Streaming phase: accumulate one available input block (par3_ctx->work_buf)
// into the residual rows it touches, or XOR one read recovery block into its
// own residual row.
void rs_peel_input(PAR3_CTX *par3_ctx, int block_index);
void rs_peel_recovery(PAR3_CTX *par3_ctx, int row_index);

// Solve phase: execute the peel order and apply the core inverse; recovered
// blocks are written into par3_ctx->block_data (lost_count slots).
int rs_peel_solve(PAR3_CTX *par3_ctx, int lost_count);

void rs_peel_free(PAR3_CTX *par3_ctx);

#endif
