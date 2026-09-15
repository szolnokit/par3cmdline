#include "libpar3.h"

#include "common.h"

#include "../blake3/blake3.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "map.h"
#include "packet.h"
#include "write.h"
#include "block.h"
#include "read.h"


// Copy a packet buffer from the parent context (returns 0 on success).
static int copy_parent_buffer(uint8_t *src, size_t src_size, uint32_t src_count,
		uint8_t **dst, size_t *dst_size, uint32_t *dst_count)
{
	if (src == NULL || src_size == 0)
		return 0;
	*dst = malloc(src_size);
	if (*dst == NULL){
		perror("Failed to allocate memory for parent packets");
		return RET_MEMORY_ERROR;
	}
	memcpy(*dst, src, src_size);
	*dst_size = src_size;
	*dst_count = src_count;
	return 0;
}

// Load an existing PAR3 set as the parent of an incremental backup.
int load_parent_backup(PAR3_CTX *par3_ctx, const char *parent_path)
{
	PAR3_CTX parent;
	int ret;
	uint64_t packet_size;

	if ( (parent_path == NULL) || (parent_path[0] == 0) )
		return RET_INVALID_COMMAND;

	memset(&parent, 0, sizeof(parent));
	parent.noise_level = -1;
	if (path_copy(parent.par_filename, (char *)parent_path, _MAX_PATH - 8) == 0){
		printf("Parent PAR filename is too long.\n");
		return RET_INVALID_COMMAND;
	}

	ret = par_search(&parent, parent.par_filename, 0);
	if (ret != 0){
		printf("Failed to search parent PAR3: %s\n", parent_path);
		par3_release(&parent);
		return ret;
	}
	ret = read_packet(&parent);
	if (ret != 0){
		printf("Failed to read parent PAR3: %s\n", parent_path);
		par3_release(&parent);
		return ret;
	}
	ret = parse_vital_packet(&parent);
	if (ret != 0){
		printf("Failed to parse parent PAR3: %s\n", parent_path);
		par3_release(&parent);
		return ret;
	}

	memcpy(par3_ctx->parent_set_id, parent.set_id, 8);
	if (parent.root_packet == NULL){
		printf("Parent PAR3 has no Root Packet.\n");
		par3_release(&parent);
		return RET_INSUFFICIENT_DATA;
	}
	memcpy(&packet_size, parent.root_packet + 24, 8);
	if (packet_size < 48){
		par3_release(&parent);
		return RET_LOGIC_ERROR;
	}
	memcpy(par3_ctx->parent_root_hash, parent.root_packet + 8, 16);

	/* Child must use the same block size and Galois field as the parent. */
	if (par3_ctx->block_size == 0){
		par3_ctx->block_size = parent.block_size;
	} else if (par3_ctx->block_size != parent.block_size){
		printf("Block size must match the parent (%"PRIu64").\n", parent.block_size);
		par3_release(&parent);
		return RET_INVALID_COMMAND;
	}
	par3_ctx->parent_gf_size = parent.gf_size;
	par3_ctx->parent_galois_poly = parent.galois_poly;

	path_copy(par3_ctx->parent_filename, (char *)parent_path, _MAX_PATH - 8);

	// Retain the parent's vital packets. They are copied into the child's
	// PAR3 files so the child set is self-contained for verification.
	par3_ctx->parent_block_count = parent.block_count;
	ret = copy_parent_buffer(parent.start_packet, parent.start_packet_size, parent.start_packet_count,
			&(par3_ctx->parent_start_packet), &(par3_ctx->parent_start_packet_size), &(par3_ctx->parent_start_packet_count));
	if (ret == 0)
		ret = copy_parent_buffer(parent.file_packet, parent.file_packet_size, parent.file_packet_count,
				&(par3_ctx->parent_file_packet), &(par3_ctx->parent_file_packet_size), &(par3_ctx->parent_file_packet_count));
	if (ret == 0)
		ret = copy_parent_buffer(parent.ext_data_packet, parent.ext_data_packet_size, parent.ext_data_packet_count,
				&(par3_ctx->parent_ext_packet), &(par3_ctx->parent_ext_packet_size), &(par3_ctx->parent_ext_packet_count));
	if (ret == 0)
		ret = copy_parent_buffer(parent.file_system_packet, parent.file_system_packet_size, parent.file_system_packet_count,
				&(par3_ctx->parent_fs_packet), &(par3_ctx->parent_fs_packet_size), &(par3_ctx->parent_fs_packet_count));
	if (ret != 0){
		par3_release(&parent);
		return ret;
	}

	// Retain the parent's file table (name, size, fingerprint, packet checksum)
	// for detecting unchanged files during the child's create.
	if (parent.input_file_count > 0){
		uint32_t i;
		PAR3_PARENT_FILE *list;

		list = calloc(parent.input_file_count, sizeof(PAR3_PARENT_FILE));
		if (list == NULL){
			perror("Failed to allocate memory for parent file table");
			par3_release(&parent);
			return RET_MEMORY_ERROR;
		}
		par3_ctx->parent_file_list = list;
		for (i = 0; i < parent.input_file_count; i++){
			PAR3_FILE_CTX *fp = parent.input_file_list + i;
			uint8_t *pkt;
			size_t name_len;
			uint16_t len16;

			list[i].name = malloc(strlen(fp->name) + 1);
			if (list[i].name == NULL){
				perror("Failed to allocate memory for parent file name");
				par3_release(&parent);
				return RET_MEMORY_ERROR;
			}
			strcpy(list[i].name, fp->name);
			list[i].size = fp->size;
			memcpy(list[i].chk, fp->chk, 16);
			// Read the fingerprint hash from the File packet bytes:
			// body = name length (2), name, CRC-64 (8), fingerprint (16)
			pkt = parent.file_packet + fp->offset;
			memcpy(&len16, pkt + 48, 2);
			name_len = len16;
			memcpy(list[i].hash, pkt + 48 + 2 + name_len + 8, 16);
		}
		par3_ctx->parent_file_count = parent.input_file_count;
	}

	if (par3_ctx->noise_level >= 0){
		printf("Incremental parent = \"%s\"\n", parent_path);
		printf("Parent block count = %"PRIu64", file count = %u, block size = %"PRIu64"\n",
				parent.block_count, parent.input_file_count, parent.block_size);
	}

	par3_release(&parent);
	return 0;
}


// Compute the BLAKE3 hash of a whole file (streaming).
static int file_blake3(const char *path, uint8_t hash[16])
{
	uint8_t buf[65536];
	size_t len;
	FILE *fp;
	blake3_hasher hasher;

	fp = fopen(path, "rb");
	if (fp == NULL)
		return -1;
	blake3_hasher_init(&hasher);
	while ((len = fread(buf, 1, sizeof(buf), fp)) > 0)
		blake3_hasher_update(&hasher, buf, len);
	if (ferror(fp)){
		fclose(fp);
		return -1;
	}
	fclose(fp);
	blake3_hasher_finalize(&hasher, hash, 16);
	return 0;
}

// Mark input files that are unchanged since the parent backup.
// Marked files reuse the parent's File packet and input blocks:
// they get no new blocks and no new File packet in the child set.
static int mark_inherited_files(PAR3_CTX *par3_ctx)
{
	uint8_t hash[16];
	uint32_t num, i, inherited_count;
	PAR3_FILE_CTX *file_p;
	PAR3_PARENT_FILE *parent_p;

	if (par3_ctx->parent_file_count == 0)
		return 0;

	inherited_count = 0;
	file_p = par3_ctx->input_file_list;
	for (num = 0; num < par3_ctx->input_file_count; num++, file_p++){
		// Search a parent file with the same relative path and size.
		parent_p = NULL;
		for (i = 0; i < par3_ctx->parent_file_count; i++){
			if ( (par3_ctx->parent_file_list[i].size == file_p->size)
					&& (strcmp(par3_ctx->parent_file_list[i].name, file_p->name) == 0) ){
				parent_p = par3_ctx->parent_file_list + i;
				break;
			}
		}
		if (parent_p == NULL)
			continue;

		// Compare content fingerprints.
		if (file_blake3(file_p->name, hash) != 0)
			continue;
		if (memcmp(hash, parent_p->hash, 16) != 0)
			continue;

		// Unchanged: reuse the parent's File packet.
		file_p->state |= 0x01000000;
		memcpy(file_p->chk, parent_p->chk, 16);
		memcpy(file_p->hash, parent_p->hash, 16);
		file_p->chunk_num = 0;
		if (par3_ctx->total_file_size >= file_p->size)
			par3_ctx->total_file_size -= file_p->size;	// exclude from progress
		inherited_count++;
		if (par3_ctx->noise_level >= 2)
			printf("inherited from parent: \"%s\"\n", file_p->name);
	}

	if (par3_ctx->noise_level >= 0)
		printf("Inherited %u of %u input files from the parent backup.\n",
				inherited_count, par3_ctx->input_file_count);
	return 0;
}

// Append parent packets to a child packet buffer (after the child's own).
static int append_parent_packets(uint8_t **buf, size_t *buf_size, uint32_t *buf_count,
		uint8_t *add, size_t add_size, uint32_t add_count)
{
	uint8_t *tmp_p;

	if (add == NULL || add_size == 0)
		return 0;
	tmp_p = realloc(*buf, *buf_size + add_size);
	if (tmp_p == NULL){
		perror("Failed to re-allocate memory for parent packets");
		return RET_MEMORY_ERROR;
	}
	memcpy(tmp_p + *buf_size, add, add_size);
	*buf = tmp_p;
	*buf_size += add_size;
	*buf_count += add_count;
	return 0;
}

// add text in Creator Packet
int add_creator_text(PAR3_CTX *par3_ctx, char *text)
{
	uint8_t *tmp_p;
	size_t len, alloc_size;

	len = strlen(text);
	if (len == 0)
		return 0;

	if (par3_ctx->creator_packet == NULL){	// When there is no packet yet, allocate now.
		alloc_size = 48 + len;
		par3_ctx->creator_packet = malloc(alloc_size);
		if (par3_ctx->creator_packet == NULL){
			perror("Failed to allocate memory for Creator Packet");
			return RET_MEMORY_ERROR;
		}
	} else {	// When there is packet already, add new text to previous text.
		alloc_size = par3_ctx->creator_packet_size + len;
		tmp_p = realloc(par3_ctx->creator_packet, alloc_size);
		if (tmp_p == NULL){
			perror("Failed to re-allocate memory for Creator Packet");
			return RET_MEMORY_ERROR;
		}
		par3_ctx->creator_packet = tmp_p;
	}
	par3_ctx->creator_packet_size = alloc_size;
	memcpy(par3_ctx->creator_packet + alloc_size - len, text, len);
	par3_ctx->creator_packet_count = 1;

	return 0;
}

// add text in Comment Packet
int add_comment_text(PAR3_CTX *par3_ctx, char *text)
{
	uint8_t *tmp_p;
	size_t len, alloc_size;

	// If text is covered by ", remove them.
	len = strlen(text);
	if ( (len > 2) && (text[0] == '"') && (text[len - 1] == '"') ){
		text++;
		len -= 2;
	}
	if (len == 0)
		return 0;

	if (par3_ctx->comment_packet == NULL){	// When there is no packet yet, allocate now.
		alloc_size = 48 + len;
		par3_ctx->comment_packet = malloc(alloc_size);
		if (par3_ctx->comment_packet == NULL){
			perror("Failed to allocate memory for Comment Packet");
			return RET_MEMORY_ERROR;
		}
	} else {	// When there is packet already, add new comment to previous comment.
		alloc_size = par3_ctx->comment_packet_size + 1 + len;
		tmp_p = realloc(par3_ctx->comment_packet, alloc_size);
		if (tmp_p == NULL){
			perror("Failed to re-allocate memory for Comment Packet");
			return RET_MEMORY_ERROR;
		}
		par3_ctx->comment_packet = tmp_p;
		tmp_p += par3_ctx->comment_packet_size;
		tmp_p[0] = '\n';	// Put "\n" between comments.
	}
	par3_ctx->comment_packet_size = alloc_size;
	memcpy(par3_ctx->comment_packet + alloc_size - len, text, len);
	par3_ctx->comment_packet_count = 1;

	return 0;
}


int par3_trial(PAR3_CTX *par3_ctx, char *temp_path)
{
	int ret;
	uint64_t total_par_size;	// Total size of Index File, Archive Files, and Recovery Files.

	// Load input blocks on memory.
	if (par3_ctx->block_count == 0){
		ret = map_chunk_tail(par3_ctx);
	} else if (par3_ctx->deduplication == '1'){	// Simple deduplication
		ret = map_input_block(par3_ctx);
	} else if (par3_ctx->deduplication == '2'){	// Deduplication with slide search
		ret = map_input_block_slide(par3_ctx);
	} else {
		// Because this doesn't read file data, InputSetID will differ.
		ret = map_input_block_trial(par3_ctx);

		// This is for debug.
		// When no deduplication, no need to read input files in trial.
//		ret = map_input_block_simple(par3_ctx);
	}
	if (ret != 0)
		return ret;

	// Call this function before creating Start Packet.
	ret = calculate_recovery_count(par3_ctx);
	if (ret != 0)
		return ret;

	// Creator Packet, Comment Packet, Start Packet
	ret = make_start_packet(par3_ctx, 1);
	if (ret != 0)
		return ret;

	// Only when recovery blocks will be created, make Matrix Packet.
	if (par3_ctx->recovery_block_count > 0){
		ret = make_matrix_packet(par3_ctx);
		if (ret != 0)
			return ret;
	}

	// File Packet, Directory Packet, Root Packet
	ret = make_file_packet(par3_ctx);
	if (ret != 0)
		return ret;

	// External Data Packet
	ret = make_ext_data_packet(par3_ctx);
	if (ret != 0)
		return ret;

	// Try Index File
	total_par_size = try_index_file(par3_ctx);

	// Try other PAR3 files
	if ( (par3_ctx->block_count > 0) && ( (par3_ctx->data_packet != 0) || (par3_ctx->recovery_block_count > 0) ) ){
		ret = duplicate_common_packet(par3_ctx);
		if (ret != 0)
			return ret;

		// Write PAR3 files with input blocks
		if (par3_ctx->data_packet != 0){
			ret = try_archive_file(par3_ctx, temp_path, &total_par_size);
			if (ret != 0)
				return ret;
		}

		// Write PAR3 files with recovery blocks
		if (par3_ctx->recovery_block_count > 0){
			ret = try_recovery_file(par3_ctx, temp_path, &total_par_size);
			if (ret != 0)
				return ret;
		}
	}

	// Show efficiency rate
	if (par3_ctx->noise_level >= -1){
		double rate1, rate2;
		// rate1 "File data in Source blocks" = "total size of input file data" / "total size of source blocks"
		// rate2 "Recovery data in PAR files" = "total size of recovery blocks" / "total size of PAR files"
		// rate of "Efficiency of PAR files" = rate1 * rate2
		printf("\nTotal size of PAR files = %"PRIu64"\n", total_par_size);
		if ( (par3_ctx->block_count == 0) || (total_par_size == 0) ){
			rate1 = 0;
			rate2 = 0;
		} else {
			// Tiny chunk tails (1~39 bytes) don't consume blocks.
			// Duplicate data reuses same blocks.
			// Sum using bytes in every blocks to calculate total file data size.
			uint64_t block_count, total_data_size;
			PAR3_BLOCK_CTX *block_p;

			block_count = par3_ctx->block_count;
			block_p = par3_ctx->block_list;
			total_data_size = 0;
			while (block_count > 0){
				total_data_size += block_p->size;
				block_p++;
				block_count--;
			}
			//printf("Total file data in input blocks = %"PRIu64"\n", total_data_size);

			rate1 = (double)total_data_size / (double)(par3_ctx->block_size * par3_ctx->block_count);
			if (par3_ctx->data_packet != 0){	// Archive Files are same as 100% redundancy.
				rate2 = (double)(total_data_size + par3_ctx->block_size * par3_ctx->recovery_block_count) / (double)total_par_size;
			} else {
				rate2 = (double)(par3_ctx->block_size * par3_ctx->recovery_block_count) / (double)total_par_size;
			}
		}
		// Truncate two decimal places (use integer instead of showing double directly)
		//printf("rate1 = %f, rate2 = %f\n", rate1, rate2);
		ret = (int)(rate1 * 1000);
		printf("File data in Source blocks = %d.%d%%\n", ret / 10, ret % 10);
		ret = (int)(rate2 * 1000);
		printf("Recovery data in PAR files = %d.%d%%\n", ret / 10, ret % 10);
		ret = (int)(rate1 * rate2 * 1000);
		printf("Efficiency of PAR files    = %d.%d%%\n", ret / 10, ret % 10);
	}

	return 0;
}

int par3_create(PAR3_CTX *par3_ctx, char *temp_path)
{
	int ret;

	// Incremental backup: detect unchanged files, set block index offset.
	if ( (par3_ctx->parent_file_count > 0) || (par3_ctx->parent_block_count > 0) ){
		if (par3_ctx->ecc_method & 8){
			printf("FFT based Reed-Solomon Codes cannot be used for incremental backup.\n");
			printf("Use -e1 (Cauchy) or -e2 (Sparse) instead.\n");
			return RET_INVALID_COMMAND;
		}
		if (par3_ctx->deduplication != 0){
			printf("Deduplication is disabled for incremental backup.\n");
			par3_ctx->deduplication = 0;
		}
		par3_ctx->block_index_offset = par3_ctx->parent_block_count;
		ret = mark_inherited_files(par3_ctx);
		if (ret != 0)
			return ret;
	}

	// Map input file slices into input blocks.
	if (par3_ctx->block_count == 0){
		ret = map_chunk_tail(par3_ctx);
	} else if (par3_ctx->deduplication == '1'){	// Simple deduplication
		ret = map_input_block(par3_ctx);
	} else if (par3_ctx->deduplication == '2'){	// Deduplication with slide search
		ret = map_input_block_slide(par3_ctx);
	} else {
		ret = map_input_block_simple(par3_ctx);
	}
	if (ret != 0)
		return ret;

	// Call this function before creating Start Packet.
	ret = calculate_recovery_count(par3_ctx);
	if (ret != 0)
		return ret;

	// Creator Packet, Comment Packet, Start Packet
	ret = make_start_packet(par3_ctx, 0);
	if (ret != 0)
		return ret;

	// Only when recovery blocks will be created, make Matrix Packet.
	if (par3_ctx->recovery_block_count > 0){
		ret = make_matrix_packet(par3_ctx);
		if (ret != 0)
			return ret;
	}

	// File Packet, Directory Packet, Root Packet
	ret = make_file_packet(par3_ctx);
	if (ret != 0)
		return ret;

	// External Data Packet
	ret = make_ext_data_packet(par3_ctx);
	if (ret != 0)
		return ret;

	// Incremental backup: append the parent's vital packets so the child
	// set is self-contained (Start chain, File packets of inherited files,
	// block checksums, file-system metadata).
	if (par3_ctx->parent_start_packet_size > 0){
		ret = append_parent_packets(&(par3_ctx->start_packet), &(par3_ctx->start_packet_size), &(par3_ctx->start_packet_count),
				par3_ctx->parent_start_packet, par3_ctx->parent_start_packet_size, par3_ctx->parent_start_packet_count);
		if (ret == 0)
			ret = append_parent_packets(&(par3_ctx->file_packet), &(par3_ctx->file_packet_size), &(par3_ctx->file_packet_count),
					par3_ctx->parent_file_packet, par3_ctx->parent_file_packet_size, par3_ctx->parent_file_packet_count);
		if (ret == 0)
			ret = append_parent_packets(&(par3_ctx->ext_data_packet), &(par3_ctx->ext_data_packet_size), &(par3_ctx->ext_data_packet_count),
					par3_ctx->parent_ext_packet, par3_ctx->parent_ext_packet_size, par3_ctx->parent_ext_packet_count);
		if (ret == 0)
			ret = append_parent_packets(&(par3_ctx->file_system_packet), &(par3_ctx->file_system_packet_size), &(par3_ctx->file_system_packet_count),
					par3_ctx->parent_fs_packet, par3_ctx->parent_fs_packet_size, par3_ctx->parent_fs_packet_count);
		if (ret != 0)
			return ret;
	}

	// Write Index File
	ret = write_index_file(par3_ctx);
	if (ret != 0)
		return ret;

	// Write other PAR3 files
	if ( (par3_ctx->block_count > 0) && ( (par3_ctx->data_packet != 0) || (par3_ctx->recovery_block_count > 0) ) ){
		ret = duplicate_common_packet(par3_ctx);
		if (ret != 0)
			return ret;

		// When it uses Reed-Solomon Erasure Codes, it tries to keep all recovery blocks on memory.
		if (par3_ctx->ecc_method & 3){
			ret = allocate_recovery_block(par3_ctx);
			if (ret != 0)
				return ret;
		}

		// Write PAR3 files with input blocks
		if (par3_ctx->data_packet != 0){
			ret = write_archive_file(par3_ctx, temp_path);
			if (ret != 0)
				return ret;
		}

		// If there are enough memory to keep all recovery blocks,
		// it calculates recovery blocks before writing Recovery Data Packets.
		if (par3_ctx->ecc_method & 0x8000){
			ret = create_recovery_block(par3_ctx);
			if (ret < 0){
				par3_ctx->ecc_method &= ~0x8000;
			} else if (ret > 0){
				return ret;
			}
		}

		// Write PAR3 files with recovery blocks
		if (par3_ctx->recovery_block_count > 0){
			ret = write_recovery_file(par3_ctx, temp_path);
			if (ret != 0){
				//remove_recovery_file(par3_ctx);	// Remove partially created files
				return ret;
			}
		}

		// When recovery blocks were not created yet, calculate and write at here.
		if ((par3_ctx->ecc_method & 0x8000) == 0){
			if ( (par3_ctx->ecc_method & 8) && (par3_ctx->interleave > 0) ){
				// Interleaving is adapted only for FFT based Reed-Solomon Codes.
				ret = create_recovery_block_cohort(par3_ctx);
			} else {
				ret = create_recovery_block_split(par3_ctx);
			}
			if (ret != 0){
				//remove_recovery_file(par3_ctx);	// Remove partially created files
				return ret;
			}
		}
	}

	return 0;
}

