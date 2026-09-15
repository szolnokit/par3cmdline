# par3cmdline
Official repo for par3cmdline and par3lib

**As of January 29, 2022** The Par3 specification is in near-final form. This repository will hold a reference implementation. After we have working code and worked out most of the bugs, the specification will be finalized.

The current draft of the specification (aligned with this reference implementation) is:
[Parity_Volume_Set_Specification_v3.0.html](Parity_Volume_Set_Specification_v3.0.html)

An earlier public draft is also available at:
https://parchive.github.io/doc/Parity_Volume_Set_Specification_v3.0.html

## Implemented Error Correction Codes

- `-e1` Cauchy Reed-Solomon (default; 8-bit or 16-bit Galois field)
- `-e2` Sparse Random Matrix (faster for large block counts; keep a few
  extra recovery blocks as safety margin)
- `-e2x` Sparse Random Matrix with extended block count: allows more than
  65536 input blocks (up to 2^24) using the `PAR SPX` packet. Sets created
  this way can be verified by any client, but repaired only by clients
  that support the extension.
- `-e8` FFT based Reed-Solomon (Leopard-RS)

## Incremental backup

A new PAR3 set can reuse an existing set as its parent:

```
par3 c -s1000 -r20 backup1.par3 *          # full backup
...add or change files...
par3 c -s1000 -r20 -Pbackup1.par3 backup2.par3 *   # incremental
```

Files that are unchanged since the parent are inherited: they get no new
input blocks and the parent's packets are reused, so the child set stays
small. The child's recovery data covers only the new blocks; the parent's
recovery files keep protecting the old blocks. Verifying the child checks
the whole file tree. If an old file is damaged, repair it with the parent's
PAR3 files (the child will tell you when that is needed).

Notes:
- The child inherits the parent's block size and Galois field.
- Deduplication is disabled and FFT codes are not supported with `-P`.
- Keep the parent's PAR3 files: the chain needs every generation.

## Performance

The Reed-Solomon hot paths use runtime CPU dispatch:

- Galois-field region multiply (GF(2^8) and GF(2^16)) runs on AVX2 or SSSE3
  (PSHUFB split-table technique) when available, with a scalar fallback.
- Recovery block encoding, lost block decoding, and Gaussian elimination
  are multithreaded with OpenMP (rows are independent).

On a modern multi-core CPU this speeds up creation roughly 4x and repair
roughly 2.5x compared to the scalar single-threaded code. No format change:
the produced PAR3 files are identical in structure to the unoptimized build.

Repair with sparse codes (`-e2`, `-e2x`) uses a peeling decoder: the input
data is streamed once to build residuals for the recovery blocks that touch
the lost blocks, degree-1 equations are peeled off, and whatever remains is
solved by a small dense elimination in the lost-block space. The cost is
proportional to the damage instead of the whole block count: on an 80 000
block set, repairing 3000 lost blocks dropped from 425 s to 44 s. Memory for
the residual buffers is about (lost blocks x 6) block sizes; when `-m` does not
allow that, repair falls back to the dense path automatically.

## Testing

Run the end-to-end suite on Windows:

```
powershell -File tests\par3-testsuite.ps1
```
