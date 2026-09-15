add_library(blake3 STATIC
    blake3.c
    blake3_dispatch.c
    blake3_portable.c
    blake3_sse2.c
    blake3_sse41.c
    blake3_avx2.c
    blake3_avx512.c
)

if(MSVC)
    # MSVC compiles SSE2/SSE4.1/AVX2 intrinsics without extra flags;
    # only AVX-512 needs an explicit arch switch.
    set_source_files_properties(blake3_avx512.c PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
else()
    set_source_files_properties(blake3_sse2.c   PROPERTIES COMPILE_OPTIONS "-msse2")
    set_source_files_properties(blake3_sse41.c  PROPERTIES COMPILE_OPTIONS "-msse4.1")
    set_source_files_properties(blake3_avx2.c   PROPERTIES COMPILE_OPTIONS "-mavx2")
    set_source_files_properties(blake3_avx512.c PROPERTIES COMPILE_OPTIONS "-mavx512vl")
endif()
