# Changelog

## 1.1.2 — 2026-10-06

### Added
- Root `NOTICE` with the HDF5 3-clause BSD copyright and permission block for the atomic n-bit, little-endian integer scale-offset, and Fletcher32 translations (`H5Znbit.c`, `H5Zscaleoffset.c`, `H5checksum.c`, `H5Zfletcher32.c`), and the zlib notice for `Huff::decode` (derived from `puff.c` 2.3, Copyright (C) 2002–2013 Mark Adler). Header comments on those functions name the source files.
- README Credits for those sources. Acknowledgments (AI assistants) stay in the README and are not part of `NOTICE`.

### Known issues reviewed for this bump

These are the items listed as still unsupported in 1.1.1 and in the README. Each still returns a named error. None guessed a value. They are deferred, not closed.

- **SZIP (Rice / libaec)** stays `FilteredNotSupported`. A correct decoder is a separate codec. Porting libaec would need its own BSD notice review, and this bump does not add that codec. Oracle: `gold_read_h5py_unsupported_filters_are_named` on `h5py_szip.h5` when the fixture is present.
- **Floating-point scale-offset** stays `FilteredNotSupported`. The crate contract forbids hardware `f32`/`f64` arithmetic in `src/` (see CONTRIBUTING). Doing this filter properly is a software IEEE path, not a small fix on the integer unpacker. Oracle: the same gold on `h5py_scaleoffset_f64.h5`.
- **N-bit on array or compound datatypes** stays `FilteredNotSupported`. The shipped path is atomic integers only. Array and compound n-bit need the rest of the HDF5 datatype-parameter tree (`H5Znbit.c` array and compound walkers), which is a new decoder, not a repair of the atomic walk. This tree has no h5py fixture for n-bit array or compound data; adding one only to keep asserting the named error would not implement the feature.
- **Paged extensible-array data blocks** stay `ChunkedNotSupported`. Unpaged `EADB` blocks are implemented and covered by the h5py extensible-array golds (`h5py_earray_*.h5`). Paged blocks add page-init bitmasks on superblocks and separately stored pages. That is a new index walker. The committed extensible-array fixtures are unpaged.

## 1.1.1 — 2026-10-05

### Added
- Layout v4/v5 chunk indexes: implicit, extensible array (unpaged `EAHD`/`EAIB`/`EADB`/`EASB`), and B-tree v2 (`BTHD`/`BTLF`/`BTIN`, including internal nodes).
- Filters besides gzip: shuffle, Fletcher32, atomic n-bit, and integer scale-offset. Pipeline order and per-chunk filter masks are honoured.

### Still unsupported (honest errors)
- SZIP (Rice / libaec bitstream) → `FilteredNotSupported`.
- Floating-point scale-offset (needs hardware float arithmetic) → `FilteredNotSupported`.
- N-bit on array or compound datatypes → `FilteredNotSupported`.
- Paged extensible-array data blocks → `ChunkedNotSupported`.

## 1.1.0 — 2026-10-05

### Added
- Pure-Rust zlib inflate for HDF5 deflate/gzip chunked reads (still zero crate deps).
- Layout message v4: single-chunk and fixed-array (FAHD/FADB) chunk indexes.
- Compound datatype field introspection (`compound_fields`) and `read_raw`.
- Big-endian IEEE and integer reads normalized to little-endian bit lanes.

### Changed
- `HDF5DType::Compound { size }` for compound datasets (was `Other`).
- Gzip h5py golds now expect successful inflate (no longer `FilteredNotSupported`).

### Still Unsupported (honest errors)
- Layout v4 extensible-array, implicit, and B-tree v2 indexes → `ChunkedNotSupported`.
- Non-deflate filters (shuffle, szip, …) → `FilteredNotSupported`.

## 1.0.5 — 2026-10-05

- `no_std` + alloc build fixed: bare `vec!` macros in `messages` / `encode` now use `alloc::vec!` (std prelude hid the bug under default features).
- Docs aligned: README install lines and `doc/CAPABILITIES.md` version/date match the crate (were stuck at 1.0.2 after 1.0.3/1.0.4).


## 1.0.4 — 2026-09-20

Clippy debt clear (`-D warnings`): needless collects, lifetime elision, `WalkCtx` for decode arity, gold cast tidy.

## 1.0.3 — 2026-09-19

Coordinated patch with zenith-float, latex-rust, and redb-view (pure-Rust FOSS family adjacent to Accumath; Accumath itself stays proprietary).

## 1.0.2 — 2026-09-04

- Build plan removed from the crate package. Capabilities and contributing stay.
- Author email `jscarr1964@gmail.com`.

## 1.0.1 — 2026-09-04

- Author email `jscarr1964@gmail.com`.
- Integer LE datasets: `i8`/`i16`/`i32`/`i64`/`u8`/`u16`/`u32`/`u64` write, read, and selected append.
- Rank 0 (scalar) through **32** (HDF5 format max). Rank 2 is not a cap. Rank 3/4/5 golds; mixed files open.
- Uncompressed chunked read (B-tree v1 type 1). Gzip/deflate is `FilteredNotSupported`. Layout v4 chunk index stays `ChunkedNotSupported`.
- Unknown dtypes (compound, big-endian float, …) are `HDF5DType::Other`, not `Opaque` or `Float64`.
- `dataset_shape` reports shape for chunked/filtered datasets; only payload reads fail.
- `write_attr_str` / `list_attrs`. `append_f32` / `append_i32` / `append_u64` / `append_opaque`.
- h5py golds: int32, rank-3, mixed table+cube, uncompressed chunked values, gzip filter error, complex → Other.

## 1.0.0 — 2026-08-31

First stable crates.io release.

- Pure-Rust HDF5 reader and writer. No `libhdf5`, no C FFI, no crate dependencies.
- Write: superblock v2, compact groups, contiguous `IEEE_F64LE` / `IEEE_F32LE` / opaque, 1-D and 2-D.
- Read: superblock v0–v3, object headers v1/v2, old-style symbol-table groups, contiguous data, string attributes (fixed and VL).
- Chunked datasets return `Err(HDF5Error::ChunkedNotSupported)`.
- IEEE values are integer bit patterns (`u64` / `u32`). Hardware IEEE arithmetic is not used.
- `std` (default) for `save` / `open`. `no_std` + alloc via `from_bytes` / `to_bytes`.
- Interop golds: h5py and MATLAB `h5read` of files this crate writes.
