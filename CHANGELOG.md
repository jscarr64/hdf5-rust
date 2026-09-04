# Changelog

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
