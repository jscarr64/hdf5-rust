# Changelog

## 1.0.0 — 2026-08-31

First stable crates.io release.

- Pure-Rust HDF5 reader and writer. No `libhdf5`, no C FFI, no crate dependencies.
- Write: superblock v2, compact groups, contiguous `IEEE_F64LE` / `IEEE_F32LE` / opaque, 1-D and 2-D.
- Read: superblock v0–v3, object headers v1/v2, old-style symbol-table groups, contiguous data, string attributes (fixed and VL).
- Chunked datasets return `Err(HDF5Error::ChunkedNotSupported)`.
- IEEE values are integer bit patterns (`u64` / `u32`). Hardware IEEE arithmetic is not used.
- `std` (default) for `save` / `open`. `no_std` + alloc via `from_bytes` / `to_bytes`.
- Interop golds: h5py and MATLAB `h5read` of files this crate writes.
