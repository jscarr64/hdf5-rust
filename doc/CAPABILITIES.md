# hdf5-rust capabilities

**Last updated:** 2026-10-05
**Crate version:** 1.1.1

Independent HDF5 implementation. Not a wrapper. Not affiliated with The HDF Group. **Zero crate dependencies.**

| Symbol | Meaning |
| -------- | --------- |
| ✅ | Built, in public API, covered by default CI |
| 🟡 | Built but partial |
| ⬜ | Not implemented |

## Kernel

| Item | Status |
| ------ | -------- |
| Pure Rust, no libhdf5 / C FFI / crate deps | ✅ |
| `no_std` + alloc; `std` feature for `std::fs` | ✅ |
| Software IEEE bits only (`u32`/`u64` lanes) | ✅ |
| Jenkins lookup3 checksum | ✅ |
| Superblock write v2 / read v0+v1+v2+v3 | ✅ |
| WASM-capable (no libc HDF5) | ✅ |
| Pure-Rust zlib inflate (deflate filter) | ✅ |

## Datasets

| Item | Status |
| ------ | -------- |
| Contiguous IEEE_F64LE / IEEE_F32LE | ✅ |
| Integer LE i8–i64 / u8–u64 | ✅ |
| Big-endian IEEE / integer → LE lanes on read | ✅ |
| Opaque records | ✅ |
| Compound: field layout + `read_raw` blob | ✅ |
| Other unspecialized → `HDF5DType::Other` | ✅ |
| Rank 0 (scalar) through 32 (HDF5 max; not capped at 2) | ✅ |
| Uncompressed chunked read (B-tree v1) | ✅ |
| Layout v4 single-chunk index | ✅ |
| Layout v4 fixed-array index (FAHD/FADB) | ✅ |
| Implicit chunk index | ✅ |
| Extensible-array index (unpaged data blocks) | ✅ |
| B-tree v2 chunk index (leaf and internal) | ✅ |
| Paged extensible-array data blocks | ⬜ → `ChunkedNotSupported` |
| Gzip/deflate filter on chunked read | ✅ |
| Shuffle, Fletcher32, atomic n-bit, integer scale-offset | ✅ |
| SZIP, floating-point scale-offset, n-bit array/compound | ⬜ → `FilteredNotSupported` |
| Append along first dimension | ✅ |

## Groups and attributes

| Item | Status |
| ------ | -------- |
| Compact groups, nested paths | ✅ |
| Old-style symbol-table groups (read) | ✅ |
| String attributes (fixed) | ✅ |
| VL string attributes (global heap, read) | ✅ |
| `write_attr_str` / `list_attrs` | ✅ |
| Write `hdf5-rust-version` and `created` | ✅ |
| h5py reads our files / we read h5py files | ✅ |
| MATLAB `h5read` of our files | ✅ |

## Caller boundary

This crate has **zero crate dependencies**. IEEE datasets are `u64`/`u32` bit lanes (BE sources are byte-swapped into LE lanes on read). Integers are `i*`/`u*` lanes. Opaque datasets are raw records of a caller-chosen element size. Compound datasets report `HDF5DType::Compound { size }` with `compound_fields` introspection and `read_raw` for the opaque element bytes. `write_ieee64` / `read_ieee64` / `write_ieee32` / `read_ieee32` / `append_ieee64` are aliases of the `write_f64` family.
