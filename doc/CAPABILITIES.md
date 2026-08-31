# hdf5-rust capabilities

**Last updated:** 2026-08-31
**Crate version:** 0.1.0

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

## Datasets

| Item | Status |
| ------ | -------- |
| Contiguous IEEE_F64LE / IEEE_F32LE | ✅ |
| Opaque records | ✅ |
| 1-D and 2-D simple dataspace | ✅ |
| Chunked → `ChunkedNotSupported` | ✅ |
| Append along first dimension | ✅ |

## Groups and attributes

| Item | Status |
| ------ | -------- |
| Compact groups, nested paths | ✅ |
| Old-style symbol-table groups (read) | ✅ |
| String attributes (fixed) | ✅ |
| VL string attributes (global heap, read) | ✅ |
| Write `hdf5-rust-version` and `created` | ✅ |
| h5py reads our files / we read h5py files | ✅ |
| MATLAB `h5read` of our files | ✅ |

## Caller boundary

This crate has **zero crate dependencies**. IEEE datasets are `u64`/`u32` bit lanes. Opaque datasets are raw records of a caller-chosen element size. `write_ieee64` / `read_ieee64` / `write_ieee32` / `read_ieee32` / `append_ieee64` are aliases of the `write_f64` family.
