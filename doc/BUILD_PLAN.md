# hdf5-rust build plan

**Trio:** this file · inventory [`CAPABILITIES.md`](CAPABILITIES.md) · golds in `tests/`. A gold is an expected file, bit pattern, shape, or named error. Do not invent substitutes. Incomplete paths return `Err`.

**Last updated:** 2026-08-31
**Crate:** `hdf5-rust` 1.0.0
**License:** MIT OR Apache-2.0
**Policy:** 100% Rust. No libhdf5. No C FFI. **No crate dependencies.** No hardware float in any arithmetic path. `unsafe` only if a byte cast is strictly required (none in v1.0).

Crate name on crates.io cannot be `hdf5` (C bindings). This package is `hdf5-rust`.

---

## 1. Format (write)

| Item | Status | Notes |
| ------ | -------- | ------- |
| Superblock v2, offsets/lengths = 8 | ✅ | Jenkins lookup3 checksum |
| Root group, compact links (no fractal heap) | ✅ | Link + Link Info messages |
| `create_group` one level under root | ✅ | Nested path `results/data` gold |
| Contiguous layout v3 | ✅ | |
| `IEEE_F64LE` / `IEEE_F32LE` | ✅ | Integer bits only |
| Opaque + ASCII tag | ✅ | 50×50 × 16-byte records gold |
| Simple dataspace 1-D and 2-D | ✅ | Unlimited first dim for append |
| String attrs `hdf5-rust-version`, `created` | ✅ | On every written dataset |
| `append_f64` grows first dimension | ✅ | `ShapeMismatch` if cols differ |

## 2. Format (read)

| Item | Status | Notes |
| ------ | -------- | ------- |
| Superblock v0 / v1 / v2 / v3 | ✅ | v0 h5py default; v2 our writes; v3 h5py `libver=latest` |
| Object header v1 and v2 | ✅ | v1 old-style files; v2 compact |
| Old-style groups (TREE + SNOD + HEAP) | ✅ | h5py default gold |
| New-style compact links | ✅ | |
| Contiguous raw data | ✅ | |
| Chunked layout | ✅ | h5py `chunks=` → `ChunkedNotSupported` |
| VL string attributes (global heap) | ✅ | h5py `ds.attrs['units']='kelvin'` |
| Fixed-length string attributes | ✅ | |
| Rank > 2 | ✅ | `RankNotSupported` on write |

## 3. Public API

| Item | Status |
| ------ | -------- |
| `list_datasets` / `list_groups` | ✅ |
| `dataset_shape` / `dataset_dtype` | ✅ |
| `read_f64` / `read_f32` / `read_opaque` | ✅ |
| `read_f64_slice` | ✅ |
| `write_f64` / `write_f32` / `write_opaque` | ✅ |
| `write_ieee64` / `write_ieee32` / `read_ieee64` / `read_ieee32` / `append_ieee64` | ✅ | Aliases for callers that forbid hardware float tokens |
| `append_f64` / `create_group` | ✅ |
| `read_attr_str` | ✅ |
| `std`: `open` / `create_file` / `save` | ✅ |
| IEEE as `u64`/`u32` bits; opaque as `&[u8]` (no numeric crates) | ✅ |

Caller aliases `write_ieee64` / `read_ieee64` / `write_ieee32` / `read_ieee32` / `append_ieee64` match the `write_f64` family (for crates that forbid hardware float tokens).

## 4. Named errors (required)

`InvalidSignature`, `UnsupportedVersion(v)`, `ChunkedNotSupported`, `NotFound(path)`, `ShapeMismatch`. Additional named variants for truncated/malformed files are allowed; no string-only errors.

## 5. Golds

Passing in `tests/golds.rs`: 100×3 F64 bits, 50×50 opaque, 1-D 1000 F32 bits, nested group, append 10×3→20×3, three dataset list, signature / not-found / shape / chunked errors, row slice.

Passing in `tests/interop.rs` against h5py 3.16 / HDF5 2.0 fixtures in `tests/fixtures/`: default v0 float64, latest VL string attr, three datasets, chunked → `ChunkedNotSupported`, **h5py opens our write_f64** (values + `hdf5-rust-version`).

MATLAB `h5read` gold: **passed** on R2026a Update 5 (`/usr/local/MATLAB/R2026a/bin/matlab`). `h5read` + `h5readatt` of `hdf5-rust-version` on our `write_f64` file. Re-run: `MATLAB=/usr/local/MATLAB/R2026a/bin/matlab ./scripts/matlab_h5read_gold.sh`.

## 6. Done means

Every gold in `tests/` that we can run here passes (self-roundtrip, h5py read/write, MATLAB `h5read`). No libhdf5 in Cargo.toml. `scripts/ci_full.sh` green. Purity grep: no hardware float arithmetic in `src/`.
