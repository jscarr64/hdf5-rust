# Contributing to hdf5-rust

Thank you for looking at the crate. This is a small, standalone HDF5 **format** library. Version **1.0.2** is the current public API (`Hdf5File`, named `HDF5Error` variants, zero crate dependencies). Breaking changes would be a new major version. Issues and pull requests are welcome when they match that job.

## What this crate is

- Pure Rust. No `libhdf5`, no C FFI, **no crate dependencies**.
- IEEE values are integer bit patterns (`u64` / `u32` lanes). Hardware IEEE arithmetic is not used in `src/`.
- Callers wrap their own array types. This crate does not depend on a numeric library.
- Write: superblock v2, compact groups, contiguous datasets (`IEEE_F64LE`, `IEEE_F32LE`, integer LE, opaque), rank 0–32 (format max; not capped at 2).
- Read: superblock v0–v3, object headers v1/v2, old-style symbol-table groups, contiguous data, uncompressed chunked data, string attributes, integer dtypes.
- Filtered (gzip/deflate) datasets return `Err(HDF5Error::FilteredNotSupported)`. A chunked layout that cannot be walked returns `ChunkedNotSupported`. Those are named stops, not panics and not guessed values.

Incomplete paths return `Err`. They must not panic and must not invent data.

## Before you write code

Open an **issue** first for anything larger than a typo. Say what file or API you hit, what you expected, and what happened.

A useful report includes:

- How the file was written (this crate, h5py, MATLAB, …)
- Dataset path, shape, and datatype if you know them
- The `HDF5Error` variant, not only “it failed”

## Development

Rust 1.70 or newer.

```bash
cargo test
cargo test --no-default-features
bash scripts/ci_full.sh
```

`scripts/ci_full.sh` is the gate: tests with and without `std`, empty `[dependencies]`, no hardware IEEE type tokens in `src/` except the named `write_f64` / `read_f32` family, no `TODO`/`FIXME`/`HACK` in `src/`.

Interop tests in `tests/interop.rs` need **h5py** and a working `python3`. They read fixtures under `tests/fixtures/` (regenerate with `python3 scripts/gen_h5py_fixtures.py`). The MATLAB `h5read` gold is `#[ignore]` unless you run:

```bash
MATLAB=/path/to/matlab ./scripts/matlab_h5read_gold.sh
```

## Golds

A gold is an expected **bit pattern**, **shape**, **attribute string**, or **named error**. Tests that only show “did not panic” or `is_ok()` are not enough.

Existing golds in `tests/golds.rs` / `tests/interop.rs` are the bar. Do not weaken them. Do not mark a failing gold `#[ignore]` to go green.

## Pull requests

- Keep the diff to the problem. Do not reformat unrelated files.
- Do not add crate dependencies, `unsafe`, or C/FFI.
- Do not add hardware IEEE types (`f32` / `f64`) in `src/` except by extending the existing `write_f64` / `read_f32` method names if the HDF5 native type requires that name.
- Public errors stay named enum variants (`HDF5Error::…`), not string-only failures.
- Run `bash scripts/ci_full.sh` before you push.

By submitting a change you agree it is licensed under **MIT OR Apache-2.0**, the same as the rest of the crate (`LICENSE-MIT`, `LICENSE-APACHE`).

## Out of scope (will be closed)

- Linking `libhdf5` or wrapping another HDF5 crate
- Chunked or filtered read that returns guessed values
- Depending on a numeric library inside this crate
- Hardware floating-point arithmetic

Those belong in a caller, or they contradict the crate contract.
