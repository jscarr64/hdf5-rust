# hdf5-rust

Pure-Rust [HDF5](https://www.hdfgroup.org/solutions/hdf5/) reader and writer. **No libhdf5. No C FFI. No crate dependencies.** Current release: **1.0.2** on [crates.io](https://crates.io/crates/hdf5-rust).

Files are standard HDF5: h5py, MATLAB `h5read`, HDFView, and Julia HDF5.jl can open what this crate writes, and this crate can open what those tools write (contiguous IEEE, integer, and opaque; uncompressed chunked IEEE).

Inventory: [capabilities](doc/CAPABILITIES.md). Patches: [CONTRIBUTING.md](CONTRIBUTING.md).

## Why this crate exists

Every other Rust HDF5 crate binds the C library. That blocks WebAssembly, `no_std`, reproducible builds, and any project that refuses foreign I/O kernels. Scientific code still needs the file format. `hdf5-rust` implements the on-disk specification in Rust.

## What it does

- **Write** superblock version 2, compact groups, contiguous datasets.
- **Read** superblock versions 0, 1, 2, and 3; object headers v1 and v2; old-style symbol-table groups and new-style compact links.
- **Datatypes:** `IEEE_F64LE`, `IEEE_F32LE`, signed/unsigned integers (8/16/32/64-bit LE), opaque records. Compound / big-endian / other classes are `HDF5DType::Other` (typed reads are `TypeMismatch`, not opaque).
- **Dataspace:** simple rank 0 (scalar) through **32** (the HDF5 format maximum). Images, volumes, and time series (rank 3–5) are normal, not an error. A rank-3 cube does not prevent reading a rank-2 table in the same file.
- **Chunked** uncompressed datasets are assembled on read. Gzip/deflate and other filters return `Err(HDF5Error::FilteredNotSupported)`. A chunked layout this crate cannot walk returns `ChunkedNotSupported`. Never a panic and never guessed values.

IEEE values are stored as integer bit patterns (`u64` / `u32` lanes). This crate never uses hardware `f32`/`f64` arithmetic. Callers wrap their own array types at the boundary.

Written datasets carry string attributes `hdf5-rust-version` and `created`.

## Install

```toml
[dependencies]
hdf5-rust = "1.0.2"
```

`std` (file I/O) is on by default. For `no_std` + alloc:

```toml
hdf5-rust = { version = "1.0.2", default-features = false }
```

Then use `Hdf5File::from_bytes` / `to_bytes`.

## Quick start

```rust
use hdf5_rust::Hdf5File;

fn main() -> hdf5_rust::Result<()> {
    // Integer IEEE binary64 bit patterns (1.0, 2.0, 3.0, 4.0, 5.0, 6.0).
    let bits: &[u64] = &[
        0x3FF0_0000_0000_0000,
        0x4000_0000_0000_0000,
        0x4008_0000_0000_0000,
        0x4010_0000_0000_0000,
        0x4014_0000_0000_0000,
        0x4018_0000_0000_0000,
    ];

    let mut file = Hdf5File::create();
    file.create_group("results")?;
    file.write_f64("results/data", &[2, 3], bits)?;
    file.write_i32("results/counts", &[2, 3], &[1, 2, 3, 4, 5, 6])?;

    let (shape, got) = file.read_f64("results/data")?;
    assert_eq!(shape, vec![2, 3]);
    assert_eq!(got, bits);

    let bytes = file.to_bytes()?;
    assert_eq!(&bytes[..8], &hdf5_rust::HDF5_SIGNATURE);
    Ok(())
}
```

`save` / `open` require the `std` feature (on by default). `write_ieee64` / `read_ieee64` are the same methods under names that do not contain hardware float tokens.

## License

MIT OR Apache-2.0. See [CONTRIBUTING.md](CONTRIBUTING.md) if you want to report a bug or send a patch.

## HDF5 trademark

HDF5 is a trademark of The HDF Group. This crate is an independent implementation of the published file format. It is not affiliated with The HDF Group.
