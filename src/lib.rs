//! Pure-Rust HDF5 reader and writer. No libhdf5, no C FFI, no crate dependencies.
//!
//! IEEE values are integer bit patterns (`u64` / `u32`). Opaque datasets are
//! raw records. Hardware IEEE arithmetic is not used.

#![cfg_attr(not(feature = "std"), no_std)]
#![deny(missing_docs)]
#![deny(clippy::float_arithmetic)]
#![deny(clippy::suspicious)]

extern crate alloc;

mod btree;
mod buf;
mod checksum;
mod consts;
mod decode;
mod encode;
mod error;
mod file;
mod messages;
mod model;
mod ohdr;
mod superblock;

pub use consts::*;
pub use error::{HDF5Error, Result};
pub use file::{dataset_dtype, dataset_shape, list_datasets, list_groups, Hdf5File};

/// Dataset datatype as reported by [`Hdf5File::dataset_dtype`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HDF5DType {
    /// IEEE 754 binary64, little-endian.
    Float64,
    /// IEEE 754 binary32, little-endian.
    Float32,
    /// Opaque element of `size` bytes.
    Opaque(usize),
    /// Any other HDF5 datatype this crate does not specialize.
    Other,
}

/// Well-formed file whose dataset `data` is chunked (error-gold fixture).
#[doc(hidden)]
pub fn fixture_chunked_dataset() -> Result<alloc::vec::Vec<u8>> {
    encode::encode_chunked_stub()
}
