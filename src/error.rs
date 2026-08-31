//! Named errors. No string-only error type.

use alloc::string::String;
use core::fmt;

/// Recoverable HDF5 failure. Public variants listed in the build spec are
/// stable; additional variants describe truncated or malformed files.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HDF5Error {
    /// Bytes at the superblock location are not [`crate::HDF5_SIGNATURE`].
    InvalidSignature,
    /// Superblock or object-header version this crate does not read.
    UnsupportedVersion(u8),
    /// Dataset storage layout is chunked.
    ChunkedNotSupported,
    /// No object at `path`.
    NotFound(String),
    /// Append rank or trailing dimensions do not match the existing dataset.
    ShapeMismatch,
    /// File ended before a field was complete.
    Truncated,
    /// A header, checksum, or message failed validation.
    InvalidHeader,
    /// Dataspace rank is greater than 2 (or greater than [`crate::HDF5_MAX_DIMS`]).
    RankNotSupported,
    /// Dataset datatype cannot satisfy the requested read (e.g. `read_f64` on F32).
    TypeMismatch,
    /// Group stores links in a fractal heap / dense index.
    DenseGroupsNotSupported,
    /// Link or attribute name exceeds [`crate::HDF5_MAX_NAME_LEN`].
    NameTooLong,
    /// I/O error from `std::fs` (feature `std`).
    #[cfg(feature = "std")]
    Io(String),
}

impl fmt::Display for HDF5Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSignature => f.write_str("invalid HDF5 signature"),
            Self::UnsupportedVersion(v) => write!(f, "unsupported HDF5 version {v}"),
            Self::ChunkedNotSupported => f.write_str("chunked dataset not supported"),
            Self::NotFound(p) => write!(f, "HDF5 path not found: {p}"),
            Self::ShapeMismatch => f.write_str("HDF5 dataspace shape mismatch"),
            Self::Truncated => f.write_str("truncated HDF5 file"),
            Self::InvalidHeader => f.write_str("invalid HDF5 header"),
            Self::RankNotSupported => f.write_str("HDF5 rank not supported"),
            Self::TypeMismatch => f.write_str("HDF5 datatype mismatch"),
            Self::DenseGroupsNotSupported => f.write_str("dense HDF5 groups not supported"),
            Self::NameTooLong => f.write_str("HDF5 name too long"),
            #[cfg(feature = "std")]
            Self::Io(s) => write!(f, "HDF5 I/O error: {s}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for HDF5Error {}

#[cfg(feature = "std")]
impl From<std::io::Error> for HDF5Error {
    fn from(e: std::io::Error) -> Self {
        Self::Io(alloc::string::ToString::to_string(&e))
    }
}

/// `Result` alias for this crate’s [`HDF5Error`].
pub type Result<T> = core::result::Result<T, HDF5Error>;
