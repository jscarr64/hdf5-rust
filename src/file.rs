//! Public [`Hdf5File`] API: in-memory image, optional `std` paths.

use alloc::string::{String, ToString};
use alloc::vec::Vec;

use crate::error::{HDF5Error, Result};
use crate::model::{DTypeKind, DatasetRec, FileModel};
use crate::HDF5DType;

/// An HDF5 file held as a logical graph and (re)serialized to bytes on demand.
#[derive(Clone, Debug)]
pub struct Hdf5File {
    model: FileModel,
}

impl Default for Hdf5File {
    fn default() -> Self {
        Self::create()
    }
}

impl Hdf5File {
    /// Empty file with a root group. Writes superblock version 2 on [`Self::to_bytes`].
    pub fn create() -> Self {
        Self {
            model: FileModel::new(),
        }
    }

    /// Parse a complete HDF5 file image. Returns [`HDF5Error::InvalidSignature`]
    /// if the format signature is missing.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        Ok(Self {
            model: crate::decode::decode(bytes)?,
        })
    }

    /// Serialize to a standard HDF5 superblock-v2 file image.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        crate::encode::encode(&self.model)
    }

    /// Open an existing file from a filesystem path.
    ///
    /// Errors: [`HDF5Error::Io`], plus every error [`Self::from_bytes`] can return.
    #[cfg(feature = "std")]
    pub fn open<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        Self::from_bytes(&bytes)
    }

    /// Create an empty file object (does not write until [`Self::save`]).
    #[cfg(feature = "std")]
    pub fn create_file<P: AsRef<std::path::Path>>(_path: P) -> Result<Self> {
        Ok(Self::create())
    }

    /// Write the current image to `path`.
    ///
    /// Errors: [`HDF5Error::Io`] and encode failures.
    #[cfg(feature = "std")]
    pub fn save<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        let bytes = self.to_bytes()?;
        std::fs::write(path, bytes)?;
        Ok(())
    }

    /// Dataset paths (no leading slash), sorted.
    pub fn list_datasets(&self) -> Vec<String> {
        self.model.list_datasets()
    }

    /// Group paths. Root is `"/"`.
    pub fn list_groups(&self) -> Vec<String> {
        self.model.list_groups()
    }

    /// Dataspace dimensions of `path`.
    ///
    /// Errors: [`HDF5Error::NotFound`], [`HDF5Error::ChunkedNotSupported`].
    pub fn dataset_shape(&self, path: &str) -> Result<Vec<usize>> {
        let ds = self.dataset(path)?;
        if ds.chunked {
            return Err(HDF5Error::ChunkedNotSupported);
        }
        Ok(ds.shape.iter().map(|d| *d as usize).collect())
    }

    /// Datatype of `path`.
    ///
    /// Errors: [`HDF5Error::NotFound`]. Chunked datasets still report a type.
    pub fn dataset_dtype(&self, path: &str) -> Result<HDF5DType> {
        let p = FileModel::normalize(path)?;
        let ds = self
            .model
            .datasets
            .get(&p)
            .ok_or_else(|| HDF5Error::NotFound(p))?;
        Ok(ds.kind.to_public())
    }

    /// Create a group. Parent must exist (root always does).
    ///
    /// Errors: [`HDF5Error::NotFound`], [`HDF5Error::NameTooLong`].
    pub fn create_group(&mut self, path: &str) -> Result<()> {
        self.model.create_group(path)
    }

    /// Write IEEE binary64 little-endian lanes. `bits.len()` must equal the
    /// product of `shape`. Rank 1 or 2 only.
    ///
    /// Errors: [`HDF5Error::ShapeMismatch`], [`HDF5Error::RankNotSupported`],
    /// [`HDF5Error::NameTooLong`].
    pub fn write_f64(&mut self, path: &str, shape: &[usize], bits: &[u64]) -> Result<()> {
        self.write_ieee64(path, shape, bits)
    }

    /// IEEE binary64 little-endian lanes. Same as [`Self::write_f64`].
    ///
    /// Named for callers whose sources cannot contain hardware float tokens.
    pub fn write_ieee64(&mut self, path: &str, shape: &[usize], bits: &[u64]) -> Result<()> {
        self.write_bits(path, shape, DTypeKind::Float64, lanes_u64(bits))
    }

    /// Write IEEE binary32 little-endian lanes.
    ///
    /// Errors: same as [`Self::write_f64`].
    pub fn write_f32(&mut self, path: &str, shape: &[usize], bits: &[u32]) -> Result<()> {
        self.write_ieee32(path, shape, bits)
    }

    /// IEEE binary32 little-endian lanes. Same as [`Self::write_f32`].
    pub fn write_ieee32(&mut self, path: &str, shape: &[usize], bits: &[u32]) -> Result<()> {
        self.write_bits(path, shape, DTypeKind::Float32, lanes_u32(bits))
    }

    /// Write opaque records of `elem_size` bytes. `data.len()` must equal
    /// `shape.product() * elem_size`.
    ///
    /// Errors: same as [`Self::write_f64`].
    pub fn write_opaque(
        &mut self,
        path: &str,
        shape: &[usize],
        elem_size: usize,
        data: &[u8],
    ) -> Result<()> {
        self.write_bits(path, shape, DTypeKind::Opaque(elem_size), data.to_vec())
    }

    /// Read IEEE binary64 lanes and the dataspace shape.
    ///
    /// Errors: [`HDF5Error::NotFound`], [`HDF5Error::ChunkedNotSupported`],
    /// [`HDF5Error::TypeMismatch`].
    pub fn read_f64(&self, path: &str) -> Result<(Vec<usize>, Vec<u64>)> {
        self.read_ieee64(path)
    }

    /// IEEE binary64 lanes and dataspace shape. Same as [`Self::read_f64`].
    pub fn read_ieee64(&self, path: &str) -> Result<(Vec<usize>, Vec<u64>)> {
        let ds = self.dataset(path)?;
        if ds.kind != DTypeKind::Float64 {
            return Err(HDF5Error::TypeMismatch);
        }
        let shape = ds.shape.iter().map(|d| *d as usize).collect();
        Ok((shape, from_u64_le(&ds.data)))
    }

    /// Read IEEE binary32 lanes and the dataspace shape.
    ///
    /// Errors: same as [`Self::read_f64`].
    pub fn read_f32(&self, path: &str) -> Result<(Vec<usize>, Vec<u32>)> {
        self.read_ieee32(path)
    }

    /// IEEE binary32 lanes and dataspace shape. Same as [`Self::read_f32`].
    pub fn read_ieee32(&self, path: &str) -> Result<(Vec<usize>, Vec<u32>)> {
        let ds = self.dataset(path)?;
        if ds.kind != DTypeKind::Float32 {
            return Err(HDF5Error::TypeMismatch);
        }
        let shape = ds.shape.iter().map(|d| *d as usize).collect();
        Ok((shape, from_u32_le(&ds.data)))
    }

    /// Read opaque payload, shape, and element size.
    ///
    /// Errors: [`HDF5Error::NotFound`], [`HDF5Error::ChunkedNotSupported`],
    /// [`HDF5Error::TypeMismatch`].
    pub fn read_opaque(&self, path: &str) -> Result<(Vec<usize>, usize, Vec<u8>)> {
        let ds = self.dataset(path)?;
        let DTypeKind::Opaque(sz) = ds.kind else {
            return Err(HDF5Error::TypeMismatch);
        };
        let shape = ds.shape.iter().map(|d| *d as usize).collect();
        Ok((shape, sz, ds.data.clone()))
    }

    /// Rows `[row_start, row_end)` of a rank-1 or rank-2 IEEE binary64 dataset.
    ///
    /// Errors: [`HDF5Error::NotFound`], [`HDF5Error::ChunkedNotSupported`],
    /// [`HDF5Error::TypeMismatch`], [`HDF5Error::ShapeMismatch`].
    pub fn read_f64_slice(
        &self,
        path: &str,
        row_start: usize,
        row_end: usize,
    ) -> Result<(Vec<usize>, Vec<u64>)> {
        let (shape, bits) = self.read_f64(path)?;
        if row_end < row_start {
            return Err(HDF5Error::ShapeMismatch);
        }
        match shape.len() {
            1 => {
                if row_end > shape[0] {
                    return Err(HDF5Error::ShapeMismatch);
                }
                Ok((
                    alloc::vec![row_end - row_start],
                    bits[row_start..row_end].to_vec(),
                ))
            }
            2 => {
                if row_end > shape[0] {
                    return Err(HDF5Error::ShapeMismatch);
                }
                let cols = shape[1];
                let start = row_start.saturating_mul(cols);
                let end = row_end.saturating_mul(cols);
                Ok((
                    alloc::vec![row_end - row_start, cols],
                    bits[start..end].to_vec(),
                ))
            }
            _ => Err(HDF5Error::RankNotSupported),
        }
    }

    /// Append rows to an existing IEEE binary64 dataset. Trailing dimensions
    /// must match. Rank-1 appends to the single axis.
    ///
    /// Errors: [`HDF5Error::NotFound`], [`HDF5Error::ShapeMismatch`],
    /// [`HDF5Error::TypeMismatch`], [`HDF5Error::ChunkedNotSupported`].
    pub fn append_f64(&mut self, path: &str, shape: &[usize], bits: &[u64]) -> Result<()> {
        self.append_ieee64(path, shape, bits)
    }

    /// Append IEEE binary64 rows. Same as [`Self::append_f64`].
    pub fn append_ieee64(&mut self, path: &str, shape: &[usize], bits: &[u64]) -> Result<()> {
        let p = FileModel::normalize(path)?;
        let ds = self
            .model
            .datasets
            .get_mut(&p)
            .ok_or_else(|| HDF5Error::NotFound(p.clone()))?;
        if ds.chunked {
            return Err(HDF5Error::ChunkedNotSupported);
        }
        if ds.kind != DTypeKind::Float64 {
            return Err(HDF5Error::TypeMismatch);
        }
        let extra = lanes_u64(bits);
        let expect: usize = shape
            .iter()
            .try_fold(1usize, |a, b| a.checked_mul(*b))
            .ok_or(HDF5Error::ShapeMismatch)?;
        if bits.len() != expect {
            return Err(HDF5Error::ShapeMismatch);
        }
        match (ds.shape.len(), shape.len()) {
            (1, 1) => {
                ds.shape[0] = ds.shape[0].saturating_add(shape[0] as u64);
                ds.data.extend_from_slice(&extra);
            }
            (2, 2) => {
                if ds.shape[1] as usize != shape[1] {
                    return Err(HDF5Error::ShapeMismatch);
                }
                ds.shape[0] = ds.shape[0].saturating_add(shape[0] as u64);
                ds.data.extend_from_slice(&extra);
            }
            _ => return Err(HDF5Error::ShapeMismatch),
        }
        Ok(())
    }

    /// Read a string attribute on a dataset.
    ///
    /// Errors: [`HDF5Error::NotFound`].
    pub fn read_attr_str(&self, path: &str, name: &str) -> Result<String> {
        let ds = self.dataset(path)?;
        ds.attrs
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| v.clone())
            .ok_or_else(|| HDF5Error::NotFound(name.to_string()))
    }

    fn dataset(&self, path: &str) -> Result<&DatasetRec> {
        let p = FileModel::normalize(path)?;
        let ds = self
            .model
            .datasets
            .get(&p)
            .ok_or_else(|| HDF5Error::NotFound(p))?;
        if ds.chunked {
            return Err(HDF5Error::ChunkedNotSupported);
        }
        Ok(ds)
    }

    fn write_bits(
        &mut self,
        path: &str,
        shape: &[usize],
        kind: DTypeKind,
        data: Vec<u8>,
    ) -> Result<()> {
        if shape.len() > 2 || shape.is_empty() {
            return Err(HDF5Error::RankNotSupported);
        }
        let n: usize = shape
            .iter()
            .try_fold(1usize, |a, b| a.checked_mul(*b))
            .ok_or(HDF5Error::ShapeMismatch)?;
        let need = n
            .checked_mul(kind.elem_size())
            .ok_or(HDF5Error::ShapeMismatch)?;
        if data.len() != need {
            return Err(HDF5Error::ShapeMismatch);
        }
        let rec = DatasetRec {
            shape: shape.iter().map(|d| *d as u64).collect(),
            kind,
            data,
            attrs: FileModel::default_attrs(),
            chunked: false,
        };
        self.model.put_dataset(path, rec)
    }
}

fn lanes_u64(bits: &[u64]) -> Vec<u8> {
    let mut o = Vec::with_capacity(bits.len().saturating_mul(8));
    for b in bits {
        o.extend_from_slice(&b.to_le_bytes());
    }
    o
}

fn lanes_u32(bits: &[u32]) -> Vec<u8> {
    let mut o = Vec::with_capacity(bits.len().saturating_mul(4));
    for b in bits {
        o.extend_from_slice(&b.to_le_bytes());
    }
    o
}

fn from_u64_le(data: &[u8]) -> Vec<u64> {
    data.chunks_exact(8)
        .map(|c| u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
        .collect()
}

fn from_u32_le(data: &[u8]) -> Vec<u32> {
    data.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Dataset paths in `file`.
pub fn list_datasets(file: &Hdf5File) -> Vec<String> {
    file.list_datasets()
}

/// Group paths in `file`.
pub fn list_groups(file: &Hdf5File) -> Vec<String> {
    file.list_groups()
}

/// Shape of dataset `path`.
pub fn dataset_shape(file: &Hdf5File, path: &str) -> Result<Vec<usize>> {
    file.dataset_shape(path)
}

/// Datatype of dataset `path`.
pub fn dataset_dtype(file: &Hdf5File, path: &str) -> Result<HDF5DType> {
    file.dataset_dtype(path)
}
