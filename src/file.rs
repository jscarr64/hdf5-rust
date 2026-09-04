//! Public [`Hdf5File`] API: in-memory image, optional `std` paths.

use alloc::string::{String, ToString};
use alloc::vec::Vec;

use crate::error::{HDF5Error, Result};
use crate::model::{DTypeKind, DatasetRec, FileModel};
use crate::{HDF5DType, HDF5_MAX_DIMS, HDF5_MAX_NAME_LEN};

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
    /// Errors: [`HDF5Error::NotFound`]. Shape is reported for chunked and
    /// filtered datasets; reading those still returns a named error.
    pub fn dataset_shape(&self, path: &str) -> Result<Vec<usize>> {
        let ds = self.dataset_meta(path)?;
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
    /// product of `shape` (empty `shape` is a scalar, one element).
    /// Rank 0 through [`HDF5_MAX_DIMS`].
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

    /// Write signed 8-bit lanes.
    pub fn write_i8(&mut self, path: &str, shape: &[usize], bits: &[i8]) -> Result<()> {
        self.write_bits(path, shape, DTypeKind::Int8, i8_lanes(bits))
    }
    /// Write signed 16-bit little-endian lanes.
    pub fn write_i16(&mut self, path: &str, shape: &[usize], bits: &[i16]) -> Result<()> {
        self.write_bits(path, shape, DTypeKind::Int16, i16_lanes(bits))
    }
    /// Write signed 32-bit little-endian lanes.
    pub fn write_i32(&mut self, path: &str, shape: &[usize], bits: &[i32]) -> Result<()> {
        self.write_bits(path, shape, DTypeKind::Int32, i32_lanes(bits))
    }
    /// Write signed 64-bit little-endian lanes.
    pub fn write_i64(&mut self, path: &str, shape: &[usize], bits: &[i64]) -> Result<()> {
        self.write_bits(path, shape, DTypeKind::Int64, i64_lanes(bits))
    }
    /// Write unsigned 8-bit lanes.
    pub fn write_u8(&mut self, path: &str, shape: &[usize], bits: &[u8]) -> Result<()> {
        self.write_bits(path, shape, DTypeKind::UInt8, bits.to_vec())
    }
    /// Write unsigned 16-bit little-endian lanes.
    pub fn write_u16(&mut self, path: &str, shape: &[usize], bits: &[u16]) -> Result<()> {
        self.write_bits(path, shape, DTypeKind::UInt16, u16_lanes(bits))
    }
    /// Write unsigned 32-bit little-endian lanes.
    pub fn write_u32(&mut self, path: &str, shape: &[usize], bits: &[u32]) -> Result<()> {
        self.write_bits(path, shape, DTypeKind::UInt32, lanes_u32(bits))
    }
    /// Write unsigned 64-bit little-endian lanes.
    pub fn write_u64(&mut self, path: &str, shape: &[usize], bits: &[u64]) -> Result<()> {
        self.write_bits(path, shape, DTypeKind::UInt64, lanes_u64(bits))
    }

    /// Read IEEE binary64 lanes and the dataspace shape.
    ///
    /// Errors: [`HDF5Error::NotFound`], [`HDF5Error::ChunkedNotSupported`],
    /// [`HDF5Error::FilteredNotSupported`], [`HDF5Error::TypeMismatch`].
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

    /// Read signed 8-bit lanes.
    pub fn read_i8(&self, path: &str) -> Result<(Vec<usize>, Vec<i8>)> {
        self.read_kind(path, DTypeKind::Int8, from_i8)
    }
    /// Read signed 16-bit little-endian lanes.
    pub fn read_i16(&self, path: &str) -> Result<(Vec<usize>, Vec<i16>)> {
        self.read_kind(path, DTypeKind::Int16, from_i16_le)
    }
    /// Read signed 32-bit little-endian lanes.
    pub fn read_i32(&self, path: &str) -> Result<(Vec<usize>, Vec<i32>)> {
        self.read_kind(path, DTypeKind::Int32, from_i32_le)
    }
    /// Read signed 64-bit little-endian lanes.
    pub fn read_i64(&self, path: &str) -> Result<(Vec<usize>, Vec<i64>)> {
        self.read_kind(path, DTypeKind::Int64, from_i64_le)
    }
    /// Read unsigned 8-bit lanes.
    pub fn read_u8(&self, path: &str) -> Result<(Vec<usize>, Vec<u8>)> {
        let ds = self.dataset(path)?;
        if ds.kind != DTypeKind::UInt8 {
            return Err(HDF5Error::TypeMismatch);
        }
        let shape = ds.shape.iter().map(|d| *d as usize).collect();
        Ok((shape, ds.data.clone()))
    }
    /// Read unsigned 16-bit little-endian lanes.
    pub fn read_u16(&self, path: &str) -> Result<(Vec<usize>, Vec<u16>)> {
        self.read_kind(path, DTypeKind::UInt16, from_u16_le)
    }
    /// Read unsigned 32-bit little-endian lanes.
    pub fn read_u32(&self, path: &str) -> Result<(Vec<usize>, Vec<u32>)> {
        self.read_kind(path, DTypeKind::UInt32, from_u32_le)
    }
    /// Read unsigned 64-bit little-endian lanes.
    pub fn read_u64(&self, path: &str) -> Result<(Vec<usize>, Vec<u64>)> {
        self.read_kind(path, DTypeKind::UInt64, from_u64_le)
    }

    /// Rows `[row_start, row_end)` along the first axis of an IEEE binary64 dataset.
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
        if shape.is_empty() {
            return Err(HDF5Error::ShapeMismatch);
        }
        if row_end > shape[0] {
            return Err(HDF5Error::ShapeMismatch);
        }
        let rest: usize = shape[1..]
            .iter()
            .try_fold(1usize, |a, b| a.checked_mul(*b))
            .ok_or(HDF5Error::ShapeMismatch)?;
        let start = row_start.saturating_mul(rest);
        let end = row_end.saturating_mul(rest);
        let mut out_shape = alloc::vec![row_end - row_start];
        out_shape.extend_from_slice(&shape[1..]);
        Ok((out_shape, bits[start..end].to_vec()))
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
        self.append_kind(
            path,
            shape,
            DTypeKind::Float64,
            &lanes_u64(bits),
            bits.len(),
        )
    }

    /// Append IEEE binary32 rows.
    pub fn append_f32(&mut self, path: &str, shape: &[usize], bits: &[u32]) -> Result<()> {
        self.append_ieee32(path, shape, bits)
    }

    /// Append IEEE binary32 rows. Same as [`Self::append_f32`].
    pub fn append_ieee32(&mut self, path: &str, shape: &[usize], bits: &[u32]) -> Result<()> {
        self.append_kind(
            path,
            shape,
            DTypeKind::Float32,
            &lanes_u32(bits),
            bits.len(),
        )
    }

    /// Append signed 32-bit rows.
    pub fn append_i32(&mut self, path: &str, shape: &[usize], bits: &[i32]) -> Result<()> {
        self.append_kind(path, shape, DTypeKind::Int32, &i32_lanes(bits), bits.len())
    }

    /// Append unsigned 64-bit rows.
    pub fn append_u64(&mut self, path: &str, shape: &[usize], bits: &[u64]) -> Result<()> {
        self.append_kind(path, shape, DTypeKind::UInt64, &lanes_u64(bits), bits.len())
    }

    /// Append opaque records. `data.len()` must equal `shape.product() * elem_size`
    /// of the existing dataset.
    pub fn append_opaque(&mut self, path: &str, shape: &[usize], data: &[u8]) -> Result<()> {
        let p = FileModel::normalize(path)?;
        let elem = match self.model.datasets.get(&p) {
            Some(ds) => ds.kind.elem_size(),
            None => return Err(HDF5Error::NotFound(p)),
        };
        let n: usize = if shape.is_empty() {
            1
        } else {
            shape
                .iter()
                .try_fold(1usize, |a, b| a.checked_mul(*b))
                .ok_or(HDF5Error::ShapeMismatch)?
        };
        let need = n.checked_mul(elem).ok_or(HDF5Error::ShapeMismatch)?;
        if data.len() != need {
            return Err(HDF5Error::ShapeMismatch);
        }
        self.append_kind(path, shape, DTypeKind::Opaque(elem), data, n)
    }

    /// Read a string attribute on a dataset.
    ///
    /// Errors: [`HDF5Error::NotFound`].
    pub fn read_attr_str(&self, path: &str, name: &str) -> Result<String> {
        let ds = self.dataset_meta(path)?;
        ds.attrs
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| v.clone())
            .ok_or_else(|| HDF5Error::NotFound(name.to_string()))
    }

    /// String attributes on `path` (name, value), in file order.
    pub fn list_attrs(&self, path: &str) -> Result<Vec<(String, String)>> {
        Ok(self.dataset_meta(path)?.attrs.clone())
    }

    /// Write or replace a string attribute on a dataset.
    ///
    /// Errors: [`HDF5Error::NotFound`], [`HDF5Error::NameTooLong`].
    pub fn write_attr_str(&mut self, path: &str, name: &str, value: &str) -> Result<()> {
        if name.len() > HDF5_MAX_NAME_LEN || value.len() > HDF5_MAX_NAME_LEN {
            return Err(HDF5Error::NameTooLong);
        }
        let p = FileModel::normalize(path)?;
        let ds = self
            .model
            .datasets
            .get_mut(&p)
            .ok_or_else(|| HDF5Error::NotFound(p))?;
        if let Some(slot) = ds.attrs.iter_mut().find(|(n, _)| n == name) {
            slot.1 = value.to_string();
        } else {
            ds.attrs.push((name.to_string(), value.to_string()));
        }
        Ok(())
    }

    fn dataset_meta(&self, path: &str) -> Result<&DatasetRec> {
        let p = FileModel::normalize(path)?;
        self.model
            .datasets
            .get(&p)
            .ok_or_else(|| HDF5Error::NotFound(p))
    }

    fn dataset(&self, path: &str) -> Result<&DatasetRec> {
        let ds = self.dataset_meta(path)?;
        if ds.filtered {
            return Err(HDF5Error::FilteredNotSupported);
        }
        if ds.chunked {
            return Err(HDF5Error::ChunkedNotSupported);
        }
        Ok(ds)
    }

    fn read_kind<T>(
        &self,
        path: &str,
        kind: DTypeKind,
        decode: fn(&[u8]) -> Vec<T>,
    ) -> Result<(Vec<usize>, Vec<T>)> {
        let ds = self.dataset(path)?;
        if ds.kind != kind {
            return Err(HDF5Error::TypeMismatch);
        }
        let shape = ds.shape.iter().map(|d| *d as usize).collect();
        Ok((shape, decode(&ds.data)))
    }

    fn append_kind(
        &mut self,
        path: &str,
        shape: &[usize],
        kind: DTypeKind,
        extra: &[u8],
        n_elems: usize,
    ) -> Result<()> {
        let p = FileModel::normalize(path)?;
        let ds = self
            .model
            .datasets
            .get_mut(&p)
            .ok_or_else(|| HDF5Error::NotFound(p.clone()))?;
        if ds.filtered {
            return Err(HDF5Error::FilteredNotSupported);
        }
        if ds.chunked {
            return Err(HDF5Error::ChunkedNotSupported);
        }
        if ds.kind != kind {
            return Err(HDF5Error::TypeMismatch);
        }
        let expect: usize = if shape.is_empty() {
            1
        } else {
            shape
                .iter()
                .try_fold(1usize, |a, b| a.checked_mul(*b))
                .ok_or(HDF5Error::ShapeMismatch)?
        };
        if n_elems != expect {
            return Err(HDF5Error::ShapeMismatch);
        }
        if ds.shape.is_empty() {
            return Err(HDF5Error::ShapeMismatch);
        }
        if ds.shape.len() != shape.len() {
            return Err(HDF5Error::ShapeMismatch);
        }
        if ds.shape.len() >= 2
            && ds.shape[1..] != shape[1..].iter().map(|d| *d as u64).collect::<Vec<_>>()
        {
            return Err(HDF5Error::ShapeMismatch);
        }
        ds.shape[0] = ds.shape[0].saturating_add(shape[0] as u64);
        ds.data.extend_from_slice(extra);
        Ok(())
    }

    fn write_bits(
        &mut self,
        path: &str,
        shape: &[usize],
        kind: DTypeKind,
        data: Vec<u8>,
    ) -> Result<()> {
        if shape.len() > HDF5_MAX_DIMS {
            return Err(HDF5Error::RankNotSupported);
        }
        let n: usize = if shape.is_empty() {
            1
        } else {
            shape
                .iter()
                .try_fold(1usize, |a, b| a.checked_mul(*b))
                .ok_or(HDF5Error::ShapeMismatch)?
        };
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
            filtered: false,
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

fn u16_lanes(bits: &[u16]) -> Vec<u8> {
    let mut o = Vec::with_capacity(bits.len().saturating_mul(2));
    for b in bits {
        o.extend_from_slice(&b.to_le_bytes());
    }
    o
}

fn i8_lanes(bits: &[i8]) -> Vec<u8> {
    bits.iter().map(|b| b.to_le_bytes()[0]).collect()
}

fn i16_lanes(bits: &[i16]) -> Vec<u8> {
    let mut o = Vec::with_capacity(bits.len().saturating_mul(2));
    for b in bits {
        o.extend_from_slice(&b.to_le_bytes());
    }
    o
}

fn i32_lanes(bits: &[i32]) -> Vec<u8> {
    let mut o = Vec::with_capacity(bits.len().saturating_mul(4));
    for b in bits {
        o.extend_from_slice(&b.to_le_bytes());
    }
    o
}

fn i64_lanes(bits: &[i64]) -> Vec<u8> {
    let mut o = Vec::with_capacity(bits.len().saturating_mul(8));
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

fn from_u16_le(data: &[u8]) -> Vec<u16> {
    data.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

fn from_i8(data: &[u8]) -> Vec<i8> {
    data.iter().map(|b| i8::from_le_bytes([*b])).collect()
}

fn from_i16_le(data: &[u8]) -> Vec<i16> {
    data.chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect()
}

fn from_i32_le(data: &[u8]) -> Vec<i32> {
    data.chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn from_i64_le(data: &[u8]) -> Vec<i64> {
    data.chunks_exact(8)
        .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
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
