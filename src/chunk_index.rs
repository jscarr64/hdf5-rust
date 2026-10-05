//! Layout v4/v5 chunk indexes: single-chunk, implicit, fixed array,
//! extensible array, and version-2 B-tree.

use alloc::vec::Vec;

use crate::btree::ChunkRef;
use crate::buf::Reader;
use crate::error::{HDF5Error, Result};
use crate::HDF5_MAX_WALK_DEPTH;

/// Fixed-array header signature.
pub const HDF5_FAHD_SIGNATURE: [u8; 4] = *b"FAHD";
/// Fixed-array data-block signature.
pub const HDF5_FADB_SIGNATURE: [u8; 4] = *b"FADB";

/// Layout v4 chunk index kinds this crate walks.
#[derive(Clone, Debug)]
pub enum ChunkIndex {
    /// Version-1 B-tree address (layout v1–v3).
    BTreeV1 { addr: u64 },
    /// Single chunk: `addr` is the raw chunk address.
    Single {
        addr: u64,
        /// Compressed size when filters are present; `None` means use full chunk nbytes.
        nbytes: Option<u32>,
        filter_mask: u32,
    },
    /// Fixed array header address.
    FixedArray { addr: u64 },
    /// Implicit index: chunks packed contiguously at `addr` in C order.
    Implicit { addr: u64 },
    /// Extensible-array header address (`EAHD`).
    ExtensibleArray { addr: u64 },
    /// Version-2 B-tree header address (`BTHD`).
    BTreeV2 { addr: u64 },
}

/// Build a single-chunk reference. `ndims` includes the trailing element-size dim.
pub fn single_chunk_ref(
    addr: u64,
    nbytes: Option<u32>,
    filter_mask: u32,
    ndims: usize,
    full_chunk_nbytes: u32,
) -> ChunkRef {
    ChunkRef {
        size: nbytes.unwrap_or(full_chunk_nbytes),
        filter_mask,
        offset: alloc::vec![0u64; ndims],
        addr,
    }
}

/// Walk a fixed-array chunk index.
///
/// `shape` is the dataset dataspace; `chunk_dims` includes the trailing element-size
/// dimension from the layout message.
pub fn walk_fixed_array(
    data: &[u8],
    header_addr: u64,
    offset_size: u8,
    length_size: u8,
    shape: &[u64],
    chunk_dims: &[u32],
    depth: usize,
) -> Result<Vec<ChunkRef>> {
    if depth > HDF5_MAX_WALK_DEPTH {
        return Err(HDF5Error::InvalidHeader);
    }
    let mut r = Reader::new(data, offset_size, length_size)?.at(header_addr)?;
    let sig = r.bytes(4)?;
    if sig != HDF5_FAHD_SIGNATURE {
        return Err(HDF5Error::InvalidHeader);
    }
    let version = r.u8()?;
    if version != 0 {
        return Err(HDF5Error::UnsupportedVersion(version));
    }
    let _client = r.u8()?;
    let entry_size = r.u8()? as usize;
    let _page_bits = r.u8()?;
    let max_nelmts = r.length()?;
    let dblk_addr = r.addr()?;
    let _sum = r.u32()?;
    if r.is_undef(dblk_addr) || max_nelmts == 0 {
        return Err(HDF5Error::ChunkedNotSupported);
    }
    if entry_size == 0 {
        return Err(HDF5Error::InvalidHeader);
    }

    let mut r = Reader::new(data, offset_size, length_size)?.at(dblk_addr)?;
    let sig = r.bytes(4)?;
    if sig != HDF5_FADB_SIGNATURE {
        return Err(HDF5Error::InvalidHeader);
    }
    let version = r.u8()?;
    if version != 0 {
        return Err(HDF5Error::UnsupportedVersion(version));
    }
    let _client = r.u8()?;
    let _hdr = r.addr()?;

    let unc_size_u32 = full_chunk_nbytes(chunk_dims)?;
    let spatial_chunk = spatial_chunk_dims(shape, chunk_dims)?;
    let nchunks_per_dim = nchunks_per_dim(shape, &spatial_chunk)?;
    let ndims_key = chunk_dims.len();

    let mut chunks = Vec::with_capacity(max_nelmts as usize);
    for i in 0..max_nelmts {
        let addr = r.sized_uint(offset_size)?;
        let (size, filter_mask) = if entry_size == offset_size as usize {
            (unc_size_u32, 0u32)
        } else if entry_size > offset_size as usize + 4 {
            let size_w = entry_size - offset_size as usize - 4;
            let size = r.sized_uint(size_w as u8)? as u32;
            let filter_mask = r.u32()?;
            (size, filter_mask)
        } else {
            return Err(HDF5Error::ChunkedNotSupported);
        };
        if r.is_undef(addr) || size == 0 {
            continue;
        }
        let offset = linear_chunk_offset(i, &nchunks_per_dim, &spatial_chunk, ndims_key)?;
        chunks.push(ChunkRef {
            size,
            filter_mask,
            offset,
            addr,
        });
    }
    Ok(chunks)
}

/// Spatial chunk shape (drops the trailing element-size dimension when present).
pub(crate) fn spatial_chunk_dims(shape: &[u64], chunk_dims: &[u32]) -> Result<Vec<u32>> {
    let spatial = if chunk_dims.len() == shape.len() + 1 {
        &chunk_dims[..shape.len()]
    } else if chunk_dims.len() == shape.len() || shape.is_empty() {
        chunk_dims
    } else {
        return Err(HDF5Error::ShapeMismatch);
    };
    if !shape.is_empty() && spatial.len() != shape.len() {
        return Err(HDF5Error::ShapeMismatch);
    }
    Ok(spatial.to_vec())
}

/// Number of chunks along each spatial dimension.
pub(crate) fn nchunks_per_dim(shape: &[u64], spatial_chunk: &[u32]) -> Result<Vec<u64>> {
    if spatial_chunk.len() != shape.len() {
        return Err(HDF5Error::ShapeMismatch);
    }
    Ok(shape
        .iter()
        .zip(spatial_chunk.iter())
        .map(|(&s, &c)| {
            if c == 0 {
                0
            } else {
                s.saturating_add(u64::from(c) - 1) / u64::from(c)
            }
        })
        .collect())
}

/// Product of per-dimension chunk counts.
pub(crate) fn chunk_count(nchunks_per_dim: &[u64]) -> Result<u64> {
    nchunks_per_dim
        .iter()
        .try_fold(1u64, |a, &b| a.checked_mul(b))
        .ok_or(HDF5Error::ShapeMismatch)
}

/// Uncompressed size of one chunk, in bytes.
pub(crate) fn full_chunk_nbytes(chunk_dims: &[u32]) -> Result<u32> {
    let unc = chunk_dims
        .iter()
        .try_fold(1u64, |a, &b| a.checked_mul(u64::from(b)))
        .ok_or(HDF5Error::ShapeMismatch)?;
    u32::try_from(unc).map_err(|_| HDF5Error::ShapeMismatch)
}

/// Implicit chunk index: `addr` points at chunk 0, and chunk `i` follows it
/// by `i * full_chunk_nbytes` bytes. Filters are not representable here.
pub fn walk_implicit(
    addr: u64,
    shape: &[u64],
    chunk_dims: &[u32],
    full_chunk_nbytes: u32,
) -> Result<Vec<ChunkRef>> {
    let spatial = spatial_chunk_dims(shape, chunk_dims)?;
    let nper = nchunks_per_dim(shape, &spatial)?;
    let n = chunk_count(&nper)?;
    let ndims_key = chunk_dims.len();
    let step = u64::from(full_chunk_nbytes);
    let mut chunks = Vec::with_capacity(usize::try_from(n).unwrap_or(0));
    for i in 0..n {
        let offset = linear_chunk_offset(i, &nper, &spatial, ndims_key)?;
        let at = addr
            .checked_add(i.checked_mul(step).ok_or(HDF5Error::ShapeMismatch)?)
            .ok_or(HDF5Error::ShapeMismatch)?;
        chunks.push(ChunkRef {
            size: full_chunk_nbytes,
            filter_mask: 0,
            offset,
            addr: at,
        });
    }
    Ok(chunks)
}

pub(crate) fn linear_chunk_offset(
    index: u64,
    nchunks_per_dim: &[u64],
    spatial_chunk: &[u32],
    ndims_key: usize,
) -> Result<Vec<u64>> {
    let rank = nchunks_per_dim.len();
    if rank == 0 {
        return Ok(alloc::vec![0u64; ndims_key]);
    }
    let mut coords = alloc::vec![0u64; rank];
    let mut rem = index;
    // C order: last spatial dim varies fastest.
    for d in (0..rank).rev() {
        let n = nchunks_per_dim[d];
        if n == 0 {
            return Err(HDF5Error::ShapeMismatch);
        }
        coords[d] = rem % n;
        rem /= n;
    }
    if rem != 0 {
        return Err(HDF5Error::InvalidHeader);
    }
    let mut offset = alloc::vec![0u64; ndims_key];
    for d in 0..rank {
        offset[d] = coords[d].saturating_mul(u64::from(spatial_chunk[d]));
    }
    Ok(offset)
}
