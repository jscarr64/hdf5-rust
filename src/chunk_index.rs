//! Layout v4 chunk indexes: single-chunk and fixed array (FAHD/FADB).

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

    let unc_size = chunk_dims
        .iter()
        .try_fold(1u64, |a, &b| a.checked_mul(u64::from(b)))
        .ok_or(HDF5Error::ShapeMismatch)?;
    let unc_size_u32 = u32::try_from(unc_size).map_err(|_| HDF5Error::ShapeMismatch)?;

    let spatial_chunk: Vec<u32> = if chunk_dims.len() == shape.len() + 1 {
        chunk_dims[..shape.len()].to_vec()
    } else if chunk_dims.len() == shape.len() {
        chunk_dims.to_vec()
    } else {
        return Err(HDF5Error::ShapeMismatch);
    };
    if spatial_chunk.len() != shape.len() {
        return Err(HDF5Error::ShapeMismatch);
    }
    let nchunks_per_dim: Vec<u64> = shape
        .iter()
        .zip(spatial_chunk.iter())
        .map(|(&s, &c)| {
            if c == 0 {
                0
            } else {
                s.saturating_add(u64::from(c) - 1) / u64::from(c)
            }
        })
        .collect();
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

fn linear_chunk_offset(
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
