//! Version-2 B-tree chunk index (`BTHD` / `BTLF` / `BTIN`).
//!
//! Record type 10 is an unfiltered chunk (address, then one 8-byte scaled
//! offset per spatial dimension). Record type 11 is filtered (address,
//! stored size, filter mask, then the same offsets). Internal-node records
//! are real chunks, not just separators.

use alloc::vec::Vec;

use crate::btree::ChunkRef;
use crate::buf::Reader;
use crate::checksum::lookup3;
use crate::chunk_index::spatial_chunk_dims;
use crate::error::{HDF5Error, Result};
use crate::HDF5_MAX_WALK_DEPTH;

const BTHD: [u8; 4] = *b"BTHD";
const BTLF: [u8; 4] = *b"BTLF";
const BTIN: [u8; 4] = *b"BTIN";

const TYPE_UNFILT: u8 = 10;
const TYPE_FILT: u8 = 11;

/// Bytes of metadata before records, including the trailing checksum slot
/// that HDF5 counts in the prefix when sizing a node.
const PREFIX_WITH_CKSUM: u64 = 10;

struct Header {
    record_type: u8,
    node_size: u32,
    record_size: u16,
    depth: u16,
    root_addr: u64,
    root_nrec: u16,
    total_nrec: u64,
}

struct Level {
    max_nrec: u64,
    cum_max_nrec: u64,
    cum_max_nrec_size: u8,
}

/// Walk a version-2 B-tree of raw-data chunks.
pub fn walk_btree_v2(
    data: &[u8],
    header_addr: u64,
    offset_size: u8,
    length_size: u8,
    shape: &[u64],
    chunk_dims: &[u32],
    full_chunk: u32,
) -> Result<Vec<ChunkRef>> {
    let hdr = read_header(data, header_addr, offset_size, length_size)?;
    let rdr = Reader::new(data, offset_size, length_size)?;
    if rdr.is_undef(hdr.root_addr) || hdr.total_nrec == 0 {
        return Ok(Vec::new());
    }
    if hdr.record_type != TYPE_UNFILT && hdr.record_type != TYPE_FILT {
        return Err(HDF5Error::ChunkedNotSupported);
    }
    let spatial = spatial_chunk_dims(shape, chunk_dims)?;
    let levels = node_levels(&hdr, offset_size)?;
    let walk = NodeWalk {
        data,
        hdr: &hdr,
        levels: &levels,
        offset_size,
        length_size,
        spatial: &spatial,
        ndims_key: chunk_dims.len(),
        full_chunk,
    };
    let mut chunks = Vec::new();
    walk_node(
        &walk,
        hdr.root_addr,
        u64::from(hdr.root_nrec),
        hdr.depth,
        &mut chunks,
        0,
    )?;
    if u64::try_from(chunks.len()).map_err(|_| HDF5Error::InvalidHeader)? != hdr.total_nrec {
        return Err(HDF5Error::InvalidHeader);
    }
    Ok(chunks)
}

fn read_header(data: &[u8], addr: u64, offset_size: u8, length_size: u8) -> Result<Header> {
    let sa = offset_size as usize;
    let ss = length_size as usize;
    let body = 4 + 1 + 1 + 4 + 2 + 2 + 1 + 1 + sa + 2 + ss;
    let bytes = Reader::new(data, offset_size, length_size)?.slice_at(addr, (body + 4) as u64)?;
    if bytes[..4] != BTHD {
        return Err(HDF5Error::InvalidHeader);
    }
    if bytes[4] != 0 {
        return Err(HDF5Error::UnsupportedVersion(bytes[4]));
    }
    check_sum(&bytes[..body], &bytes[body..body + 4])?;
    let mut p = 5;
    let record_type = bytes[p];
    p += 1;
    let node_size = u32::from_le_bytes([bytes[p], bytes[p + 1], bytes[p + 2], bytes[p + 3]]);
    p += 4;
    let record_size = u16::from_le_bytes([bytes[p], bytes[p + 1]]);
    p += 2;
    let depth = u16::from_le_bytes([bytes[p], bytes[p + 1]]);
    p += 2;
    p += 2; // split / merge percents; not needed to read
    let root_addr = take_le(&bytes[p..], sa)?;
    p += sa;
    let root_nrec = u16::from_le_bytes([bytes[p], bytes[p + 1]]);
    p += 2;
    let total_nrec = take_le(&bytes[p..], ss)?;
    if node_size < PREFIX_WITH_CKSUM as u32 || record_size == 0 || depth > 64 {
        return Err(HDF5Error::InvalidHeader);
    }
    Ok(Header {
        record_type,
        node_size,
        record_size,
        depth,
        root_addr,
        root_nrec,
        total_nrec,
    })
}

fn node_levels(hdr: &Header, offset_size: u8) -> Result<Vec<Level>> {
    let leaf_max = (u64::from(hdr.node_size) - PREFIX_WITH_CKSUM) / u64::from(hdr.record_size);
    if leaf_max == 0 {
        return Err(HDF5Error::InvalidHeader);
    }
    let max_nrec_size = enc_size(leaf_max);
    let mut levels = Vec::new();
    levels.push(Level {
        max_nrec: leaf_max,
        cum_max_nrec: leaf_max,
        cum_max_nrec_size: 0,
    });
    for d in 1..=hdr.depth {
        let prev = &levels[d as usize - 1];
        let ptr =
            u64::from(offset_size) + u64::from(max_nrec_size) + u64::from(prev.cum_max_nrec_size);
        let denom = u64::from(hdr.record_size) + ptr;
        let numer = u64::from(hdr.node_size) - PREFIX_WITH_CKSUM - ptr;
        if denom == 0 || numer / denom == 0 {
            return Err(HDF5Error::InvalidHeader);
        }
        let max_nrec = numer / denom;
        let cum = max_nrec
            .checked_add(1)
            .and_then(|n| n.checked_mul(prev.cum_max_nrec))
            .and_then(|n| n.checked_add(max_nrec))
            .ok_or(HDF5Error::InvalidHeader)?;
        levels.push(Level {
            max_nrec,
            cum_max_nrec: cum,
            cum_max_nrec_size: enc_size(cum),
        });
    }
    Ok(levels)
}

struct NodeWalk<'a> {
    data: &'a [u8],
    hdr: &'a Header,
    levels: &'a [Level],
    offset_size: u8,
    length_size: u8,
    spatial: &'a [u32],
    ndims_key: usize,
    full_chunk: u32,
}

fn walk_node(
    w: &NodeWalk<'_>,
    addr: u64,
    nrec: u64,
    depth: u16,
    out: &mut Vec<ChunkRef>,
    walk_depth: usize,
) -> Result<()> {
    if walk_depth > HDF5_MAX_WALK_DEPTH {
        return Err(HDF5Error::InvalidHeader);
    }
    let rdr = Reader::new(w.data, w.offset_size, w.length_size)?;
    if rdr.is_undef(addr) {
        return Err(HDF5Error::InvalidHeader);
    }
    let nrec_us = usize::try_from(nrec).map_err(|_| HDF5Error::InvalidHeader)?;
    if depth == 0 {
        let body = 6 + nrec_us * w.hdr.record_size as usize;
        let bytes = rdr.slice_at(addr, (body + 4) as u64)?;
        if bytes[..4] != BTLF || bytes[4] != 0 || bytes[5] != w.hdr.record_type {
            return Err(HDF5Error::InvalidHeader);
        }
        check_sum(&bytes[..body], &bytes[body..body + 4])?;
        for i in 0..nrec_us {
            let start = 6 + i * w.hdr.record_size as usize;
            let rec = &bytes[start..start + w.hdr.record_size as usize];
            out.push(decode_record(
                rec,
                w.hdr.record_type,
                w.offset_size,
                w.spatial,
                w.ndims_key,
                w.full_chunk,
            )?);
        }
        return Ok(());
    }
    let child_extra = w
        .levels
        .get(depth as usize - 1)
        .ok_or(HDF5Error::InvalidHeader)?
        .cum_max_nrec_size as usize;
    let max_nrec_size = enc_size(w.levels[0].max_nrec) as usize;
    let ptr = w.offset_size as usize + max_nrec_size + child_extra;
    let nchild = nrec_us + 1;
    let body = 6 + nrec_us * w.hdr.record_size as usize + nchild * ptr;
    let bytes = rdr.slice_at(addr, (body + 4) as u64)?;
    if bytes[..4] != BTIN || bytes[4] != 0 || bytes[5] != w.hdr.record_type {
        return Err(HDF5Error::InvalidHeader);
    }
    check_sum(&bytes[..body], &bytes[body..body + 4])?;
    let mut child_at = 6 + nrec_us * w.hdr.record_size as usize;
    let mut children = Vec::with_capacity(nchild);
    for _ in 0..nchild {
        let c = &bytes[child_at..child_at + ptr];
        let caddr = take_le(c, w.offset_size as usize)?;
        let cnrec = take_le(&c[w.offset_size as usize..], max_nrec_size)?;
        children.push((caddr, cnrec));
        child_at += ptr;
    }
    for (i, &(caddr, cnrec)) in children.iter().enumerate().take(nrec_us) {
        walk_node(w, caddr, cnrec, depth - 1, out, walk_depth + 1)?;
        let start = 6 + i * w.hdr.record_size as usize;
        out.push(decode_record(
            &bytes[start..start + w.hdr.record_size as usize],
            w.hdr.record_type,
            w.offset_size,
            w.spatial,
            w.ndims_key,
            w.full_chunk,
        )?);
    }
    let (caddr, cnrec) = children[nrec_us];
    walk_node(w, caddr, cnrec, depth - 1, out, walk_depth + 1)?;
    Ok(())
}

fn decode_record(
    rec: &[u8],
    record_type: u8,
    offset_size: u8,
    spatial: &[u32],
    ndims_key: usize,
    full_chunk: u32,
) -> Result<ChunkRef> {
    let sa = offset_size as usize;
    let off_bytes = spatial.len() * 8;
    if rec.len() < sa + off_bytes {
        return Err(HDF5Error::InvalidHeader);
    }
    let addr = take_le(rec, sa)?;
    let (size, mask, off_at) = if record_type == TYPE_UNFILT {
        if rec.len() != sa + off_bytes {
            return Err(HDF5Error::ChunkedNotSupported);
        }
        (full_chunk, 0u32, sa)
    } else {
        if rec.len() < sa + 4 + off_bytes {
            return Err(HDF5Error::InvalidHeader);
        }
        let size_w = rec.len() - sa - 4 - off_bytes;
        let size = take_le(&rec[sa..], size_w)?;
        let mask = take_le(&rec[sa + size_w..], 4)? as u32;
        (
            u32::try_from(size).map_err(|_| HDF5Error::InvalidHeader)?,
            mask,
            sa + size_w + 4,
        )
    };
    let mut offset = alloc::vec![0u64; ndims_key];
    for (d, dim) in spatial.iter().enumerate() {
        let scaled = take_le(&rec[off_at + d * 8..], 8)?;
        offset[d] = scaled.saturating_mul(u64::from(*dim));
    }
    Ok(ChunkRef {
        size,
        filter_mask: mask,
        offset,
        addr,
    })
}

/// Bytes needed to store every integer in `0..=limit` (HDF5 `H5VM_limit_enc_size`).
fn enc_size(limit: u64) -> u8 {
    if limit <= 0xff {
        1
    } else if limit <= 0xffff {
        2
    } else if limit <= 0xff_ffff {
        3
    } else if limit <= 0xffff_ffff {
        4
    } else if limit <= 0xff_ffff_ffff {
        5
    } else if limit <= 0xffff_ffff_ffff {
        6
    } else if limit <= 0xff_ffff_ffff_ffff {
        7
    } else {
        8
    }
}

fn take_le(bytes: &[u8], n: usize) -> Result<u64> {
    if n == 0 || n > 8 || bytes.len() < n {
        return Err(HDF5Error::InvalidHeader);
    }
    let mut tmp = [0u8; 8];
    tmp[..n].copy_from_slice(&bytes[..n]);
    Ok(u64::from_le_bytes(tmp))
}

fn check_sum(body: &[u8], stored: &[u8]) -> Result<()> {
    if stored.len() < 4 {
        return Err(HDF5Error::Truncated);
    }
    let got = u32::from_le_bytes([stored[0], stored[1], stored[2], stored[3]]);
    if lookup3(body, 0) != got {
        return Err(HDF5Error::InvalidHeader);
    }
    Ok(())
}
