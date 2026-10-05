//! Extensible-array chunk index (`EAHD` / `EAIB` / `EADB` / `EASB`).
//!
//! Used for chunked datasets with exactly one unlimited dimension.
//! Data blocks larger than one page (HDF5 paging) are refused: guessing a
//! page layout would invent chunk addresses.

use alloc::collections::btree_map::Entry;
use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use crate::btree::ChunkRef;
use crate::buf::Reader;
use crate::checksum::lookup3;
use crate::chunk_index::{linear_chunk_offset, nchunks_per_dim, spatial_chunk_dims};
use crate::error::{HDF5Error, Result};

const EAHD: [u8; 4] = *b"EAHD";
const EAIB: [u8; 4] = *b"EAIB";
const EADB: [u8; 4] = *b"EADB";
const EASB: [u8; 4] = *b"EASB";

struct Header {
    elmt_size: usize,
    max_nelmts_bits: u8,
    idx_blk_elmts: u64,
    data_blk_min_elmts: u64,
    sup_blk_min_data_ptrs: u64,
    page_nelmts: u64,
    max_idx: u64,
    idx_blk_addr: u64,
}

struct Geometry {
    idx_blk_elmts: u64,
    data_blk_min_elmts: u64,
    page_nelmts: u64,
    iblock_nsblks: usize,
    ndblk_addrs: usize,
    nsblk_addrs: usize,
    sblk: Vec<Sblk>,
}

struct Sblk {
    ndblks: u64,
    dblk_nelmts: u64,
    start_idx: u64,
    start_dblk: u64,
}

enum Loc {
    Index {
        elem: usize,
    },
    Direct {
        dblk: usize,
        offset: u64,
        nelmts: u64,
    },
    Super {
        sblk_off: usize,
        local_dblk: usize,
        offset: u64,
        nelmts: u64,
        ndblks: usize,
    },
}

/// Walk an extensible-array chunk index into chunk references.
pub fn walk_extensible_array(
    data: &[u8],
    header_addr: u64,
    offset_size: u8,
    length_size: u8,
    shape: &[u64],
    chunk_dims: &[u32],
    full_chunk: u32,
) -> Result<Vec<ChunkRef>> {
    let hdr = read_header(data, header_addr, offset_size, length_size)?;
    if hdr.max_idx == 0 {
        return Ok(Vec::new());
    }
    let geo = geometry(&hdr)?;
    let iblock = read_index_block(data, &hdr, &geo, offset_size, length_size)?;
    let spatial = spatial_chunk_dims(shape, chunk_dims)?;
    let nper = nchunks_per_dim(shape, &spatial)?;
    let ndims_key = chunk_dims.len();
    let mut dblks: BTreeMap<u64, Vec<u8>> = BTreeMap::new();
    let mut sblks: BTreeMap<u64, Vec<u64>> = BTreeMap::new();
    let mut chunks = Vec::new();
    for idx in 0..hdr.max_idx {
        let loc = locate(&geo, idx)?;
        let owned = match loc {
            Loc::Index { elem } => {
                let start = elem * hdr.elmt_size;
                iblock
                    .elements
                    .get(start..start + hdr.elmt_size)
                    .ok_or(HDF5Error::InvalidHeader)?
                    .to_vec()
            }
            Loc::Direct {
                dblk,
                offset,
                nelmts,
            } => {
                let addr = *iblock
                    .dblk_addrs
                    .get(dblk)
                    .ok_or(HDF5Error::InvalidHeader)?;
                element_bytes(
                    &FileBytes {
                        data,
                        offset_size,
                        length_size,
                    },
                    &hdr,
                    &mut dblks,
                    addr,
                    offset,
                    nelmts,
                )?
            }
            Loc::Super {
                sblk_off,
                local_dblk,
                offset,
                nelmts,
                ndblks,
            } => {
                let saddr = *iblock
                    .sblk_addrs
                    .get(sblk_off)
                    .ok_or(HDF5Error::InvalidHeader)?;
                let daddrs = super_dblk_addrs(
                    data,
                    &hdr,
                    &mut sblks,
                    saddr,
                    ndblks,
                    offset_size,
                    length_size,
                )?;
                let addr = *daddrs.get(local_dblk).ok_or(HDF5Error::InvalidHeader)?;
                element_bytes(
                    &FileBytes {
                        data,
                        offset_size,
                        length_size,
                    },
                    &hdr,
                    &mut dblks,
                    addr,
                    offset,
                    nelmts,
                )?
            }
        };
        let (addr, size, mask) = decode_element(&owned, offset_size, full_chunk)?;
        let rdr = Reader::new(data, offset_size, length_size)?;
        if rdr.is_undef(addr) || size == 0 {
            continue;
        }
        let offset = linear_chunk_offset(idx, &nper, &spatial, ndims_key)?;
        chunks.push(ChunkRef {
            size,
            filter_mask: mask,
            offset,
            addr,
        });
    }
    Ok(chunks)
}

fn read_header(data: &[u8], addr: u64, offset_size: u8, length_size: u8) -> Result<Header> {
    let ss = length_size as usize;
    let sa = offset_size as usize;
    let body = 4 + 8 + 6 * ss + sa;
    let bytes = Reader::new(data, offset_size, length_size)?.slice_at(addr, (body + 4) as u64)?;
    if bytes[..4] != EAHD {
        return Err(HDF5Error::InvalidHeader);
    }
    if bytes[4] != 0 {
        return Err(HDF5Error::UnsupportedVersion(bytes[4]));
    }
    check_sum(&bytes[..body], &bytes[body..body + 4])?;
    let elmt_size = bytes[6] as usize;
    if elmt_size == 0 {
        return Err(HDF5Error::InvalidHeader);
    }
    let max_nelmts_bits = bytes[7];
    let idx_blk_elmts = u64::from(bytes[8]);
    let data_blk_min_elmts = u64::from(bytes[9]);
    let sup_blk_min_data_ptrs = u64::from(bytes[10]);
    let page_bits = bytes[11];
    if page_bits >= 63 {
        return Err(HDF5Error::InvalidHeader);
    }
    let mut p = 12;
    let _num_sblks = take_le(&bytes[p..], ss)?;
    p += ss;
    let _size_sblks = take_le(&bytes[p..], ss)?;
    p += ss;
    let _num_dblks = take_le(&bytes[p..], ss)?;
    p += ss;
    let _size_dblks = take_le(&bytes[p..], ss)?;
    p += ss;
    let max_idx = take_le(&bytes[p..], ss)?;
    p += ss;
    let _nelmts = take_le(&bytes[p..], ss)?;
    p += ss;
    let idx_blk_addr = take_le(&bytes[p..], sa)?;
    Ok(Header {
        elmt_size,
        max_nelmts_bits,
        idx_blk_elmts,
        data_blk_min_elmts,
        sup_blk_min_data_ptrs,
        page_nelmts: 1u64 << page_bits,
        max_idx,
        idx_blk_addr,
    })
}

fn geometry(hdr: &Header) -> Result<Geometry> {
    let min = hdr.data_blk_min_elmts;
    let sup = hdr.sup_blk_min_data_ptrs;
    if min == 0 || !min.is_power_of_two() || sup == 0 || !sup.is_power_of_two() {
        return Err(HDF5Error::InvalidHeader);
    }
    let min_bits = min.trailing_zeros();
    if u32::from(hdr.max_nelmts_bits) < min_bits || hdr.max_nelmts_bits > 64 {
        return Err(HDF5Error::InvalidHeader);
    }
    let nsblks = 1 + (hdr.max_nelmts_bits as usize - min_bits as usize);
    let iblock_nsblks = 2 * sup.trailing_zeros() as usize;
    if iblock_nsblks > nsblks {
        return Err(HDF5Error::InvalidHeader);
    }
    let mut sblk = Vec::with_capacity(nsblks);
    let mut start_idx = 0u64;
    let mut start_dblk = 0u64;
    for u in 0..nsblks {
        let ndblks = 1u64 << (u / 2);
        let dblk_nelmts = (1u64 << (((u as u32) + 1) / 2))
            .checked_mul(min)
            .ok_or(HDF5Error::InvalidHeader)?;
        sblk.push(Sblk {
            ndblks,
            dblk_nelmts,
            start_idx,
            start_dblk,
        });
        start_idx = start_idx
            .checked_add(
                ndblks
                    .checked_mul(dblk_nelmts)
                    .ok_or(HDF5Error::InvalidHeader)?,
            )
            .ok_or(HDF5Error::InvalidHeader)?;
        start_dblk = start_dblk
            .checked_add(ndblks)
            .ok_or(HDF5Error::InvalidHeader)?;
    }
    Ok(Geometry {
        idx_blk_elmts: hdr.idx_blk_elmts,
        data_blk_min_elmts: min,
        page_nelmts: hdr.page_nelmts,
        iblock_nsblks,
        ndblk_addrs: 2 * (sup as usize - 1),
        nsblk_addrs: nsblks - iblock_nsblks,
        sblk,
    })
}

struct IndexBlock {
    elements: Vec<u8>,
    dblk_addrs: Vec<u64>,
    sblk_addrs: Vec<u64>,
}

fn read_index_block(
    data: &[u8],
    hdr: &Header,
    geo: &Geometry,
    offset_size: u8,
    length_size: u8,
) -> Result<IndexBlock> {
    let rdr = Reader::new(data, offset_size, length_size)?;
    if rdr.is_undef(hdr.idx_blk_addr) {
        return Err(HDF5Error::ChunkedNotSupported);
    }
    let sa = offset_size as usize;
    let n_el = usize::try_from(hdr.idx_blk_elmts).map_err(|_| HDF5Error::InvalidHeader)?;
    let body = 6 + sa + n_el * hdr.elmt_size + geo.ndblk_addrs * sa + geo.nsblk_addrs * sa;
    let bytes = Reader::new(data, offset_size, length_size)?
        .slice_at(hdr.idx_blk_addr, (body + 4) as u64)?;
    if bytes[..4] != EAIB {
        return Err(HDF5Error::InvalidHeader);
    }
    if bytes[4] != 0 {
        return Err(HDF5Error::UnsupportedVersion(bytes[4]));
    }
    check_sum(&bytes[..body], &bytes[body..body + 4])?;
    let mut p = 6 + sa;
    let elements = bytes[p..p + n_el * hdr.elmt_size].to_vec();
    p += n_el * hdr.elmt_size;
    let mut dblk_addrs = Vec::with_capacity(geo.ndblk_addrs);
    for _ in 0..geo.ndblk_addrs {
        dblk_addrs.push(take_le(&bytes[p..], sa)?);
        p += sa;
    }
    let mut sblk_addrs = Vec::with_capacity(geo.nsblk_addrs);
    for _ in 0..geo.nsblk_addrs {
        sblk_addrs.push(take_le(&bytes[p..], sa)?);
        p += sa;
    }
    Ok(IndexBlock {
        elements,
        dblk_addrs,
        sblk_addrs,
    })
}

fn locate(geo: &Geometry, idx: u64) -> Result<Loc> {
    if idx < geo.idx_blk_elmts {
        return Ok(Loc::Index { elem: idx as usize });
    }
    let e = idx - geo.idx_blk_elmts;
    let sblk_idx = floor_log2(e / geo.data_blk_min_elmts + 1) as usize;
    let s = geo.sblk.get(sblk_idx).ok_or(HDF5Error::InvalidHeader)?;
    if s.dblk_nelmts > geo.page_nelmts {
        return Err(HDF5Error::ChunkedNotSupported);
    }
    let elmt = e - s.start_idx;
    let local = elmt / s.dblk_nelmts;
    let offset = elmt % s.dblk_nelmts;
    if sblk_idx < geo.iblock_nsblks {
        let dblk = s
            .start_dblk
            .checked_add(local)
            .ok_or(HDF5Error::InvalidHeader)?;
        Ok(Loc::Direct {
            dblk: usize::try_from(dblk).map_err(|_| HDF5Error::InvalidHeader)?,
            offset,
            nelmts: s.dblk_nelmts,
        })
    } else {
        Ok(Loc::Super {
            sblk_off: sblk_idx - geo.iblock_nsblks,
            local_dblk: usize::try_from(local).map_err(|_| HDF5Error::InvalidHeader)?,
            offset,
            nelmts: s.dblk_nelmts,
            ndblks: usize::try_from(s.ndblks).map_err(|_| HDF5Error::InvalidHeader)?,
        })
    }
}

struct FileBytes<'a> {
    data: &'a [u8],
    offset_size: u8,
    length_size: u8,
}

fn element_bytes(
    src: &FileBytes<'_>,
    hdr: &Header,
    cache: &mut BTreeMap<u64, Vec<u8>>,
    addr: u64,
    offset: u64,
    nelmts: u64,
) -> Result<Vec<u8>> {
    let block = match cache.entry(addr) {
        Entry::Occupied(hit) => hit.into_mut(),
        Entry::Vacant(hole) => {
            let bytes = load_dblk(
                src.data,
                hdr,
                addr,
                nelmts,
                src.offset_size,
                src.length_size,
            )?;
            hole.insert(bytes)
        }
    };
    if block.is_empty() {
        return Ok(alloc::vec![0xff; hdr.elmt_size]);
    }
    let off = usize::try_from(offset).map_err(|_| HDF5Error::InvalidHeader)?;
    let start = off
        .checked_mul(hdr.elmt_size)
        .ok_or(HDF5Error::InvalidHeader)?;
    block
        .get(start..start + hdr.elmt_size)
        .map(|s| s.to_vec())
        .ok_or(HDF5Error::InvalidHeader)
}

fn super_dblk_addrs(
    data: &[u8],
    hdr: &Header,
    cache: &mut BTreeMap<u64, Vec<u64>>,
    addr: u64,
    ndblks: usize,
    offset_size: u8,
    length_size: u8,
) -> Result<Vec<u64>> {
    if let Some(v) = cache.get(&addr) {
        return Ok(v.clone());
    }
    let rdr = Reader::new(data, offset_size, length_size)?;
    if rdr.is_undef(addr) {
        return Err(HDF5Error::InvalidHeader);
    }
    let sa = offset_size as usize;
    let bo = block_offset_size(hdr.max_nelmts_bits);
    let body = 6 + sa + bo + ndblks * sa;
    let bytes = rdr.slice_at(addr, (body + 4) as u64)?;
    if bytes[..4] != EASB {
        return Err(HDF5Error::InvalidHeader);
    }
    if bytes[4] != 0 {
        return Err(HDF5Error::UnsupportedVersion(bytes[4]));
    }
    check_sum(&bytes[..body], &bytes[body..body + 4])?;
    let mut p = 6 + sa + bo;
    let mut addrs = Vec::with_capacity(ndblks);
    for _ in 0..ndblks {
        addrs.push(take_le(&bytes[p..], sa)?);
        p += sa;
    }
    cache.insert(addr, addrs.clone());
    Ok(addrs)
}

fn load_dblk(
    data: &[u8],
    hdr: &Header,
    addr: u64,
    nelmts: u64,
    offset_size: u8,
    length_size: u8,
) -> Result<Vec<u8>> {
    let rdr = Reader::new(data, offset_size, length_size)?;
    if rdr.is_undef(addr) {
        return Ok(Vec::new());
    }
    let sa = offset_size as usize;
    let bo = block_offset_size(hdr.max_nelmts_bits);
    let n = usize::try_from(nelmts).map_err(|_| HDF5Error::InvalidHeader)?;
    let body = 6 + sa + bo + n * hdr.elmt_size;
    let bytes = rdr.slice_at(addr, (body + 4) as u64)?;
    if bytes[..4] != EADB {
        return Err(HDF5Error::InvalidHeader);
    }
    if bytes[4] != 0 {
        return Err(HDF5Error::UnsupportedVersion(bytes[4]));
    }
    check_sum(&bytes[..body], &bytes[body..body + 4])?;
    let start = 6 + sa + bo;
    Ok(bytes[start..start + n * hdr.elmt_size].to_vec())
}

fn decode_element(raw: &[u8], offset_size: u8, full_chunk: u32) -> Result<(u64, u32, u32)> {
    let sa = offset_size as usize;
    if raw.len() < sa {
        return Err(HDF5Error::InvalidHeader);
    }
    let addr = take_le(raw, sa)?;
    if raw.len() == sa {
        return Ok((addr, full_chunk, 0));
    }
    if raw.len() < sa + 4 {
        return Err(HDF5Error::ChunkedNotSupported);
    }
    let size_w = raw.len() - sa - 4;
    let size = take_le(&raw[sa..], size_w)?;
    let mask = take_le(&raw[sa + size_w..], 4)? as u32;
    let size_u = u32::try_from(size).map_err(|_| HDF5Error::InvalidHeader)?;
    Ok((addr, size_u, mask))
}

fn block_offset_size(max_nelmts_bits: u8) -> usize {
    core::cmp::max(1, (usize::from(max_nelmts_bits) + 7) / 8)
}

fn floor_log2(n: u64) -> u32 {
    if n == 0 {
        0
    } else {
        63 - n.leading_zeros()
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
