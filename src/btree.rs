//! Version-1 group B-trees, symbol-table nodes, local heaps, global heaps.

use alloc::string::String;
use alloc::vec::Vec;

use crate::buf::Reader;
use crate::error::{HDF5Error, Result};
use crate::messages::cstr_from_heap;
use crate::{
    HDF5_BTREE_GROUP, HDF5_GCOL_SIGNATURE, HDF5_HEAP_SIGNATURE, HDF5_MAX_WALK_DEPTH,
    HDF5_SNOD_SIGNATURE, HDF5_TREE_SIGNATURE,
};

pub fn read_local_heap<'a>(
    data: &'a [u8],
    addr: u64,
    offset_size: u8,
    length_size: u8,
) -> Result<&'a [u8]> {
    let mut r = Reader::new(data, offset_size, length_size)?.at(addr)?;
    let sig = r.bytes(4)?;
    if sig != HDF5_HEAP_SIGNATURE {
        return Err(HDF5Error::InvalidHeader);
    }
    let version = r.u8()?;
    if version != 0 {
        return Err(HDF5Error::UnsupportedVersion(version));
    }
    r.skip(3)?;
    let seg_size = r.length()?;
    let _free = r.length()?;
    let seg_addr = r.addr()?;
    Reader::new(data, offset_size, length_size)?.slice_at(seg_addr, seg_size)
}

pub struct GroupChild {
    pub name: String,
    pub ohdr: u64,
}

pub fn walk_group_btree(
    data: &[u8],
    btree_addr: u64,
    heap: &[u8],
    offset_size: u8,
    length_size: u8,
    depth: usize,
) -> Result<Vec<GroupChild>> {
    if depth > HDF5_MAX_WALK_DEPTH {
        return Err(HDF5Error::InvalidHeader);
    }
    let mut r = Reader::new(data, offset_size, length_size)?.at(btree_addr)?;
    let sig = r.bytes(4)?;
    if sig != HDF5_TREE_SIGNATURE {
        return Err(HDF5Error::InvalidHeader);
    }
    let node_type = r.u8()?;
    if node_type != HDF5_BTREE_GROUP {
        return Err(HDF5Error::InvalidHeader);
    }
    let level = r.u8()?;
    let used = r.u16()? as usize;
    let _left = r.addr()?;
    let _right = r.addr()?;
    let mut children = Vec::new();
    for _ in 0..used {
        let _key = r.length()?;
        let child = r.addr()?;
        if level == 0 {
            children.extend(read_snod(data, child, heap, offset_size, length_size)?);
        } else {
            children.extend(walk_group_btree(
                data,
                child,
                heap,
                offset_size,
                length_size,
                depth + 1,
            )?);
        }
    }
    Ok(children)
}

fn read_snod(
    data: &[u8],
    addr: u64,
    heap: &[u8],
    offset_size: u8,
    length_size: u8,
) -> Result<Vec<GroupChild>> {
    let mut r = Reader::new(data, offset_size, length_size)?.at(addr)?;
    let sig = r.bytes(4)?;
    if sig != HDF5_SNOD_SIGNATURE {
        return Err(HDF5Error::InvalidHeader);
    }
    let version = r.u8()?;
    if version != 1 {
        return Err(HDF5Error::UnsupportedVersion(version));
    }
    r.u8()?;
    let nsym = r.u16()? as usize;
    let mut out = Vec::new();
    for _ in 0..nsym {
        let name_off = r.addr()?;
        let ohdr = r.addr()?;
        let cache = r.u32()?;
        r.u32()?;
        r.skip(16)?;
        let _ = cache;
        let name = cstr_from_heap(heap, name_off)?;
        if !name.is_empty() {
            out.push(GroupChild { name, ohdr });
        }
    }
    Ok(out)
}

pub fn read_gheap_object(
    data: &[u8],
    collection: u64,
    index: u32,
    offset_size: u8,
    length_size: u8,
) -> Result<Vec<u8>> {
    let mut r = Reader::new(data, offset_size, length_size)?.at(collection)?;
    let sig = r.bytes(4)?;
    if sig != HDF5_GCOL_SIGNATURE {
        return Err(HDF5Error::InvalidHeader);
    }
    let version = r.u8()?;
    if version != 1 {
        return Err(HDF5Error::UnsupportedVersion(version));
    }
    r.skip(3)?;
    let col_size = r.length()? as usize;
    let start = usize::try_from(collection).map_err(|_| HDF5Error::Truncated)?;
    let end = start.saturating_add(col_size).min(data.len());
    while r.pos() + 8 < end {
        let obj_index = r.u16()? as u32;
        let _refcnt = r.u16()?;
        r.u32()?;
        let obj_size = r.length()? as usize;
        let payload = r.bytes(obj_size)?.to_vec();
        let pad = (8 - (obj_size % 8)) % 8;
        if pad > 0 && r.remaining() >= pad {
            r.skip(pad)?;
        }
        if obj_index == index {
            return Ok(payload);
        }
        if obj_index == 0 {
            break;
        }
    }
    Err(HDF5Error::InvalidHeader)
}
