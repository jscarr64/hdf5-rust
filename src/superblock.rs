//! Superblock v0/v1/v2 locate and parse.

use crate::buf::Reader;
use crate::checksum::lookup3;
use crate::error::{HDF5Error, Result};
use crate::{
    HDF5_SIGNATURE, HDF5_SUPERBLOCK_SEARCH_START, HDF5_SUPERBLOCK_V2, HDF5_SUPERBLOCK_V2_SIZE,
};

pub struct Superblock {
    pub offset_size: u8,
    pub length_size: u8,
    pub root_ohdr: u64,
    pub root_btree: Option<u64>,
    pub root_heap: Option<u64>,
}

pub fn find_signature(data: &[u8]) -> Result<u64> {
    if data.len() >= 8 && data[..8] == HDF5_SIGNATURE {
        return Ok(0);
    }
    let mut off = HDF5_SUPERBLOCK_SEARCH_START;
    while let Ok(start) = usize::try_from(off) {
        if start + 8 > data.len() {
            break;
        }
        if data[start..start + 8] == HDF5_SIGNATURE {
            return Ok(off);
        }
        match off.checked_mul(2) {
            Some(n) => off = n,
            None => break,
        }
    }
    Err(HDF5Error::InvalidSignature)
}

pub fn parse(data: &[u8]) -> Result<Superblock> {
    let off = find_signature(data)?;
    let mut r = Reader::new(data, 8, 8)?.at(off)?;
    let sig = r.bytes(8)?;
    if sig != HDF5_SIGNATURE {
        return Err(HDF5Error::InvalidSignature);
    }
    let version = r.u8()?;
    match version {
        0 | 1 => parse_v01(data, off, version),
        2 | 3 => parse_v2(data, off, version),
        v => Err(HDF5Error::UnsupportedVersion(v)),
    }
}

fn parse_v2(data: &[u8], off: u64, version: u8) -> Result<Superblock> {
    let mut peek = Reader::new(data, 8, 8)?.at(off)?;
    let _ = peek.bytes(8)?;
    let _ver = peek.u8()?;
    let offset_size = peek.u8()?;
    let length_size = peek.u8()?;
    let mut r = Reader::new(data, offset_size, length_size)?.at(off)?;
    r.bytes(8)?;
    r.u8()?;
    r.u8()?;
    r.u8()?;
    let _flags = r.u8()?;
    let base = r.addr()?;
    let _ext = r.addr()?;
    let eof = r.addr()?;
    let root_ohdr = r.addr()?;
    let stored = r.u32()?;
    let start = usize::try_from(off).map_err(|_| HDF5Error::Truncated)?;
    let hashed_end = r.pos() - 4;
    if lookup3(&data[start..hashed_end], 0) != stored {
        return Err(HDF5Error::InvalidHeader);
    }
    let _ = version;
    let _ = base;
    let _ = eof;
    Ok(Superblock {
        offset_size,
        length_size,
        root_ohdr,
        root_btree: None,
        root_heap: None,
    })
}

fn parse_v01(data: &[u8], off: u64, version: u8) -> Result<Superblock> {
    let mut peek = Reader::new(data, 8, 8)?.at(off)?;
    peek.bytes(8)?;
    peek.u8()?;
    peek.u8()?;
    peek.u8()?;
    peek.u8()?;
    peek.u8()?;
    let offset_size = peek.u8()?;
    let length_size = peek.u8()?;
    let mut r = Reader::new(data, offset_size, length_size)?.at(off)?;
    r.bytes(8)?;
    r.u8()?;
    r.u8()?;
    r.u8()?;
    r.u8()?;
    r.u8()?;
    r.u8()?;
    r.u8()?;
    r.u8()?;
    let _leaf_k = r.u16()?;
    let _int_k = r.u16()?;
    let _flags = r.u32()?;
    if version == 1 {
        let _ = r.u16()?;
        let _ = r.u16()?;
    }
    let base = r.addr()?;
    let _free = r.addr()?;
    let eof = r.addr()?;
    let _driver = r.addr()?;
    let _name_off = r.addr()?;
    let root_ohdr = r.addr()?;
    let cache_type = r.u32()?;
    r.u32()?;
    let mut root_btree = None;
    let mut root_heap = None;
    if cache_type == 1 {
        root_btree = Some(r.addr()?);
        root_heap = Some(r.addr()?);
    } else {
        r.skip(16)?;
    }
    let _ = base;
    let _ = eof;
    let _ = off;
    Ok(Superblock {
        offset_size,
        length_size,
        root_ohdr,
        root_btree,
        root_heap,
    })
}

pub fn encode_v2(root_ohdr: u64, eof: u64) -> [u8; HDF5_SUPERBLOCK_V2_SIZE] {
    let mut buf = [0u8; HDF5_SUPERBLOCK_V2_SIZE];
    buf[0..8].copy_from_slice(&HDF5_SIGNATURE);
    buf[8] = HDF5_SUPERBLOCK_V2;
    buf[9] = crate::HDF5_OFFSET_SIZE;
    buf[10] = crate::HDF5_LENGTH_SIZE;
    buf[11] = 0;
    buf[12..20].copy_from_slice(&0u64.to_le_bytes());
    buf[20..28].copy_from_slice(&crate::HDF5_UNDEF_ADDR_8.to_le_bytes());
    buf[28..36].copy_from_slice(&eof.to_le_bytes());
    buf[36..44].copy_from_slice(&root_ohdr.to_le_bytes());
    let sum = lookup3(&buf[..44], 0);
    buf[44..48].copy_from_slice(&sum.to_le_bytes());
    buf
}
