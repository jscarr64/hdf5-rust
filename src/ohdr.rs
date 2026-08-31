//! Object header v1 and v2 encode/decode.

use alloc::vec::Vec;

use crate::buf::Reader;
use crate::checksum::lookup3;
use crate::error::{HDF5Error, Result};
use crate::{
    HDF5_MSG_CONTINUATION, HDF5_MSG_NIL, HDF5_OCHK_SIGNATURE, HDF5_OHDR_FLAGS_CHUNK4,
    HDF5_OHDR_SIGNATURE, HDF5_OHDR_VERSION,
};

pub struct RawMsg {
    pub ty: u8,
    pub flags: u8,
    pub body: Vec<u8>,
}

pub fn encode_ohdr_v2(messages: &[RawMsg]) -> Vec<u8> {
    let mut chunk = Vec::new();
    for m in messages {
        chunk.push(m.ty);
        let n = m.body.len() as u16;
        chunk.extend_from_slice(&n.to_le_bytes());
        chunk.push(m.flags);
        chunk.extend_from_slice(&m.body);
    }
    let mut out = Vec::new();
    out.extend_from_slice(&HDF5_OHDR_SIGNATURE);
    out.push(HDF5_OHDR_VERSION);
    out.push(HDF5_OHDR_FLAGS_CHUNK4);
    out.extend_from_slice(&(chunk.len() as u32).to_le_bytes());
    out.extend_from_slice(&chunk);
    let sum = lookup3(&out, 0);
    out.extend_from_slice(&sum.to_le_bytes());
    out
}

pub fn parse_ohdr(data: &[u8], addr: u64, offset_size: u8, length_size: u8) -> Result<Vec<RawMsg>> {
    let mut r = Reader::new(data, offset_size, length_size)?.at(addr)?;
    if r.remaining() < 4 {
        return Err(HDF5Error::Truncated);
    }
    let sig = r.bytes(4)?;
    if sig == HDF5_OHDR_SIGNATURE {
        parse_ohdr_v2(data, addr, offset_size, length_size)
    } else {
        parse_ohdr_v1(data, addr, offset_size, length_size)
    }
}

fn parse_ohdr_v2(data: &[u8], addr: u64, offset_size: u8, length_size: u8) -> Result<Vec<RawMsg>> {
    let mut r = Reader::new(data, offset_size, length_size)?.at(addr)?;
    let sig = r.bytes(4)?;
    if sig != HDF5_OHDR_SIGNATURE {
        return Err(HDF5Error::InvalidHeader);
    }
    let version = r.u8()?;
    if version != 2 {
        return Err(HDF5Error::UnsupportedVersion(version));
    }
    let flags = r.u8()?;
    if flags & 0x20 != 0 {
        r.skip(16)?;
    }
    if flags & 0x10 != 0 {
        r.skip(4)?;
    }
    let chunk0_width: u8 = match flags & 0x03 {
        0 => 1,
        1 => 2,
        2 => 4,
        3 => 8,
        _ => 4,
    };
    let chunk0 = r.sized_uint(chunk0_width)? as usize;
    let mut msgs = Vec::new();
    parse_chunk_v2(&mut r, chunk0, flags, &mut msgs)?;
    let stored = r.u32()?;
    let prefix_and_chunk = r.pos() - usize::try_from(addr).map_err(|_| HDF5Error::Truncated)? - 4;
    let start = usize::try_from(addr).map_err(|_| HDF5Error::Truncated)?;
    let hashed = &data[start..start + prefix_and_chunk];
    if lookup3(hashed, 0) != stored {
        return Err(HDF5Error::InvalidHeader);
    }
    expand_continuations(data, offset_size, length_size, flags, &mut msgs, 0)?;
    Ok(msgs)
}

fn parse_chunk_v2(
    r: &mut Reader<'_>,
    chunk_len: usize,
    ohdr_flags: u8,
    msgs: &mut Vec<RawMsg>,
) -> Result<()> {
    let end = r.pos() + chunk_len;
    while r.pos() + 4 <= end {
        let gap = end - r.pos();
        if gap < 4 {
            break;
        }
        let ty = r.u8()?;
        let size = r.u16()? as usize;
        let flags = r.u8()?;
        if ohdr_flags & 0x04 != 0 {
            let _ = r.u16()?;
        }
        if r.pos() + size > end {
            return Err(HDF5Error::InvalidHeader);
        }
        let body = r.bytes(size)?.to_vec();
        if ty != HDF5_MSG_NIL {
            msgs.push(RawMsg { ty, flags, body });
        }
    }
    if r.pos() < end {
        r.skip(end - r.pos())?;
    }
    Ok(())
}

fn expand_continuations(
    data: &[u8],
    offset_size: u8,
    length_size: u8,
    ohdr_flags: u8,
    msgs: &mut Vec<RawMsg>,
    depth: usize,
) -> Result<()> {
    if depth > crate::HDF5_MAX_WALK_DEPTH {
        return Err(HDF5Error::InvalidHeader);
    }
    let cont: Vec<(u64, u64)> = msgs
        .iter()
        .filter(|m| m.ty == HDF5_MSG_CONTINUATION)
        .filter_map(|m| parse_cont(&m.body, offset_size, length_size).ok())
        .collect();
    msgs.retain(|m| m.ty != HDF5_MSG_CONTINUATION);
    for (addr, size) in cont {
        parse_ochk(data, addr, size, offset_size, length_size, ohdr_flags, msgs)?;
        expand_continuations(data, offset_size, length_size, ohdr_flags, msgs, depth + 1)?;
    }
    Ok(())
}

fn parse_cont(body: &[u8], offset_size: u8, length_size: u8) -> Result<(u64, u64)> {
    let mut r = Reader::new(body, offset_size, length_size)?;
    let addr = r.addr()?;
    let size = r.length()?;
    Ok((addr, size))
}

fn parse_ochk(
    data: &[u8],
    addr: u64,
    size: u64,
    offset_size: u8,
    length_size: u8,
    ohdr_flags: u8,
    msgs: &mut Vec<RawMsg>,
) -> Result<()> {
    let mut r = Reader::new(data, offset_size, length_size)?.at(addr)?;
    let sig = r.bytes(4)?;
    if sig != HDF5_OCHK_SIGNATURE {
        return Err(HDF5Error::InvalidHeader);
    }
    let chunk_len = size.saturating_sub(8) as usize;
    parse_chunk_v2(&mut r, chunk_len, ohdr_flags, msgs)?;
    let stored = r.u32()?;
    let start = usize::try_from(addr).map_err(|_| HDF5Error::Truncated)?;
    let hashed_len = r.pos() - start - 4;
    if lookup3(&data[start..start + hashed_len], 0) != stored {
        return Err(HDF5Error::InvalidHeader);
    }
    Ok(())
}

fn parse_ohdr_v1(data: &[u8], addr: u64, offset_size: u8, length_size: u8) -> Result<Vec<RawMsg>> {
    let mut r = Reader::new(data, offset_size, length_size)?.at(addr)?;
    let version = r.u8()?;
    if version != 1 {
        return Err(HDF5Error::UnsupportedVersion(version));
    }
    r.u8()?;
    let nmsgs = r.u16()? as usize;
    let _refcnt = r.u32()?;
    let header_size = r.u32()? as usize;
    r.skip(4)?;
    let mut msgs = Vec::new();
    let end = r.pos() + header_size;
    for _ in 0..nmsgs {
        if r.pos() + 8 > end {
            break;
        }
        let ty = r.u16()? as u8;
        let size = r.u16()? as usize;
        let flags = r.u8()?;
        r.skip(3)?;
        let body = r.bytes(size)?.to_vec();
        if ty != HDF5_MSG_NIL {
            msgs.push(RawMsg { ty, flags, body });
        }
    }
    let cont: Vec<(u64, u64)> = msgs
        .iter()
        .filter(|m| m.ty == HDF5_MSG_CONTINUATION)
        .filter_map(|m| parse_cont(&m.body, offset_size, length_size).ok())
        .collect();
    msgs.retain(|m| m.ty != HDF5_MSG_CONTINUATION);
    for (caddr, _csz) in cont {
        let extra = parse_ohdr_v1(data, caddr, offset_size, length_size)?;
        msgs.extend(extra);
    }
    Ok(msgs)
}
