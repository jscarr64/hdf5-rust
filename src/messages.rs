//! Object-header message bodies this crate writes and reads.

use alloc::string::String;
use alloc::vec::Vec;

use crate::buf::{push_u16, push_u32, push_u64, Reader};
use crate::error::{HDF5Error, Result};
use crate::{
    HDF5_ATTR_VERSION, HDF5_CLASS_FLOAT, HDF5_CLASS_OPAQUE, HDF5_CLASS_STRING, HDF5_CLASS_VLEN,
    HDF5_DATASPACE_VERSION, HDF5_DTYPE_VERSION, HDF5_FILL_VERSION, HDF5_LAYOUT_CHUNKED,
    HDF5_LAYOUT_COMPACT, HDF5_LAYOUT_CONTIGUOUS, HDF5_LAYOUT_VERSION, HDF5_LINK_VERSION,
    HDF5_MAX_DIMS, HDF5_MAX_NAME_LEN, HDF5_OPAQUE_TAG, HDF5_SPACE_SCALAR, HDF5_SPACE_SIMPLE,
    HDF5_UNDEF_ADDR_8, HDF5_UNLIMITED,
};

/// In-memory datatype after parsing a datatype message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ParsedDType {
    Float64Le,
    Float32Le,
    Opaque { size: usize },
    FixedString { size: usize },
    VlenString,
    Other { size: usize },
}

impl ParsedDType {
    pub fn element_size(&self) -> usize {
        match *self {
            Self::Float64Le => 8,
            Self::Float32Le => 4,
            Self::Opaque { size } => size,
            Self::FixedString { size } => size,
            Self::VlenString => 0,
            Self::Other { size } => size,
        }
    }
}

pub fn encode_ieee_f64le() -> Vec<u8> {
    let mut v = Vec::with_capacity(20);
    v.push((HDF5_DTYPE_VERSION << 4) | HDF5_CLASS_FLOAT);
    v.push(0x20);
    v.push(63);
    v.push(0);
    push_u32(&mut v, 8);
    push_u16(&mut v, 0);
    push_u16(&mut v, 64);
    v.push(52);
    v.push(11);
    v.push(0);
    v.push(52);
    push_u32(&mut v, 1023);
    v
}

pub fn encode_ieee_f32le() -> Vec<u8> {
    let mut v = Vec::with_capacity(20);
    v.push((HDF5_DTYPE_VERSION << 4) | HDF5_CLASS_FLOAT);
    v.push(0x20);
    v.push(31);
    v.push(0);
    push_u32(&mut v, 4);
    push_u16(&mut v, 0);
    push_u16(&mut v, 32);
    v.push(23);
    v.push(8);
    v.push(0);
    v.push(23);
    push_u32(&mut v, 127);
    v
}

pub fn encode_opaque(elem_size: u32) -> Vec<u8> {
    let mut v = Vec::with_capacity(16);
    v.push((HDF5_DTYPE_VERSION << 4) | HDF5_CLASS_OPAQUE);
    v.push(HDF5_OPAQUE_TAG.len() as u8);
    v.push(0);
    v.push(0);
    push_u32(&mut v, elem_size);
    v.extend_from_slice(&HDF5_OPAQUE_TAG);
    v
}

pub fn encode_fixed_string(nbytes: u32) -> Vec<u8> {
    let mut v = Vec::with_capacity(8);
    v.push((HDF5_DTYPE_VERSION << 4) | HDF5_CLASS_STRING);
    v.push(0);
    v.push(0);
    v.push(0);
    push_u32(&mut v, nbytes);
    v
}

pub fn encode_dataspace(dims: &[u64], unlimited_first: bool) -> Result<Vec<u8>> {
    if dims.len() > HDF5_MAX_DIMS {
        return Err(HDF5Error::RankNotSupported);
    }
    if dims.is_empty() {
        return Ok(alloc::vec![HDF5_DATASPACE_VERSION, 0, 0, HDF5_SPACE_SCALAR]);
    }
    let mut v = Vec::new();
    v.push(HDF5_DATASPACE_VERSION);
    v.push(dims.len() as u8);
    v.push(1);
    v.push(HDF5_SPACE_SIMPLE);
    for d in dims {
        push_u64(&mut v, *d);
    }
    for (i, d) in dims.iter().enumerate() {
        if unlimited_first && i == 0 {
            push_u64(&mut v, HDF5_UNLIMITED);
        } else {
            push_u64(&mut v, *d);
        }
    }
    Ok(v)
}

pub fn encode_layout_contiguous(addr: u64, size: u64) -> Vec<u8> {
    let mut v = Vec::with_capacity(18);
    v.push(HDF5_LAYOUT_VERSION);
    v.push(HDF5_LAYOUT_CONTIGUOUS);
    push_u64(&mut v, addr);
    push_u64(&mut v, size);
    v
}

pub fn encode_fill() -> Vec<u8> {
    alloc::vec![HDF5_FILL_VERSION, 0x02]
}

pub fn encode_link_info() -> Vec<u8> {
    let mut v = Vec::with_capacity(18);
    v.push(0);
    v.push(0);
    push_u64(&mut v, HDF5_UNDEF_ADDR_8);
    push_u64(&mut v, HDF5_UNDEF_ADDR_8);
    v
}

pub fn encode_group_info() -> Vec<u8> {
    alloc::vec![0, 0]
}

pub fn encode_hard_link(name: &str, ohdr: u64) -> Result<Vec<u8>> {
    if name.len() > HDF5_MAX_NAME_LEN {
        return Err(HDF5Error::NameTooLong);
    }
    let n = name.len();
    let (flag_len, len_bytes): (u8, Vec<u8>) = if n < 256 {
        (0, alloc::vec![n as u8])
    } else if n < 65536 {
        (1, (n as u16).to_le_bytes().to_vec())
    } else {
        (2, (n as u32).to_le_bytes().to_vec())
    };
    let mut v = Vec::new();
    v.push(HDF5_LINK_VERSION);
    v.push(flag_len);
    v.extend_from_slice(&len_bytes);
    v.extend_from_slice(name.as_bytes());
    push_u64(&mut v, ohdr);
    Ok(v)
}

pub fn encode_string_attr(name: &str, value: &str) -> Result<Vec<u8>> {
    if name.len() > HDF5_MAX_NAME_LEN || value.len() > HDF5_MAX_NAME_LEN {
        return Err(HDF5Error::NameTooLong);
    }
    let name_z = name.len() + 1;
    let val_bytes = value.len() + 1;
    let dtype = encode_fixed_string(val_bytes as u32);
    let space = encode_dataspace(&[], false)?;
    let mut v = Vec::new();
    v.push(HDF5_ATTR_VERSION);
    v.push(0);
    push_u16(&mut v, name_z as u16);
    push_u16(&mut v, dtype.len() as u16);
    push_u16(&mut v, space.len() as u16);
    v.extend_from_slice(name.as_bytes());
    v.push(0);
    v.extend_from_slice(&dtype);
    v.extend_from_slice(&space);
    v.extend_from_slice(value.as_bytes());
    v.push(0);
    Ok(v)
}

pub fn parse_datatype(body: &[u8]) -> Result<ParsedDType> {
    if body.len() < 8 {
        return Err(HDF5Error::InvalidHeader);
    }
    let class_ver = body[0];
    let class = class_ver & 0x0F;
    let bit0 = body[1];
    let bit1 = body[2];
    let size = u32::from_le_bytes([body[4], body[5], body[6], body[7]]) as usize;
    match class {
        HDF5_CLASS_FLOAT => {
            let le = (bit0 & 0x41) == 0;
            if !le {
                return Ok(ParsedDType::Other { size });
            }
            match size {
                8 if bit1 == 63 => Ok(ParsedDType::Float64Le),
                4 if bit1 == 31 => Ok(ParsedDType::Float32Le),
                8 => Ok(ParsedDType::Float64Le),
                4 => Ok(ParsedDType::Float32Le),
                _ => Ok(ParsedDType::Other { size }),
            }
        }
        HDF5_CLASS_OPAQUE => Ok(ParsedDType::Opaque { size }),
        HDF5_CLASS_STRING => Ok(ParsedDType::FixedString { size }),
        HDF5_CLASS_VLEN => {
            if (bit0 & 0x0F) == 1 {
                Ok(ParsedDType::VlenString)
            } else {
                Ok(ParsedDType::Other { size })
            }
        }
        _ => Ok(ParsedDType::Other { size }),
    }
}

pub struct ParsedSpace {
    pub dims: Vec<u64>,
}

pub fn parse_dataspace(body: &[u8], length_size: u8) -> Result<ParsedSpace> {
    if body.len() < 4 {
        return Err(HDF5Error::InvalidHeader);
    }
    let version = body[0];
    let ndims = body[1] as usize;
    if ndims > HDF5_MAX_DIMS {
        return Err(HDF5Error::RankNotSupported);
    }
    let flags = body[2];
    let mut r = Reader::new(body, 8, length_size)?;
    match version {
        1 => {
            r.skip(8)?;
        }
        2 => {
            r.skip(4)?;
        }
        _ => return Err(HDF5Error::UnsupportedVersion(version)),
    }
    let mut dims = Vec::new();
    for _ in 0..ndims {
        dims.push(r.length()?);
    }
    if flags & 1 != 0 {
        for _ in 0..ndims {
            let _ = r.length()?;
        }
    }
    if version == 1 && flags & 2 != 0 {
        for _ in 0..ndims {
            let _ = r.length()?;
        }
    }
    Ok(ParsedSpace { dims })
}

pub enum ParsedLayout {
    Contiguous { addr: u64, size: u64 },
    Compact { data: Vec<u8> },
    Chunked,
}

pub fn parse_layout(body: &[u8], offset_size: u8, length_size: u8) -> Result<ParsedLayout> {
    if body.is_empty() {
        return Err(HDF5Error::InvalidHeader);
    }
    let version = body[0];
    let mut r = Reader::new(body, offset_size, length_size)?;
    r.u8()?;
    match version {
        1 | 2 => {
            let _ndims = r.u8()?;
            let class = r.u8()?;
            r.skip(5)?;
            match class {
                HDF5_LAYOUT_CHUNKED => Ok(ParsedLayout::Chunked),
                HDF5_LAYOUT_CONTIGUOUS => {
                    let addr = r.addr()?;
                    Ok(ParsedLayout::Contiguous { addr, size: 0 })
                }
                HDF5_LAYOUT_COMPACT => {
                    let sz = r.u32()? as usize;
                    Ok(ParsedLayout::Compact {
                        data: r.bytes(sz)?.to_vec(),
                    })
                }
                _ => Err(HDF5Error::InvalidHeader),
            }
        }
        3 | 4 => {
            let class = r.u8()?;
            match class {
                HDF5_LAYOUT_CHUNKED => Ok(ParsedLayout::Chunked),
                HDF5_LAYOUT_CONTIGUOUS => {
                    let addr = r.addr()?;
                    let size = r.length()?;
                    Ok(ParsedLayout::Contiguous { addr, size })
                }
                HDF5_LAYOUT_COMPACT => {
                    let sz = r.u16()? as usize;
                    Ok(ParsedLayout::Compact {
                        data: r.bytes(sz)?.to_vec(),
                    })
                }
                _ => Err(HDF5Error::InvalidHeader),
            }
        }
        v => Err(HDF5Error::UnsupportedVersion(v)),
    }
}

pub struct ParsedLink {
    pub name: String,
    pub ohdr: u64,
}

pub fn parse_link(body: &[u8], offset_size: u8) -> Result<ParsedLink> {
    let mut r = Reader::new(body, offset_size, 8)?;
    let version = r.u8()?;
    if version != 0 && version != 1 {
        return Err(HDF5Error::UnsupportedVersion(version));
    }
    let flags = r.u8()?;
    if flags & 0x08 != 0 {
        let _ty = r.u8()?;
    }
    if flags & 0x04 != 0 {
        let _ = r.u64()?;
    }
    if flags & 0x10 != 0 {
        let _ = r.u8()?;
    }
    let nsize = match flags & 0x03 {
        0 => 1,
        1 => 2,
        2 => 4,
        3 => 8,
        _ => 1,
    };
    let nlen = r.sized_uint(nsize)? as usize;
    if nlen > HDF5_MAX_NAME_LEN {
        return Err(HDF5Error::NameTooLong);
    }
    let nb = r.bytes(nlen)?;
    let name = String::from_utf8(nb.to_vec()).map_err(|_| HDF5Error::InvalidHeader)?;
    let ohdr = r.addr()?;
    Ok(ParsedLink { name, ohdr })
}

pub struct ParsedAttr {
    pub name: String,
    pub dtype: ParsedDType,
    pub data: Vec<u8>,
}

pub fn parse_attribute(body: &[u8], offset_size: u8, length_size: u8) -> Result<ParsedAttr> {
    if body.len() < 8 {
        return Err(HDF5Error::InvalidHeader);
    }
    let version = body[0];
    let mut r = Reader::new(body, offset_size, length_size)?;
    r.u8()?;
    let mut flags = 0u8;
    match version {
        1 => {
            r.u8()?;
        }
        2 | 3 => {
            flags = r.u8()?;
        }
        v => return Err(HDF5Error::UnsupportedVersion(v)),
    }
    let name_size = r.u16()? as usize;
    let dtype_size = r.u16()? as usize;
    let space_size = r.u16()? as usize;
    if version == 3 {
        let _cs = r.u8()?;
    }
    let name_raw = r.bytes(name_size)?;
    let name_end = name_raw
        .iter()
        .position(|&b| b == 0)
        .unwrap_or(name_raw.len());
    let name =
        String::from_utf8(name_raw[..name_end].to_vec()).map_err(|_| HDF5Error::InvalidHeader)?;
    if version == 1 {
        let pad = (8 - (name_size % 8)) % 8;
        r.skip(pad)?;
    }
    if flags & 1 != 0 {
        return Err(HDF5Error::InvalidHeader);
    }
    let dtype_bytes = r.bytes(dtype_size)?;
    if version == 1 {
        let pad = (8 - (dtype_size % 8)) % 8;
        r.skip(pad)?;
    }
    let dtype = parse_datatype(dtype_bytes)?;
    r.skip(space_size)?;
    if version == 1 {
        let pad = (8 - (space_size % 8)) % 8;
        r.skip(pad)?;
    }
    let data = r.bytes(r.remaining())?.to_vec();
    Ok(ParsedAttr { name, dtype, data })
}

pub fn parse_link_info_heap(body: &[u8], offset_size: u8) -> Result<Option<u64>> {
    if body.len() < 2 {
        return Ok(None);
    }
    let mut r = Reader::new(body, offset_size, 8)?;
    let _ver = r.u8()?;
    let flags = r.u8()?;
    if flags & 1 != 0 {
        let _ = r.u64()?;
    }
    let heap = r.addr()?;
    if r.is_undef(heap) {
        Ok(None)
    } else {
        Ok(Some(heap))
    }
}

pub fn parse_symbol_table(body: &[u8], offset_size: u8) -> Result<(u64, u64)> {
    let mut r = Reader::new(body, offset_size, 8)?;
    let btree = r.addr()?;
    let heap = r.addr()?;
    Ok((btree, heap))
}

pub fn cstr_from_heap(heap: &[u8], offset: u64) -> Result<String> {
    let start = usize::try_from(offset).map_err(|_| HDF5Error::Truncated)?;
    if start >= heap.len() {
        return Err(HDF5Error::Truncated);
    }
    let rest = &heap[start..];
    let end = rest.iter().position(|&b| b == 0).unwrap_or(rest.len());
    String::from_utf8(rest[..end].to_vec()).map_err(|_| HDF5Error::InvalidHeader)
}
