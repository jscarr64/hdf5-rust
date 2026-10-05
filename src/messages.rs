//! Object-header message bodies this crate writes and reads.

use alloc::string::String;
use alloc::vec::Vec;

use crate::buf::{push_u16, push_u32, push_u64, Reader};
use crate::error::{HDF5Error, Result};
use crate::{
    HDF5_ATTR_VERSION, HDF5_CLASS_COMPOUND, HDF5_CLASS_FLOAT, HDF5_CLASS_INTEGER,
    HDF5_CLASS_OPAQUE, HDF5_CLASS_STRING, HDF5_CLASS_VLEN, HDF5_DATASPACE_VERSION,
    HDF5_DTYPE_VERSION, HDF5_FILL_VERSION, HDF5_LAYOUT_CHUNKED, HDF5_LAYOUT_COMPACT,
    HDF5_LAYOUT_CONTIGUOUS, HDF5_LAYOUT_VERSION, HDF5_LINK_VERSION, HDF5_MAX_DIMS,
    HDF5_MAX_NAME_LEN, HDF5_OPAQUE_TAG, HDF5_SPACE_SCALAR, HDF5_SPACE_SIMPLE, HDF5_UNDEF_ADDR_8,
    HDF5_UNLIMITED,
};

/// In-memory datatype after parsing a datatype message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ParsedDType {
    Float64Le,
    Float32Le,
    Int8Le,
    Int16Le,
    Int32Le,
    Int64Le,
    UInt8Le,
    UInt16Le,
    UInt32Le,
    UInt64Le,
    /// IEEE binary64 big-endian (normalized to LE lanes on read).
    Float64Be,
    /// IEEE binary32 big-endian (normalized to LE lanes on read).
    Float32Be,
    Int8Be,
    Int16Be,
    Int32Be,
    Int64Be,
    UInt8Be,
    UInt16Be,
    UInt32Be,
    UInt64Be,
    Opaque {
        size: usize,
    },
    FixedString {
        size: usize,
    },
    VlenString,
    /// Compound / structured type with known field layout.
    Compound {
        size: usize,
        fields: Vec<CompoundField>,
    },
    Other {
        size: usize,
    },
}

/// One member of a compound datatype.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompoundField {
    /// Field name from the datatype message.
    pub name: String,
    /// Byte offset within the compound element.
    pub offset: usize,
    /// Field size in bytes.
    pub size: usize,
    /// Simplified member class for introspection (not a full recursive tree).
    pub kind: CompoundMemberKind,
}

/// Member kind reported for compound field introspection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompoundMemberKind {
    /// IEEE binary64 little-endian.
    Float64Le,
    /// IEEE binary32 little-endian.
    Float32Le,
    /// Signed integer, little-endian, `size` bytes.
    IntLe {
        /// Width in bytes.
        size: usize,
    },
    /// Unsigned integer, little-endian, `size` bytes.
    UIntLe {
        /// Width in bytes.
        size: usize,
    },
    /// Any other / nested / BE member.
    Other {
        /// Width in bytes.
        size: usize,
    },
}

impl ParsedDType {
    #[allow(dead_code)]
    pub fn element_size(&self) -> usize {
        match *self {
            Self::Float64Le
            | Self::Float64Be
            | Self::Int64Le
            | Self::Int64Be
            | Self::UInt64Le
            | Self::UInt64Be => 8,
            Self::Float32Le
            | Self::Float32Be
            | Self::Int32Le
            | Self::Int32Be
            | Self::UInt32Le
            | Self::UInt32Be => 4,
            Self::Int16Le | Self::Int16Be | Self::UInt16Le | Self::UInt16Be => 2,
            Self::Int8Le | Self::Int8Be | Self::UInt8Le | Self::UInt8Be => 1,
            Self::Opaque { size } => size,
            Self::FixedString { size } => size,
            Self::VlenString => 0,
            Self::Compound { size, .. } | Self::Other { size } => size,
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

pub fn encode_integer(size: u32, signed: bool) -> Vec<u8> {
    let mut v = Vec::with_capacity(12);
    v.push((HDF5_DTYPE_VERSION << 4) | HDF5_CLASS_INTEGER);
    v.push(if signed { 0x08 } else { 0x00 });
    v.push(0);
    v.push(0);
    push_u32(&mut v, size);
    push_u16(&mut v, 0);
    let prec = size.saturating_mul(8);
    push_u16(&mut v, prec as u16);
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
    let mut v = alloc::vec![
        HDF5_DATASPACE_VERSION,
        dims.len() as u8,
        1,
        HDF5_SPACE_SIMPLE,
    ];
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
    let _bit1 = body[2];
    let size = u32::from_le_bytes([body[4], body[5], body[6], body[7]]) as usize;
    match class {
        HDF5_CLASS_INTEGER => parse_integer(bit0, size, body),
        HDF5_CLASS_FLOAT => {
            // Byte order: bits 0-0 of class bitfield byte1? HDF5: bit 0 = BE when set with bit 6.
            // (bit0 & 0x41) == 0 means little-endian for the float class bitfield.
            let le = (bit0 & 0x41) == 0;
            match (le, size) {
                (true, 8) => Ok(ParsedDType::Float64Le),
                (true, 4) => Ok(ParsedDType::Float32Le),
                (false, 8) => Ok(ParsedDType::Float64Be),
                (false, 4) => Ok(ParsedDType::Float32Be),
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
        HDF5_CLASS_COMPOUND => parse_compound(body, size),
        _ => Ok(ParsedDType::Other { size }),
    }
}

/// Precision and bit offset of an integer datatype, when the message carries them.
pub struct IntegerBits {
    /// Significant bits.
    pub precision: usize,
    /// Offset of those bits from the least significant bit.
    pub offset: usize,
    /// Two's-complement signed integer.
    pub signed: bool,
}

/// Integer bit layout from a datatype message, if this is an integer class.
pub fn integer_bits(body: &[u8]) -> Option<IntegerBits> {
    if body.len() < 12 {
        return None;
    }
    let class = body[0] & 0x0F;
    if class != HDF5_CLASS_INTEGER {
        return None;
    }
    let size = u32::from_le_bytes([body[4], body[5], body[6], body[7]]) as usize;
    let offset = u16::from_le_bytes([body[8], body[9]]) as usize;
    let precision = u16::from_le_bytes([body[10], body[11]]) as usize;
    let signed = (body[1] & 0x08) != 0;
    let width = size.saturating_mul(8);
    if precision == 0 || precision > width || offset.saturating_add(precision) > width {
        return None;
    }
    Some(IntegerBits {
        precision,
        offset,
        signed,
    })
}

fn parse_integer(bit0: u8, size: usize, body: &[u8]) -> Result<ParsedDType> {
    let le = (bit0 & 0x01) == 0;
    let signed = (bit0 & 0x08) != 0;
    if body.len() >= 12 {
        let offset = u16::from_le_bytes([body[8], body[9]]) as usize;
        let prec = u16::from_le_bytes([body[10], body[11]]) as usize;
        let width = size.saturating_mul(8);
        if prec == 0 || offset.saturating_add(prec) > width {
            return Ok(ParsedDType::Other { size });
        }
    }
    Ok(match (le, signed, size) {
        (true, true, 1) => ParsedDType::Int8Le,
        (true, true, 2) => ParsedDType::Int16Le,
        (true, true, 4) => ParsedDType::Int32Le,
        (true, true, 8) => ParsedDType::Int64Le,
        (true, false, 1) => ParsedDType::UInt8Le,
        (true, false, 2) => ParsedDType::UInt16Le,
        (true, false, 4) => ParsedDType::UInt32Le,
        (true, false, 8) => ParsedDType::UInt64Le,
        (false, true, 1) => ParsedDType::Int8Be,
        (false, true, 2) => ParsedDType::Int16Be,
        (false, true, 4) => ParsedDType::Int32Be,
        (false, true, 8) => ParsedDType::Int64Be,
        (false, false, 1) => ParsedDType::UInt8Be,
        (false, false, 2) => ParsedDType::UInt16Be,
        (false, false, 4) => ParsedDType::UInt32Be,
        (false, false, 8) => ParsedDType::UInt64Be,
        _ => ParsedDType::Other { size },
    })
}

fn parse_compound(body: &[u8], size: usize) -> Result<ParsedDType> {
    if body.len() < 8 {
        return Err(HDF5Error::InvalidHeader);
    }
    let class_ver = body[0];
    let version = (class_ver >> 4) & 0x0F;
    let nmembers = body[1] as usize;
    // Properties start after the 8-byte datatype header.
    let mut pos = 8usize;
    let mut fields = Vec::with_capacity(nmembers);
    for _ in 0..nmembers {
        if pos >= body.len() {
            return Err(HDF5Error::Truncated);
        }
        // Name is null-terminated; v1 padded to 8 bytes.
        let rest = &body[pos..];
        let name_end = rest
            .iter()
            .position(|&b| b == 0)
            .ok_or(HDF5Error::InvalidHeader)?;
        let name =
            String::from_utf8(rest[..name_end].to_vec()).map_err(|_| HDF5Error::InvalidHeader)?;
        pos += name_end + 1;
        if version == 1 {
            let pad = (8 - ((name_end + 1) % 8)) % 8;
            pos += pad;
        }
        if pos + 4 > body.len() {
            return Err(HDF5Error::Truncated);
        }
        let offset =
            u32::from_le_bytes([body[pos], body[pos + 1], body[pos + 2], body[pos + 3]]) as usize;
        pos += 4;
        if version == 1 {
            // v1: ndims(1) + reserved(3) + permutation(4) + reserved(4)
            // + four dimension sizes (always 4 × u32), even when ndims == 0.
            if pos + 28 > body.len() {
                return Err(HDF5Error::Truncated);
            }
            let _mdims = body[pos];
            pos += 28;
        }
        if pos + 8 > body.len() {
            return Err(HDF5Error::Truncated);
        }
        // Member datatype is embedded recursively; size at bytes 4..8 of member header.
        let msize = u32::from_le_bytes([body[pos + 4], body[pos + 5], body[pos + 6], body[pos + 7]])
            as usize;
        let member_body = &body[pos..];
        let member_dt = parse_datatype(member_body)?;
        let member_total = datatype_message_len(member_body)?;
        pos += member_total;
        let kind = compound_member_kind(&member_dt);
        fields.push(CompoundField {
            name,
            offset,
            size: msize,
            kind,
        });
    }
    Ok(ParsedDType::Compound { size, fields })
}

fn compound_member_kind(dt: &ParsedDType) -> CompoundMemberKind {
    match *dt {
        ParsedDType::Float64Le => CompoundMemberKind::Float64Le,
        ParsedDType::Float32Le => CompoundMemberKind::Float32Le,
        ParsedDType::Int8Le => CompoundMemberKind::IntLe { size: 1 },
        ParsedDType::Int16Le => CompoundMemberKind::IntLe { size: 2 },
        ParsedDType::Int32Le => CompoundMemberKind::IntLe { size: 4 },
        ParsedDType::Int64Le => CompoundMemberKind::IntLe { size: 8 },
        ParsedDType::UInt8Le => CompoundMemberKind::UIntLe { size: 1 },
        ParsedDType::UInt16Le => CompoundMemberKind::UIntLe { size: 2 },
        ParsedDType::UInt32Le => CompoundMemberKind::UIntLe { size: 4 },
        ParsedDType::UInt64Le => CompoundMemberKind::UIntLe { size: 8 },
        ParsedDType::Opaque { size }
        | ParsedDType::FixedString { size }
        | ParsedDType::Compound { size, .. }
        | ParsedDType::Other { size } => CompoundMemberKind::Other { size },
        ParsedDType::Float64Be => CompoundMemberKind::Other { size: 8 },
        ParsedDType::Float32Be => CompoundMemberKind::Other { size: 4 },
        ParsedDType::Int8Be | ParsedDType::UInt8Be => CompoundMemberKind::Other { size: 1 },
        ParsedDType::Int16Be | ParsedDType::UInt16Be => CompoundMemberKind::Other { size: 2 },
        ParsedDType::Int32Be | ParsedDType::UInt32Be => CompoundMemberKind::Other { size: 4 },
        ParsedDType::Int64Be | ParsedDType::UInt64Be => CompoundMemberKind::Other { size: 8 },
        ParsedDType::VlenString => CompoundMemberKind::Other { size: 0 },
    }
}

/// Byte length of one datatype message including nested members.
fn datatype_message_len(body: &[u8]) -> Result<usize> {
    if body.len() < 8 {
        return Err(HDF5Error::Truncated);
    }
    let class_ver = body[0];
    let class = class_ver & 0x0F;
    let version = (class_ver >> 4) & 0x0F;
    let size = u32::from_le_bytes([body[4], body[5], body[6], body[7]]) as usize;
    match class {
        HDF5_CLASS_INTEGER => Ok(12),
        HDF5_CLASS_FLOAT => Ok(20),
        HDF5_CLASS_OPAQUE => {
            let tag_len = body[1] as usize;
            Ok(8 + tag_len)
        }
        HDF5_CLASS_STRING => Ok(8),
        HDF5_CLASS_COMPOUND => {
            // Re-parse by walking; expensive but rare.
            let nmembers = body[1] as usize;
            let mut pos = 8usize;
            for _ in 0..nmembers {
                let rest = &body[pos..];
                let name_end = rest
                    .iter()
                    .position(|&b| b == 0)
                    .ok_or(HDF5Error::InvalidHeader)?;
                pos += name_end + 1;
                if version == 1 {
                    pos += (8 - ((name_end + 1) % 8)) % 8;
                    pos += 4; // offset
                    if pos + 28 > body.len() {
                        return Err(HDF5Error::Truncated);
                    }
                    pos += 28;
                } else {
                    pos += 4; // offset
                }
                let nested = datatype_message_len(&body[pos..])?;
                pos += nested;
            }
            Ok(pos)
        }
        _ => {
            // Best-effort: header only; compound walker needs accurate nested sizes.
            let _ = size;
            Ok(8)
        }
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
    Contiguous {
        addr: u64,
        size: u64,
    },
    Compact {
        data: Vec<u8>,
    },
    /// Chunked storage. `index` describes how to find chunks.
    Chunked {
        chunk_dims: Vec<u32>,
        index: crate::chunk_index::ChunkIndex,
    },
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
            let ndims = r.u8()? as usize;
            let class = r.u8()?;
            r.skip(5)?;
            match class {
                HDF5_LAYOUT_CHUNKED => {
                    let addr = r.addr()?;
                    let mut chunk_dims = Vec::new();
                    for _ in 0..ndims {
                        chunk_dims.push(r.u32()?);
                    }
                    Ok(ParsedLayout::Chunked {
                        chunk_dims,
                        index: crate::chunk_index::ChunkIndex::BTreeV1 { addr },
                    })
                }
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
        3 => {
            let class = r.u8()?;
            match class {
                HDF5_LAYOUT_CHUNKED => {
                    let ndims = r.u8()? as usize;
                    let addr = r.addr()?;
                    let mut chunk_dims = Vec::new();
                    for _ in 0..ndims {
                        chunk_dims.push(r.u32()?);
                    }
                    Ok(ParsedLayout::Chunked {
                        chunk_dims,
                        index: crate::chunk_index::ChunkIndex::BTreeV1 { addr },
                    })
                }
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
        4 | 5 => {
            let class = r.u8()?;
            match class {
                HDF5_LAYOUT_CHUNKED => {
                    let flags = r.u8()?;
                    let ndims = r.u8()? as usize;
                    let enc = r.u8()? as usize;
                    if enc == 0 || enc > 8 {
                        return Err(HDF5Error::InvalidHeader);
                    }
                    let mut chunk_dims = Vec::with_capacity(ndims);
                    for _ in 0..ndims {
                        chunk_dims.push(r.sized_uint(enc as u8)? as u32);
                    }
                    let idx_type = r.u8()?;
                    // H5D_CHUNK_IDX_*: 1=single, 2=none/implicit, 3=farray, 4=earray, 5=bt2
                    let index = match idx_type {
                        1 => {
                            let (nbytes, filter_mask) = if flags & 0x02 != 0 {
                                let nb = r.length()?;
                                let fm = r.u32()?;
                                (
                                    Some(u32::try_from(nb).map_err(|_| HDF5Error::InvalidHeader)?),
                                    fm,
                                )
                            } else {
                                (None, 0u32)
                            };
                            let addr = r.addr()?;
                            crate::chunk_index::ChunkIndex::Single {
                                addr,
                                nbytes,
                                filter_mask,
                            }
                        }
                        3 => {
                            let _max_bits = r.u8()?;
                            let addr = r.addr()?;
                            crate::chunk_index::ChunkIndex::FixedArray { addr }
                        }
                        2 => {
                            let addr = r.addr()?;
                            crate::chunk_index::ChunkIndex::Implicit { addr }
                        }
                        4 => {
                            // max_nelmts_bits, idx_blk_elmts, sup_blk_min, data_blk_min, page bits.
                            r.skip(5)?;
                            let addr = r.addr()?;
                            crate::chunk_index::ChunkIndex::ExtensibleArray { addr }
                        }
                        5 => {
                            // node size, split percent, merge percent; the header is authoritative.
                            r.u32()?;
                            r.u8()?;
                            r.u8()?;
                            let addr = r.addr()?;
                            crate::chunk_index::ChunkIndex::BTreeV2 { addr }
                        }
                        _ => return Err(HDF5Error::ChunkedNotSupported),
                    };
                    let _ = flags;
                    Ok(ParsedLayout::Chunked { chunk_dims, index })
                }
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

/// Number of filters in a filter-pipeline message. `0` if the body is empty.
#[allow(dead_code)]
pub fn parse_filter_count(body: &[u8]) -> Result<u8> {
    if body.len() < 2 {
        return Ok(0);
    }
    match body[0] {
        1 | 2 => Ok(body[1]),
        _ => Err(HDF5Error::UnsupportedVersion(body[0])),
    }
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
