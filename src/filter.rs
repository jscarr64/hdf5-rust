//! HDF5 filter pipeline parse and apply (deflate/gzip). Pure Rust.

use alloc::vec::Vec;

use crate::buf::Reader;
use crate::error::{HDF5Error, Result};
use crate::inflate::inflate_zlib;

/// One filter in the pipeline (encode order).
#[derive(Clone, Debug)]
pub struct FilterDesc {
    /// HDF5 filter identifier (1 = deflate).
    pub id: u16,
    /// Client data words from the pipeline message (retained for future filters).
    #[allow(dead_code)]
    pub client_data: Vec<u32>,
}

/// HDF5 deflate / gzip filter id.
pub const HDF5_FILTER_DEFLATE: u16 = 1;
/// Shuffle filter id (not implemented; returns FilteredNotSupported).
#[allow(dead_code)]
pub const HDF5_FILTER_SHUFFLE: u16 = 2;

/// Parse a filter-pipeline message into ordered filter descriptors.
pub fn parse_filters(body: &[u8]) -> Result<Vec<FilterDesc>> {
    if body.len() < 2 {
        return Ok(Vec::new());
    }
    let version = body[0];
    let nfilters = body[1] as usize;
    match version {
        1 => parse_filters_v1(body, nfilters),
        2 => parse_filters_v2(body, nfilters),
        v => Err(HDF5Error::UnsupportedVersion(v)),
    }
}

fn parse_filters_v1(body: &[u8], nfilters: usize) -> Result<Vec<FilterDesc>> {
    let mut r = Reader::new(body, 8, 8)?;
    r.u8()?;
    r.u8()?;
    r.skip(6)?;
    let mut out = Vec::with_capacity(nfilters);
    for _ in 0..nfilters {
        let id = r.u16()?;
        let name_len = r.u16()? as usize;
        let _flags = r.u16()?;
        let ncd = r.u16()? as usize;
        let _name = r.bytes(name_len)?;
        let pad = (8 - (name_len % 8)) % 8;
        r.skip(pad)?;
        let mut cd = Vec::with_capacity(ncd);
        for _ in 0..ncd {
            cd.push(r.u32()?);
        }
        let pad = (8 - ((ncd * 4) % 8)) % 8;
        r.skip(pad)?;
        out.push(FilterDesc {
            id,
            client_data: cd,
        });
    }
    Ok(out)
}

fn parse_filters_v2(body: &[u8], nfilters: usize) -> Result<Vec<FilterDesc>> {
    // Standard: id, name_len, flags, ncd, [name], cd...
    // Compact (no name_len field): id, flags, ncd, cd... — fits 12-byte deflate messages.
    if let Ok(v) = parse_filters_v2_named(body, nfilters) {
        return Ok(v);
    }
    parse_filters_v2_compact(body, nfilters)
}

fn parse_filters_v2_named(body: &[u8], nfilters: usize) -> Result<Vec<FilterDesc>> {
    let mut r = Reader::new(body, 8, 8)?;
    r.u8()?;
    r.u8()?;
    let mut out = Vec::with_capacity(nfilters);
    for _ in 0..nfilters {
        let id = r.u16()?;
        let name_len = r.u16()? as usize;
        let _flags = r.u16()?;
        let ncd = r.u16()? as usize;
        if name_len > 0 {
            let _name = r.bytes(name_len)?;
        }
        let mut cd = Vec::with_capacity(ncd);
        for _ in 0..ncd {
            cd.push(r.u32()?);
        }
        out.push(FilterDesc {
            id,
            client_data: cd,
        });
    }
    if r.remaining() != 0 {
        // Trailing bytes usually mean we mis-parsed; fall through to compact.
        return Err(HDF5Error::InvalidHeader);
    }
    Ok(out)
}

fn parse_filters_v2_compact(body: &[u8], nfilters: usize) -> Result<Vec<FilterDesc>> {
    let mut r = Reader::new(body, 8, 8)?;
    r.u8()?;
    r.u8()?;
    let mut out = Vec::with_capacity(nfilters);
    for _ in 0..nfilters {
        let id = r.u16()?;
        let _flags = r.u16()?;
        let ncd = r.u16()? as usize;
        let mut cd = Vec::with_capacity(ncd);
        for _ in 0..ncd {
            cd.push(r.u32()?);
        }
        out.push(FilterDesc {
            id,
            client_data: cd,
        });
    }
    Ok(out)
}

/// Apply filters in reverse pipeline order, honouring `filter_mask`.
/// Bit `i` set in the mask means filter `i` was skipped for this chunk.
pub fn apply_filters(
    mut data: Vec<u8>,
    filters: &[FilterDesc],
    filter_mask: u32,
    expected_len: usize,
) -> Result<Vec<u8>> {
    if filters.is_empty() {
        return Ok(data);
    }
    for (i, f) in filters.iter().enumerate().rev() {
        if filter_mask & (1u32 << i) != 0 {
            continue;
        }
        match f.id {
            HDF5_FILTER_DEFLATE => {
                data = inflate_zlib(&data, Some(expected_len))?;
            }
            _ => return Err(HDF5Error::FilteredNotSupported),
        }
    }
    if data.len() != expected_len {
        return Err(HDF5Error::InvalidHeader);
    }
    Ok(data)
}
