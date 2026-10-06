//! HDF5 filter pipeline parse and apply. Pure Rust.
//!
//! Implemented on read: deflate/gzip, shuffle, Fletcher32, atomic n-bit,
//! and integer scale-offset. SZIP (Rice / libaec) and floating-point
//! scale-offset return [`HDF5Error::FilteredNotSupported`]. Integer
//! scale-offset is decoded as a little-endian writer would have packed it:
//! HDF5 does not store the writer's memory byte order.
//!
//! Atomic n-bit, little-endian integer scale-offset, and Fletcher32 are
//! derived from the HDF5 library (3-clause BSD). See the crate `NOTICE`.

use alloc::vec::Vec;

use crate::buf::Reader;
use crate::error::{HDF5Error, Result};
use crate::inflate::inflate_zlib;

/// One filter in the pipeline (encode order).
#[derive(Clone, Debug)]
pub struct FilterDesc {
    /// HDF5 filter identifier.
    pub id: u16,
    /// Client data words from the pipeline message.
    pub client_data: Vec<u32>,
}

/// Deflate / gzip filter id.
pub const HDF5_FILTER_DEFLATE: u16 = 1;
/// Byte shuffle filter id.
pub const HDF5_FILTER_SHUFFLE: u16 = 2;
/// Fletcher32 checksum filter id.
pub const HDF5_FILTER_FLETCHER32: u16 = 3;
/// SZIP (Rice) filter id. Not implemented: the bitstream is libaec.
pub const HDF5_FILTER_SZIP: u16 = 4;
/// N-bit packing filter id. Atomic integers only.
pub const HDF5_FILTER_NBIT: u16 = 5;
/// Scale-offset filter id. Integers only.
pub const HDF5_FILTER_SCALEOFFSET: u16 = 6;

const NBIT_ATOMIC: u32 = 1;
const SCALEOFFSET_NPARMS: usize = 20;
const SCALEOFFSET_INT: u32 = 2;
const SCALEOFFSET_HEADER: usize = 21;

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
        if f.id == HDF5_FILTER_SZIP {
            return Err(HDF5Error::FilteredNotSupported);
        }
        data = match f.id {
            HDF5_FILTER_DEFLATE => inflate_zlib(&data, Some(expected_len))?,
            HDF5_FILTER_SHUFFLE => unshuffle(&data, &f.client_data)?,
            HDF5_FILTER_FLETCHER32 => check_fletcher32(&data)?,
            HDF5_FILTER_NBIT => decode_nbit(&data, &f.client_data)?,
            HDF5_FILTER_SCALEOFFSET => decode_scaleoffset(&data, &f.client_data)?,
            _ => return Err(HDF5Error::FilteredNotSupported),
        };
    }
    if data.len() != expected_len {
        return Err(HDF5Error::InvalidHeader);
    }
    Ok(data)
}

fn unshuffle(data: &[u8], cd: &[u32]) -> Result<Vec<u8>> {
    let elem = cd.first().copied().ok_or(HDF5Error::InvalidHeader)? as usize;
    if elem == 0 || data.len() % elem != 0 {
        return Err(HDF5Error::InvalidHeader);
    }
    let n = data.len() / elem;
    let mut out = alloc::vec![0u8; data.len()];
    for i in 0..elem {
        for j in 0..n {
            out[j * elem + i] = data[i * n + j];
        }
    }
    Ok(out)
}

/// Fletcher32 over `data`.
///
/// Derived from HDF5 `H5_checksum_fletcher32` in `H5checksum.c`.
/// HDF5 3-clause BSD. See the crate `NOTICE`.
fn fletcher32(data: &[u8]) -> u32 {
    let mut sum1: u32 = 0;
    let mut sum2: u32 = 0;
    let mut i = 0usize;
    let mut words = data.len() / 2;
    while words > 0 {
        let tlen = words.min(360);
        words -= tlen;
        for _ in 0..tlen {
            let word = (u32::from(data[i]) << 8) | u32::from(data[i + 1]);
            i += 2;
            sum1 = sum1.wrapping_add(word);
            sum2 = sum2.wrapping_add(sum1);
        }
        sum1 = (sum1 & 0xffff) + (sum1 >> 16);
        sum2 = (sum2 & 0xffff) + (sum2 >> 16);
    }
    if data.len() % 2 == 1 {
        sum1 = sum1.wrapping_add(u32::from(data[i]) << 8);
        sum2 = sum2.wrapping_add(sum1);
        sum1 = (sum1 & 0xffff) + (sum1 >> 16);
        sum2 = (sum2 & 0xffff) + (sum2 >> 16);
    }
    sum1 = (sum1 & 0xffff) + (sum1 >> 16);
    sum2 = (sum2 & 0xffff) + (sum2 >> 16);
    (sum2 << 16) | (sum1 & 0xffff)
}

/// Drop a trailing Fletcher32 word when it matches the payload.
///
/// Accepts the big-endian word, the native word, and the pre-1.6.3
/// pair-swapped word. That acceptance is derived from
/// `H5Z__filter_fletcher32` in `H5Zfletcher32.c`. HDF5 3-clause BSD.
/// See the crate `NOTICE`.
fn check_fletcher32(data: &[u8]) -> Result<Vec<u8>> {
    if data.len() < 4 {
        return Err(HDF5Error::InvalidHeader);
    }
    let (payload, tail) = data.split_at(data.len() - 4);
    let bytes = [tail[0], tail[1], tail[2], tail[3]];
    // HDF5 1.x stored this word big-endian. HDF5 2.0 on a little-endian
    // writer stores the native (little-endian) word. Accept either, and the
    // pair-swapped value written before 1.6.3.
    let stored_le = u32::from_le_bytes(bytes);
    let stored_be = u32::from_be_bytes(bytes);
    let got = fletcher32(payload);
    let reversed = reversed_fletcher(got);
    if stored_le != got && stored_be != got && stored_le != reversed && stored_be != reversed {
        return Err(HDF5Error::InvalidHeader);
    }
    Ok(payload.to_vec())
}

/// Pair-swapped checksum HDF5 accepts for files written before 1.6.3.
///
/// Derived from the byte swaps in `H5Z__filter_fletcher32`
/// (`H5Zfletcher32.c`). HDF5 3-clause BSD. See the crate `NOTICE`.
fn reversed_fletcher(f: u32) -> u32 {
    let mut c = f.to_ne_bytes();
    c.swap(0, 1);
    c.swap(2, 3);
    u32::from_ne_bytes(c)
}

/// Decode an atomic n-bit chunk.
///
/// The bit walk is derived from `H5Z__nbit_decompress_one_atomic` and
/// `H5Z__nbit_decompress_one_byte` in `H5Znbit.c`. HDF5 3-clause BSD.
/// See the crate `NOTICE`. Array and compound n-bit types return
/// [`HDF5Error::FilteredNotSupported`].
fn decode_nbit(data: &[u8], cd: &[u32]) -> Result<Vec<u8>> {
    let nparms = cd.first().copied().ok_or(HDF5Error::InvalidHeader)? as usize;
    if nparms != cd.len() || cd.len() < 8 {
        return Err(HDF5Error::InvalidHeader);
    }
    if cd[1] != 0 {
        return Ok(data.to_vec());
    }
    if cd[3] != NBIT_ATOMIC {
        return Err(HDF5Error::FilteredNotSupported);
    }
    let d_nelmts = cd[2] as usize;
    let size = cd[4] as usize;
    let order = cd[5];
    let precision = cd[6];
    let offset = cd[7];
    if size == 0 || size > 8 || order > 1 {
        return Err(HDF5Error::InvalidHeader);
    }
    let type_bits = (size as u32).saturating_mul(8);
    if precision == 0 || precision > type_bits || precision.saturating_add(offset) > type_bits {
        return Err(HDF5Error::InvalidHeader);
    }
    let spec = NbitSpec {
        size,
        order,
        precision,
        offset,
    };
    let out_len = d_nelmts.checked_mul(size).ok_or(HDF5Error::InvalidHeader)?;
    let mut out = alloc::vec![0u8; out_len];
    let mut bits = BitIn {
        buffer: data,
        j: 0,
        buf_len: 8,
    };
    for i in 0..d_nelmts {
        nbit_one_atomic(&mut out, i * size, &mut bits, &spec)?;
    }
    Ok(out)
}

struct NbitSpec {
    size: usize,
    order: u32,
    precision: u32,
    offset: u32,
}

struct BitIn<'a> {
    buffer: &'a [u8],
    j: usize,
    buf_len: u32,
}

impl BitIn<'_> {
    fn bump(&mut self) {
        self.j += 1;
        self.buf_len = 8;
    }
}

/// One atomic value. Derived from HDF5 `H5Z__nbit_decompress_one_atomic`
/// (`H5Znbit.c`), 3-clause BSD. See the crate `NOTICE`.
fn nbit_one_atomic(
    data: &mut [u8],
    data_offset: usize,
    bits: &mut BitIn<'_>,
    spec: &NbitSpec,
) -> Result<()> {
    let datatype_len = (spec.size as u32).saturating_mul(8);
    if spec.order == 0 {
        let begin_i = if (spec.precision + spec.offset) % 8 != 0 {
            (spec.precision + spec.offset) / 8
        } else {
            (spec.precision + spec.offset) / 8 - 1
        };
        let end_i = spec.offset / 8;
        let mut k = begin_i as i32;
        while k >= end_i as i32 {
            nbit_one_byte(data, data_offset, k as u32, begin_i, end_i, bits, spec)?;
            k -= 1;
        }
    } else {
        let begin_i = (datatype_len - spec.precision - spec.offset) / 8;
        let end_i = if spec.offset % 8 != 0 {
            (datatype_len - spec.offset) / 8
        } else {
            (datatype_len - spec.offset) / 8 - 1
        };
        for k in begin_i..=end_i {
            nbit_one_byte(data, data_offset, k, begin_i, end_i, bits, spec)?;
        }
    }
    Ok(())
}

/// One byte of an atomic n-bit value. Derived from HDF5
/// `H5Z__nbit_decompress_one_byte` (`H5Znbit.c`), 3-clause BSD.
/// See the crate `NOTICE`.
fn nbit_one_byte(
    data: &mut [u8],
    data_offset: usize,
    k: u32,
    begin_i: u32,
    end_i: u32,
    bits: &mut BitIn<'_>,
    spec: &NbitSpec,
) -> Result<()> {
    let at = data_offset + k as usize;
    if at >= data.len() || bits.j >= bits.buffer.len() {
        return Err(HDF5Error::InvalidHeader);
    }
    let mut val = bits.buffer[bits.j];
    let (dat_len, dat_offset) = if begin_i != end_i {
        if k == begin_i {
            (
                8 - (spec.size as u32 * 8 - spec.precision - spec.offset) % 8,
                0u32,
            )
        } else if k == end_i {
            let dat_len = 8 - spec.offset % 8;
            (dat_len, 8 - dat_len)
        } else {
            (8u32, 0u32)
        }
    } else {
        (spec.precision, spec.offset % 8)
    };
    if dat_len == 0 || dat_len > 8 {
        return Err(HDF5Error::InvalidHeader);
    }
    if bits.buf_len > dat_len {
        data[at] = (((u32::from(val) >> (bits.buf_len - dat_len)) & mask_bits(dat_len))
            << dat_offset) as u8;
        bits.buf_len -= dat_len;
    } else {
        data[at] = ((u32::from(val & low_mask_u8(bits.buf_len)) << (dat_len - bits.buf_len))
            << dat_offset) as u8;
        let left = dat_len - bits.buf_len;
        bits.bump();
        if left == 0 {
            return Ok(());
        }
        if bits.j >= bits.buffer.len() {
            return Err(HDF5Error::InvalidHeader);
        }
        val = bits.buffer[bits.j];
        data[at] |=
            (((u32::from(val) >> (bits.buf_len - left)) & mask_bits(left)) << dat_offset) as u8;
        bits.buf_len -= left;
    }
    Ok(())
}

fn mask_bits(n: u32) -> u32 {
    if n >= 32 {
        u32::MAX
    } else {
        (1u32 << n) - 1
    }
}

fn low_mask_u8(n: u32) -> u8 {
    if n >= 8 {
        0xff
    } else {
        ((1u32 << n) - 1) as u8
    }
}

/// Decode integer scale-offset.
///
/// The little-endian bit unpack is derived from
/// `H5Z__scaleoffset_decompress_one_atomic` and
/// `H5Z__scaleoffset_decompress_one_byte` in `H5Zscaleoffset.c`.
/// HDF5 3-clause BSD. See the crate `NOTICE`. Floating-point
/// scale-offset returns [`HDF5Error::FilteredNotSupported`].
fn decode_scaleoffset(data: &[u8], cd: &[u32]) -> Result<Vec<u8>> {
    if cd.len() != SCALEOFFSET_NPARMS {
        return Err(HDF5Error::InvalidHeader);
    }
    let scale_type = cd[0];
    let class = cd[3];
    if class != 0 || scale_type == 0 || scale_type == 1 {
        return Err(HDF5Error::FilteredNotSupported);
    }
    if scale_type != SCALEOFFSET_INT {
        return Err(HDF5Error::InvalidHeader);
    }
    let d_nelmts = cd[2] as usize;
    let size = cd[4] as usize;
    let signed = cd[5] != 0;
    let order = cd[6];
    let filavail = cd[7];
    if size == 0 || size > 8 || order > 1 || filavail > 1 {
        return Err(HDF5Error::InvalidHeader);
    }
    let type_bits = (size as u32).saturating_mul(8);
    let mut scale_factor = cd[1];
    if scale_factor >= 0x8000_0000 {
        scale_factor = 0;
    }
    if scale_factor > type_bits {
        return Err(HDF5Error::InvalidHeader);
    }
    if scale_factor == type_bits {
        return Ok(data.to_vec());
    }
    if data.len() < SCALEOFFSET_HEADER {
        return Err(HDF5Error::InvalidHeader);
    }
    let minbits = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
    if minbits > type_bits {
        return Err(HDF5Error::InvalidHeader);
    }
    let stored_minval_size = data[4] as usize;
    let minval_size = stored_minval_size.min(8);
    if 5 + minval_size > data.len() {
        return Err(HDF5Error::InvalidHeader);
    }
    let mut min_bytes = [0u8; 8];
    min_bytes[..minval_size].copy_from_slice(&data[5..5 + minval_size]);
    let minval = u64::from_le_bytes(min_bytes);
    let sminval = minval as i64;
    let out_len = d_nelmts.checked_mul(size).ok_or(HDF5Error::InvalidHeader)?;
    let mut out = alloc::vec![0u8; out_len];
    if minbits == type_bits {
        let payload = data
            .get(SCALEOFFSET_HEADER..)
            .ok_or(HDF5Error::InvalidHeader)?;
        if payload.len() < out_len {
            return Err(HDF5Error::InvalidHeader);
        }
        out.copy_from_slice(&payload[..out_len]);
    } else if minbits != 0 {
        scaleoffset_unpack(
            &mut out,
            &data[SCALEOFFSET_HEADER..],
            d_nelmts,
            size,
            minbits,
        )?;
    }
    if minbits != type_bits {
        let sentinel = if minbits >= 64 {
            u64::MAX
        } else {
            (1u64 << minbits) - 1
        };
        let fill = fill_bytes(cd, size);
        for i in 0..d_nelmts {
            let slot = &mut out[i * size..(i + 1) * size];
            let raw = read_uint_le(slot);
            if filavail == 1 && raw == sentinel {
                slot.copy_from_slice(&fill[..size]);
            } else if signed {
                let sum = sign_extend(raw, size).wrapping_add(sminval);
                write_uint_le(slot, sum as u64);
            } else {
                write_uint_le(slot, raw.wrapping_add(minval));
            }
        }
    }
    if order == 1 && size > 1 {
        for chunk in out.chunks_exact_mut(size) {
            chunk.reverse();
        }
    }
    Ok(out)
}

fn fill_bytes(cd: &[u32], size: usize) -> [u8; 8] {
    let mut fill = [0u8; 8];
    let mut rem = size;
    let mut word_i = 8usize;
    let mut off = 0usize;
    while rem > 0 {
        let n = rem.min(4);
        let word = cd.get(word_i).copied().unwrap_or(0);
        fill[off..off + n].copy_from_slice(&word.to_le_bytes()[..n]);
        off += n;
        rem -= n;
        word_i += 1;
    }
    fill
}

fn scaleoffset_unpack(
    out: &mut [u8],
    buffer: &[u8],
    d_nelmts: usize,
    size: usize,
    minbits: u32,
) -> Result<()> {
    let mut bits = BitIn {
        buffer,
        j: 0,
        buf_len: 8,
    };
    for i in 0..d_nelmts {
        so_one_atomic(out, i * size, &mut bits, size, minbits)?;
    }
    Ok(())
}

/// One integer value, little-endian arm. Derived from HDF5
/// `H5Z__scaleoffset_decompress_one_atomic` (`H5Zscaleoffset.c`),
/// 3-clause BSD. See the crate `NOTICE`.
fn so_one_atomic(
    data: &mut [u8],
    data_offset: usize,
    bits: &mut BitIn<'_>,
    size: usize,
    minbits: u32,
) -> Result<()> {
    let dtype_len = (size as u32).saturating_mul(8);
    let begin_i = size as u32 - 1 - (dtype_len - minbits) / 8;
    let mut k = begin_i as i32;
    while k >= 0 {
        so_one_byte(
            data,
            data_offset + k as usize,
            &mut *bits,
            begin_i,
            minbits,
            dtype_len,
        )?;
        k -= 1;
    }
    Ok(())
}

/// One byte of an integer scale-offset value. Derived from HDF5
/// `H5Z__scaleoffset_decompress_one_byte` (`H5Zscaleoffset.c`),
/// 3-clause BSD, little-endian arm. See the crate `NOTICE`.
fn so_one_byte(
    data: &mut [u8],
    at: usize,
    bits: &mut BitIn<'_>,
    begin_i: u32,
    minbits: u32,
    dtype_len: u32,
) -> Result<()> {
    if at >= data.len() || bits.j >= bits.buffer.len() {
        return Err(HDF5Error::InvalidHeader);
    }
    let mut val = bits.buffer[bits.j];
    let on_first = at % (dtype_len as usize / 8) == begin_i as usize;
    let dat_len = if on_first {
        8 - (dtype_len - minbits) % 8
    } else {
        8
    };
    if bits.buf_len > dat_len {
        data[at] = ((u32::from(val) >> (bits.buf_len - dat_len)) & mask_bits(dat_len)) as u8;
        bits.buf_len -= dat_len;
    } else {
        data[at] = (u32::from(val & low_mask_u8(bits.buf_len)) << (dat_len - bits.buf_len)) as u8;
        let left = dat_len - bits.buf_len;
        bits.bump();
        if left == 0 {
            return Ok(());
        }
        if bits.j >= bits.buffer.len() {
            return Err(HDF5Error::InvalidHeader);
        }
        val = bits.buffer[bits.j];
        data[at] |= ((u32::from(val) >> (bits.buf_len - left)) & mask_bits(left)) as u8;
        bits.buf_len -= left;
    }
    Ok(())
}

fn read_uint_le(buf: &[u8]) -> u64 {
    let mut tmp = [0u8; 8];
    tmp[..buf.len()].copy_from_slice(buf);
    u64::from_le_bytes(tmp)
}

fn write_uint_le(buf: &mut [u8], v: u64) {
    let bytes = v.to_le_bytes();
    buf.copy_from_slice(&bytes[..buf.len()]);
}

fn sign_extend(v: u64, size: usize) -> i64 {
    let bits = size.saturating_mul(8);
    if bits == 0 || bits >= 64 {
        return v as i64;
    }
    let mask = (1u64 << bits) - 1;
    let x = v & mask;
    let sign = 1u64 << (bits - 1);
    if x & sign != 0 {
        (x | !mask) as i64
    } else {
        x as i64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unshuffle_roundtrip_u32() {
        let raw = [0x04u8, 0x03, 0x02, 0x01, 0x08, 0x07, 0x06, 0x05];
        let elem = 4usize;
        let n = raw.len() / elem;
        let mut shuffled = [0u8; 8];
        for i in 0..elem {
            for j in 0..n {
                shuffled[i * n + j] = raw[j * elem + i];
            }
        }
        let back = unshuffle(&shuffled, &[4]).expect("unshuffle");
        assert_eq!(back, raw);
    }

    #[test]
    fn fletcher32_strips_matching_checksum() {
        let payload = b"abcd";
        let sum = fletcher32(payload);
        for enc in [sum.to_be_bytes(), sum.to_le_bytes()] {
            let mut buf = payload.to_vec();
            buf.extend_from_slice(&enc);
            let got = check_fletcher32(&buf).expect("checksum");
            assert_eq!(got, payload);
        }
    }

    #[test]
    fn szip_is_refused() {
        let f = FilterDesc {
            id: HDF5_FILTER_SZIP,
            client_data: alloc::vec![0xa9, 8, 32, 16],
        };
        let err = apply_filters(alloc::vec![1, 2, 3, 4], &[f], 0, 4).unwrap_err();
        assert_eq!(err, HDF5Error::FilteredNotSupported);
    }
}
