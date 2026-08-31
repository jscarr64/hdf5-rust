//! Little-endian file cursor. Addresses and lengths follow the superblock sizes.

use crate::error::{HDF5Error, Result};
use crate::HDF5_UNDEF_ADDR_8;

/// Read cursor over a borrowed file image.
#[derive(Clone, Copy)]
pub struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
    offset_size: u8,
    length_size: u8,
}

impl<'a> Reader<'a> {
    pub fn new(data: &'a [u8], offset_size: u8, length_size: u8) -> Result<Self> {
        if offset_size != 4 && offset_size != 8 {
            return Err(HDF5Error::UnsupportedVersion(offset_size));
        }
        if length_size != 4 && length_size != 8 {
            return Err(HDF5Error::UnsupportedVersion(length_size));
        }
        Ok(Self {
            data,
            pos: 0,
            offset_size,
            length_size,
        })
    }

    pub fn at(self, pos: u64) -> Result<Self> {
        let p = usize::try_from(pos).map_err(|_| HDF5Error::Truncated)?;
        if p > self.data.len() {
            return Err(HDF5Error::Truncated);
        }
        Ok(Self { pos: p, ..self })
    }

    pub fn pos(&self) -> usize {
        self.pos
    }

    pub fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    pub fn u8(&mut self) -> Result<u8> {
        if self.pos >= self.data.len() {
            return Err(HDF5Error::Truncated);
        }
        let v = self.data[self.pos];
        self.pos += 1;
        Ok(v)
    }

    pub fn u16(&mut self) -> Result<u16> {
        let b = self.bytes(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    pub fn u32(&mut self) -> Result<u32> {
        let b = self.bytes(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    pub fn u64(&mut self) -> Result<u64> {
        let b = self.bytes(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    pub fn bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|e| *e <= self.data.len())
            .ok_or(HDF5Error::Truncated)?;
        let s = &self.data[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    pub fn skip(&mut self, n: usize) -> Result<()> {
        let _ = self.bytes(n)?;
        Ok(())
    }

    pub fn sized_uint(&mut self, nbytes: u8) -> Result<u64> {
        match nbytes {
            1 => Ok(u64::from(self.u8()?)),
            2 => Ok(u64::from(self.u16()?)),
            4 => Ok(u64::from(self.u32()?)),
            8 => self.u64(),
            _ => Err(HDF5Error::InvalidHeader),
        }
    }

    pub fn addr(&mut self) -> Result<u64> {
        self.sized_uint(self.offset_size)
    }

    pub fn length(&mut self) -> Result<u64> {
        self.sized_uint(self.length_size)
    }

    pub fn is_undef(self, addr: u64) -> bool {
        match self.offset_size {
            4 => addr == u64::from(u32::MAX),
            _ => addr == HDF5_UNDEF_ADDR_8,
        }
    }

    pub fn slice_at(&self, addr: u64, n: u64) -> Result<&'a [u8]> {
        let start = usize::try_from(addr).map_err(|_| HDF5Error::Truncated)?;
        let len = usize::try_from(n).map_err(|_| HDF5Error::Truncated)?;
        let end = start.checked_add(len).ok_or(HDF5Error::Truncated)?;
        if end > self.data.len() {
            return Err(HDF5Error::Truncated);
        }
        Ok(&self.data[start..end])
    }
}

pub fn push_u16(buf: &mut alloc::vec::Vec<u8>, v: u16) {
    buf.extend_from_slice(&v.to_le_bytes());
}

pub fn push_u32(buf: &mut alloc::vec::Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

pub fn push_u64(buf: &mut alloc::vec::Vec<u8>, v: u64) {
    buf.extend_from_slice(&v.to_le_bytes());
}

pub fn align8(n: u64) -> u64 {
    n.wrapping_add(7) & !7
}
