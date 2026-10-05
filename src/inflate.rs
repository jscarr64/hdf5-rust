//! Pure-Rust zlib inflate (RFC 1950 wrapping DEFLATE RFC 1951).
//! No crate dependencies. Errors are returned; never panics on bad input.

use alloc::vec::Vec;

use crate::error::{HDF5Error, Result};

const WINDOW: usize = 32768;

/// Inflate a zlib-wrapped DEFLATE stream into `expected_len` bytes when known,
/// or until the stream ends when `expected_len` is `None`.
pub fn inflate_zlib(input: &[u8], expected_len: Option<usize>) -> Result<Vec<u8>> {
    if input.len() < 2 {
        return Err(HDF5Error::InvalidHeader);
    }
    let cmf = input[0];
    let flg = input[1];
    if cmf & 0x0F != 8 {
        return Err(HDF5Error::FilteredNotSupported);
    }
    if (u16::from(cmf) * 256 + u16::from(flg)) % 31 != 0 {
        return Err(HDF5Error::InvalidHeader);
    }
    let pos = 2usize;
    if flg & 0x20 != 0 {
        // FDICT — dictionary present; HDF5 deflate does not use this.
        if input.len() < pos + 4 {
            return Err(HDF5Error::Truncated);
        }
        return Err(HDF5Error::FilteredNotSupported);
    }
    let mut br = BitReader {
        data: input,
        pos,
        bitbuf: 0,
        bitcnt: 0,
    };
    let mut out = match expected_len {
        Some(n) => Vec::with_capacity(n),
        None => Vec::new(),
    };
    let mut window = [0u8; WINDOW];
    let mut wpos: usize = 0;
    loop {
        let bfinal = br.bits(1)?;
        let btype = br.bits(2)?;
        match btype {
            0 => inflate_stored(&mut br, &mut out, &mut window, &mut wpos, expected_len)?,
            1 => {
                let (ll, d) = fixed_tables()?;
                inflate_codes(&mut br, &mut out, &mut window, &mut wpos, &ll, &d, expected_len)?;
            }
            2 => {
                let (ll, d) = dynamic_tables(&mut br)?;
                inflate_codes(&mut br, &mut out, &mut window, &mut wpos, &ll, &d, expected_len)?;
            }
            _ => return Err(HDF5Error::InvalidHeader),
        }
        if bfinal == 1 {
            break;
        }
    }
    // Skip Adler-32 (4 bytes) if present; do not invent data when truncated.
    let _ = br;
    if let Some(n) = expected_len {
        if out.len() != n {
            return Err(HDF5Error::InvalidHeader);
        }
    }
    Ok(out)
}

struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
    bitbuf: u32,
    bitcnt: u32,
}

impl<'a> BitReader<'a> {
    fn bits(&mut self, n: u32) -> Result<u32> {
        while self.bitcnt < n {
            if self.pos >= self.data.len() {
                return Err(HDF5Error::Truncated);
            }
            self.bitbuf |= u32::from(self.data[self.pos]) << self.bitcnt;
            self.pos += 1;
            self.bitcnt += 8;
        }
        let v = self.bitbuf & ((1u32 << n) - 1);
        self.bitbuf >>= n;
        self.bitcnt -= n;
        Ok(v)
    }

    fn align_byte(&mut self) {
        self.bitbuf = 0;
        self.bitcnt = 0;
    }

    fn bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        self.align_byte();
        if self.pos.saturating_add(n) > self.data.len() {
            return Err(HDF5Error::Truncated);
        }
        let s = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
}

struct Huff {
    /// For each bit-length group, decode via counts/symbols (canonical).
    counts: [u16; 16],
    symbols: Vec<u16>,
    maxbits: u32,
}

impl Huff {
    fn from_lengths(lengths: &[u8]) -> Result<Self> {
        let mut counts = [0u16; 16];
        let mut maxbits = 0u32;
        for &l in lengths {
            if l > 15 {
                return Err(HDF5Error::InvalidHeader);
            }
            if l > 0 {
                counts[l as usize] = counts[l as usize].saturating_add(1);
                maxbits = maxbits.max(u32::from(l));
            }
        }
        if maxbits == 0 {
            return Ok(Self {
                counts,
                symbols: Vec::new(),
                maxbits: 0,
            });
        }
        counts[0] = 0;
        let total: usize = counts.iter().map(|&c| c as usize).sum();
        let mut symbols = alloc::vec![0u16; total];
        let mut next = [0u16; 16];
        let mut acc = 0u16;
        for bits in 1..=15 {
            next[bits] = acc;
            acc = acc.saturating_add(counts[bits]);
        }
        for (sym, &l) in lengths.iter().enumerate() {
            if l != 0 {
                let slot = next[l as usize] as usize;
                if slot >= symbols.len() {
                    return Err(HDF5Error::InvalidHeader);
                }
                symbols[slot] = sym as u16;
                next[l as usize] = next[l as usize].saturating_add(1);
            }
        }
        Ok(Self {
            counts,
            symbols,
            maxbits,
        })
    }

    fn decode(&self, br: &mut BitReader<'_>) -> Result<u16> {
        if self.maxbits == 0 {
            return Err(HDF5Error::InvalidHeader);
        }
        let mut code = 0u32;
        let mut first = 0u32;
        let mut index = 0u32;
        for len in 1..=self.maxbits {
            code |= br.bits(1)?;
            let count = u32::from(self.counts[len as usize]);
            if code < first.saturating_add(count) {
                let i = index.saturating_add(code.saturating_sub(first)) as usize;
                if i >= self.symbols.len() {
                    return Err(HDF5Error::InvalidHeader);
                }
                return Ok(self.symbols[i]);
            }
            index = index.saturating_add(count);
            first = first.saturating_add(count).wrapping_mul(2);
            code <<= 1;
        }
        Err(HDF5Error::InvalidHeader)
    }
}

fn inflate_stored(
    br: &mut BitReader<'_>,
    out: &mut Vec<u8>,
    window: &mut [u8; WINDOW],
    wpos: &mut usize,
    expected: Option<usize>,
) -> Result<()> {
    let len_bytes = br.bytes(4)?;
    let len = u16::from_le_bytes([len_bytes[0], len_bytes[1]]);
    let nlen = u16::from_le_bytes([len_bytes[2], len_bytes[3]]);
    if len ^ 0xFFFF != nlen {
        return Err(HDF5Error::InvalidHeader);
    }
    let data = br.bytes(len as usize)?;
    for &b in data {
        push_out(out, window, wpos, b, expected)?;
    }
    Ok(())
}

fn push_out(
    out: &mut Vec<u8>,
    window: &mut [u8; WINDOW],
    wpos: &mut usize,
    b: u8,
    expected: Option<usize>,
) -> Result<()> {
    if let Some(n) = expected {
        if out.len() >= n {
            return Err(HDF5Error::InvalidHeader);
        }
    }
    out.push(b);
    window[*wpos % WINDOW] = b;
    *wpos = wpos.saturating_add(1);
    Ok(())
}

fn inflate_codes(
    br: &mut BitReader<'_>,
    out: &mut Vec<u8>,
    window: &mut [u8; WINDOW],
    wpos: &mut usize,
    lit: &Huff,
    dist: &Huff,
    expected: Option<usize>,
) -> Result<()> {
    const LEN_BASE: [u16; 29] = [
        3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115,
        131, 163, 195, 227, 258,
    ];
    const LEN_EXTRA: [u8; 29] = [
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
    ];
    const DIST_BASE: [u16; 30] = [
        1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
        2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
    ];
    const DIST_EXTRA: [u8; 30] = [
        0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12,
        13, 13,
    ];
    loop {
        let sym = lit.decode(br)?;
        match sym {
            0..=255 => push_out(out, window, wpos, sym as u8, expected)?,
            256 => return Ok(()),
            257..=285 => {
                let idx = (sym - 257) as usize;
                if idx >= LEN_BASE.len() {
                    return Err(HDF5Error::InvalidHeader);
                }
                let mut length = u32::from(LEN_BASE[idx]);
                let extra = LEN_EXTRA[idx];
                if extra > 0 {
                    length = length.saturating_add(br.bits(u32::from(extra))?);
                }
                let dsym = dist.decode(br)? as usize;
                if dsym >= DIST_BASE.len() {
                    return Err(HDF5Error::InvalidHeader);
                }
                let mut distance = u32::from(DIST_BASE[dsym]);
                let dextra = DIST_EXTRA[dsym];
                if dextra > 0 {
                    distance = distance.saturating_add(br.bits(u32::from(dextra))?);
                }
                if distance == 0 || distance as usize > *wpos {
                    return Err(HDF5Error::InvalidHeader);
                }
                for _ in 0..length {
                    let src = wpos.wrapping_sub(distance as usize) % WINDOW;
                    let b = window[src];
                    push_out(out, window, wpos, b, expected)?;
                }
            }
            _ => return Err(HDF5Error::InvalidHeader),
        }
    }
}

fn fixed_tables() -> Result<(Huff, Huff)> {
    let mut lit = [0u8; 288];
    for slot in &mut lit[..144] {
        *slot = 8;
    }
    for slot in &mut lit[144..256] {
        *slot = 9;
    }
    for slot in &mut lit[256..280] {
        *slot = 7;
    }
    for slot in &mut lit[280..] {
        *slot = 8;
    }
    let dist = [5u8; 32];
    Ok((Huff::from_lengths(&lit)?, Huff::from_lengths(&dist)?))
}

fn dynamic_tables(br: &mut BitReader<'_>) -> Result<(Huff, Huff)> {
    const ORDER: [usize; 19] = [
        16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
    ];
    let hlit = br.bits(5)? as usize + 257;
    let hdist = br.bits(5)? as usize + 1;
    let hclen = br.bits(4)? as usize + 4;
    let mut clen = [0u8; 19];
    for i in 0..hclen {
        clen[ORDER[i]] = br.bits(3)? as u8;
    }
    let clen_huff = Huff::from_lengths(&clen)?;
    let total = hlit + hdist;
    let mut lengths = alloc::vec![0u8; total];
    let mut i = 0;
    while i < total {
        let sym = clen_huff.decode(br)?;
        match sym {
            0..=15 => {
                lengths[i] = sym as u8;
                i += 1;
            }
            16 => {
                if i == 0 {
                    return Err(HDF5Error::InvalidHeader);
                }
                let rep = br.bits(2)? as usize + 3;
                let v = lengths[i - 1];
                for _ in 0..rep {
                    if i >= total {
                        return Err(HDF5Error::InvalidHeader);
                    }
                    lengths[i] = v;
                    i += 1;
                }
            }
            17 => {
                let rep = br.bits(3)? as usize + 3;
                for _ in 0..rep {
                    if i >= total {
                        return Err(HDF5Error::InvalidHeader);
                    }
                    lengths[i] = 0;
                    i += 1;
                }
            }
            18 => {
                let rep = br.bits(7)? as usize + 11;
                for _ in 0..rep {
                    if i >= total {
                        return Err(HDF5Error::InvalidHeader);
                    }
                    lengths[i] = 0;
                    i += 1;
                }
            }
            _ => return Err(HDF5Error::InvalidHeader),
        }
    }
    let lit = Huff::from_lengths(&lengths[..hlit])?;
    let dist = Huff::from_lengths(&lengths[hlit..])?;
    Ok((lit, dist))
}

