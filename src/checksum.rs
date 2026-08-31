//! Jenkins lookup3 (`hashlittle`), public-domain algorithm.
//! HDF5 checksums every superblock v2 and object-header v2 chunk with initval 0.
//! Byte-at-a-time path only, matching Bob Jenkins’ reference `hashlittle`.

/// Mix three 32-bit words. Matches Jenkins’ `mix` macro.
#[inline]
fn mix(a: &mut u32, b: &mut u32, c: &mut u32) {
    *a = a.wrapping_sub(*c);
    *a ^= c.rotate_left(4);
    *c = c.wrapping_add(*b);
    *b = b.wrapping_sub(*a);
    *b ^= a.rotate_left(6);
    *a = a.wrapping_add(*c);
    *c = c.wrapping_sub(*b);
    *c ^= b.rotate_left(8);
    *b = b.wrapping_add(*a);
    *a = a.wrapping_sub(*c);
    *a ^= c.rotate_left(16);
    *c = c.wrapping_add(*b);
    *b = b.wrapping_sub(*a);
    *b ^= a.rotate_left(19);
    *a = a.wrapping_add(*c);
    *c = c.wrapping_sub(*b);
    *c ^= b.rotate_left(4);
    *b = b.wrapping_add(*a);
}

/// Final mix. Matches Jenkins’ `final` macro.
#[inline]
fn final_mix(a: &mut u32, b: &mut u32, c: &mut u32) {
    *c ^= *b;
    *c = c.wrapping_sub(b.rotate_left(14));
    *a ^= *c;
    *a = a.wrapping_sub(c.rotate_left(11));
    *b ^= *a;
    *b = b.wrapping_sub(a.rotate_left(25));
    *c ^= *b;
    *c = c.wrapping_sub(b.rotate_left(16));
    *a ^= *c;
    *a = a.wrapping_sub(c.rotate_left(4));
    *b ^= *a;
    *b = b.wrapping_sub(a.rotate_left(14));
    *c ^= *b;
    *c = c.wrapping_sub(b.rotate_left(24));
}

fn pack4(k: &[u8], i: usize) -> u32 {
    u32::from(k[i])
        | (u32::from(k[i + 1]) << 8)
        | (u32::from(k[i + 2]) << 16)
        | (u32::from(k[i + 3]) << 24)
}

/// Jenkins lookup3 over `data`. HDF5 uses `initval = 0`.
///
/// Zero-length input returns `0xdeadbeef + initval` with no mixing.
pub fn lookup3(data: &[u8], initval: u32) -> u32 {
    let mut a = 0xdeadbeefu32
        .wrapping_add(data.len() as u32)
        .wrapping_add(initval);
    let mut b = a;
    let mut c = a;
    let mut k = data;
    while k.len() > 12 {
        a = a.wrapping_add(pack4(k, 0));
        b = b.wrapping_add(pack4(k, 4));
        c = c.wrapping_add(pack4(k, 8));
        mix(&mut a, &mut b, &mut c);
        k = &k[12..];
    }
    // Fall-through switch; length 0 returns without `final`.
    match k.len() {
        0 => return c,
        12 => {
            c = c.wrapping_add(u32::from(k[11]) << 24);
            c = c.wrapping_add(u32::from(k[10]) << 16);
            c = c.wrapping_add(u32::from(k[9]) << 8);
            c = c.wrapping_add(u32::from(k[8]));
            b = b.wrapping_add(pack4(k, 4));
            a = a.wrapping_add(pack4(k, 0));
        }
        11 => {
            c = c.wrapping_add(u32::from(k[10]) << 16);
            c = c.wrapping_add(u32::from(k[9]) << 8);
            c = c.wrapping_add(u32::from(k[8]));
            b = b.wrapping_add(pack4(k, 4));
            a = a.wrapping_add(pack4(k, 0));
        }
        10 => {
            c = c.wrapping_add(u32::from(k[9]) << 8);
            c = c.wrapping_add(u32::from(k[8]));
            b = b.wrapping_add(pack4(k, 4));
            a = a.wrapping_add(pack4(k, 0));
        }
        9 => {
            c = c.wrapping_add(u32::from(k[8]));
            b = b.wrapping_add(pack4(k, 4));
            a = a.wrapping_add(pack4(k, 0));
        }
        8 => {
            b = b.wrapping_add(pack4(k, 4));
            a = a.wrapping_add(pack4(k, 0));
        }
        7 => {
            b = b.wrapping_add(u32::from(k[6]) << 16);
            b = b.wrapping_add(u32::from(k[5]) << 8);
            b = b.wrapping_add(u32::from(k[4]));
            a = a.wrapping_add(pack4(k, 0));
        }
        6 => {
            b = b.wrapping_add(u32::from(k[5]) << 8);
            b = b.wrapping_add(u32::from(k[4]));
            a = a.wrapping_add(pack4(k, 0));
        }
        5 => {
            b = b.wrapping_add(u32::from(k[4]));
            a = a.wrapping_add(pack4(k, 0));
        }
        4 => {
            a = a.wrapping_add(pack4(k, 0));
        }
        3 => {
            a = a.wrapping_add(u32::from(k[2]) << 16);
            a = a.wrapping_add(u32::from(k[1]) << 8);
            a = a.wrapping_add(u32::from(k[0]));
        }
        2 => {
            a = a.wrapping_add(u32::from(k[1]) << 8);
            a = a.wrapping_add(u32::from(k[0]));
        }
        1 => {
            a = a.wrapping_add(u32::from(k[0]));
        }
        _ => return c,
    }
    final_mix(&mut a, &mut b, &mut c);
    c
}

#[cfg(test)]
mod tests {
    use super::lookup3;

    #[test]
    fn empty_is_deadbeef() {
        assert_eq!(lookup3(b"", 0), 0xdeadbeef);
        assert_eq!(lookup3(b"", 1), 0xdeadbeefu32.wrapping_add(1));
    }

    #[test]
    fn deterministic() {
        let d = b"Four score and seven years ago";
        assert_eq!(lookup3(d, 0), lookup3(d, 0));
        assert_ne!(lookup3(d, 0), lookup3(d, 1));
    }
}
