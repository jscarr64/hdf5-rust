# Provenance scan, 2026-10-06

Report only. No source changes, no crates.io publish. This file is the only addition.

Scope: three public crates by Jeffrey S. Carr, scanned for recognizable reproductions of someone else's licensed code whose copyright notice did not travel with the chunk.

| Repo | Commit scanned | Crate version |
| --- | --- | --- |
| [jscarr64/hdf5-rust](https://github.com/jscarr64/hdf5-rust) | `1116f82` | hdf5-rust 1.1.1 |
| [jscarr64/LaTeX-Rust](https://github.com/jscarr64/LaTeX-Rust) | `a842d06` | latex-rust 2.1.0 |
| [jscarr64/accu_book](https://github.com/jscarr64/accu_book) | `7985b2e` | accu_book 1.0.0 |

Categories used below:

- **(a)** verbatim or renamed copy of a specific work.
- **(b)** close line-by-line translation of a specific implementation.
- **(c)** independent implementation of a published spec or algorithm. No notice is required. A citation is courtesy.

Only (a) and (b) need the upstream notice. Public-domain source is called out separately: a translation of it is still (b), and the notice is not required.

## Method

Compared in this scan:

- HDF5 C library, `develop` at `9a03597aa6a9` (2026-10-05): `H5checksum.c`, `H5Znbit.c`, `H5Zscaleoffset.c`, `H5Zfletcher32.c`, `H5Zshuffle.c`, `H5Zdeflate.c`, `H5B2cache.c`, `H5B2int.c`, `H5EAcache.c`, `H5FAcache.c`. License text from `LICENSE` in that tree (3-clause BSD).
- Mark Adler `puff.c` / `puff.h` 2.3 (21 Jan 2013) from zlib `develop` `767c4c947852`. zlib license.
- `miniz` `tinfl.c` (master raw).
- Bob Jenkins `lookup3.c` (May 2006), public domain.
- jhdf filter sources (master raw): `ByteShuffleFilter.java`, `FletcherChecksumFilter.java`, `DeflatePipelineFilter.java`, `FilterPipeline.java`. jhdf has no n-bit or scale-offset filter.
- pyfive `569d5ac` (`pyfive/*.py`).
- KaTeX 0.19.0 `6ea2dc9` (`src/`).
- MathJax-src `fb98717`, sparse `ts/input/tex`.
- pulldown-latex `1067fd2`.
- latex2mathml (roniemartinez) default branch.
- unicode-math `184a23b`, especially `unicode-math-table.tex`.
- LaTeX graphics `drivers.dtx` `dvipsnames` guard, fetched 2026-10-06 from `https://ctan.uib.no/macros/latex-dev/required/graphics/drivers.dtx`. LPPL-1.3 or later.
- Jupyter nbformat v4 schema JSON, plus string search of accu_book for nbformat / marimo / Pluto identifiers.

Automated pass: comments and string literals stripped, then identifier/number tokens, dropping language keywords. 8-token shingles indexed across the corpus (winnowing-style). Runs of matching shingles were then checked with `difflib.SequenceMatcher` on the same tokens, and the hits that survived were read side by side. `jscpd` and `dolos` were not installed; this is the substitute.

The longest automated hit inside `inflate.rs` is the RFC 1951 length and distance base tables (31 tokens). Those same tables also match `miniz` `tinfl.c`. That overlap is the spec, category (c), and is not listed as a finding.

No `FOURTY`-style copied typo, and no copied comment block of the zenith-float / astro-float kind, turned up in these three trees.

## Findings that need a notice

### hdf5-rust

Ranked by how close the translation is and how much code it covers.

| Rank | Location | Suspected source | License | Cat. | Confidence |
| --- | --- | --- | --- | --- | --- |
| 1 | `src/filter.rs` `nbit_one_atomic` / `nbit_one_byte` (about lines 306–392) | HDF5 `src/H5Znbit.c` `H5Z__nbit_decompress_one_atomic` and `H5Z__nbit_decompress_one_byte` (about lines 1033–1169), develop `9a03597aa6a9`. https://github.com/HDFGroup/hdf5/blob/9a03597aa6a9/src/H5Znbit.c | HDF5 3-clause BSD. Copyright 2006 The HDF Group; Copyright 1998–2006 University of Illinois. https://github.com/HDFGroup/hdf5/blob/develop/LICENSE | (b) | High |
| 2 | `src/filter.rs` `so_one_atomic` / `so_one_byte` (about lines 538–598), little-endian branch only | HDF5 `src/H5Zscaleoffset.c` `H5Z__scaleoffset_decompress_one_atomic` / `_one_byte` (about lines 1615–1694), same commit | Same HDF5 3-clause BSD | (b) | High |
| 3 | `src/filter.rs` `fletcher32` (188–215), `reversed_fletcher` (237–242), and the 1.6.3 acceptance in `check_fletcher32` (217–233) | `H5_checksum_fletcher32` in `src/H5checksum.c` (about lines 95–136) and the pre-1.6.3 pair-swap in `src/H5Zfletcher32.c` (about lines 88–110) | Same HDF5 3-clause BSD | (b) | High |
| 4 | `src/inflate.rs` `Huff::decode` (about lines 165–187) | `puff.c` 2.3 slow `decode()` (about lines 235–254). https://github.com/madler/zlib/blob/767c4c947852/contrib/puff/puff.c | zlib license, Copyright (C) 2002–2013 Mark Adler (`puff.h`) | (b) | Medium |

`filter.rs` does not carry an HDF Group or University of Illinois copyright notice. `inflate.rs` does not carry the zlib / Mark Adler notice.

#### 1. N-bit decompress

HDF5, little-endian index arithmetic and the per-byte bit copy:

```c
if ((p->precision + p->offset) % 8 != 0)
    begin_i = (p->precision + p->offset) / 8;
else
    begin_i = (p->precision + p->offset) / 8 - 1;
end_i = p->offset / 8;
/* ... */
if (k == begin_i)
    dat_len = 8 - (datatype_len - p->precision - p->offset) % 8;
else if (k == end_i) {
    dat_len    = 8 - p->offset % 8;
    dat_offset = 8 - dat_len;
}
if (*buf_len > dat_len) {
    data[data_offset + k] =
        (unsigned char)(((unsigned)(val >> (*buf_len - dat_len)) &
                         ~((unsigned)(~0) << dat_len)) << dat_offset);
    *buf_len -= dat_len;
} else {
    data[data_offset + k] =
        (unsigned char)(((val & ~((unsigned)(~0) << *buf_len))
                         << (dat_len - *buf_len)) << dat_offset);
    dat_len -= *buf_len;
    H5Z__nbit_next_byte(j, buf_len);
    /* then the remainder from the next input byte */
}
```

hdf5-rust (`src/filter.rs`), same branches, same formulas, bounds checks added:

```rust
if (spec.precision + spec.offset) % 8 != 0 {
    (spec.precision + spec.offset) / 8
} else {
    (spec.precision + spec.offset) / 8 - 1
};
// end_i = spec.offset / 8
let (dat_len, dat_offset) = if begin_i != end_i {
    if k == begin_i {
        (8 - (spec.size as u32 * 8 - spec.precision - spec.offset) % 8, 0u32)
    } else if k == end_i {
        let dat_len = 8 - spec.offset % 8;
        (dat_len, 8 - dat_len)
    } else {
        (8u32, 0u32)
    }
} else {
    (spec.precision, spec.offset % 8)
};
if bits.buf_len > dat_len {
    data[at] = (((u32::from(val) >> (bits.buf_len - dat_len)) & mask_bits(dat_len))
        << dat_offset) as u8;
    bits.buf_len -= dat_len;
} else {
    data[at] = ((u32::from(val & low_mask_u8(bits.buf_len)) << (dat_len - bits.buf_len))
        << dat_offset) as u8;
    let left = dat_len - bits.buf_len;
    bits.bump();
    // remainder taken from the next input byte
}
```

The big-endian `begin_i` / `end_i` pair in `nbit_one_atomic` matches the HDF5 `H5Z_NBIT_ORDER_BE` arm the same way, including the `offset % 8 != 0` adjustment. An 8-gram index under-reports this one: the added `InvalidHeader` checks split tokens, so the longest shared run is 5 tokens. The side-by-side match is the evidence.

Compound, array, and no-op n-bit types are refused (`FilteredNotSupported`). Only the atomic path was translated.

#### 2. Integer scale-offset unpack

HDF5 little-endian arm:

```c
begin_i = p.size - 1 - (dtype_len - p.minbits) / 8;
if (k == begin_i)
    bits_to_copy = 8 - (dtype_len - p.minbits) % 8;
else
    bits_to_copy = 8;
```

hdf5-rust:

```rust
let begin_i = size as u32 - 1 - (dtype_len - minbits) / 8;
let dat_len = if on_first {
    8 - (dtype_len - minbits) % 8
} else {
    8
};
```

`on_first` is `k == begin_i` written against the output index. The following bit-copy (`buf_len > dat_len`, else consume the rest of the byte and continue in the next byte) is the same walker as n-bit and as `H5Z__scaleoffset_decompress_one_byte`. The file comment says integer scale-offset is decoded as a little-endian writer would have packed it. The big-endian arm of the HDF5 function is absent. Header parsing (`minbits`, `minval`, fill sentinel) follows the on-disk filter and is closer to the format description; the bit walker is the implementation copy.

#### 3. Fletcher32

HDF5:

```c
while (len) {
    size_t tlen = len > 360 ? 360 : len;
    len -= tlen;
    do {
        sum1 += (uint32_t)(((uint16_t)data[0]) << 8) | ((uint16_t)data[1]);
        data += 2;
        sum2 += sum1;
    } while (--tlen);
    sum1 = (sum1 & 0xffff) + (sum1 >> 16);
    sum2 = (sum2 & 0xffff) + (sum2 >> 16);
}
if (_len % 2) {
    sum1 += (uint32_t)(((uint16_t)*data) << 8);
    /* fold both sums */
}
sum1 = (sum1 & 0xffff) + (sum1 >> 16);
sum2 = (sum2 & 0xffff) + (sum2 >> 16);
return (sum2 << 16) | sum1;
```

hdf5-rust `fletcher32` is that routine with `words.min(360)`, the odd-byte `<< 8` add, and the second reduction. Token overlap on the reduction alone is 21 tokens (ratio 0.63 on the function). The rust comment names `H5_checksum_fletcher32`. The constant 360 is the HDF5 block size ("largest number of sums that can be performed without numeric overflow" in the C comment). Wikipedia's Fletcher page has carried a similar block algorithm; the odd-byte path plus the second reduction plus the name in the comment tie this copy to the HDF5 function.

`reversed_fletcher` swaps bytes 0/1 and 2/3 of the native word. `H5Zfletcher32.c` does the same swaps and documents them as the checksum bug before Release 1.6.3. The rust comment says "the pair-swapped value written before 1.6.3." That is a paraphrase of the HDF5 comment, and the swap is the same.

#### 4. Inflate decode loop (medium)

`puff.c` slow decode, the readable form Adler left next to the fast one:

```c
code = first = index = 0;
for (len = 1; len <= MAXBITS; len++) {
    code |= bits(s, 1);
    count = h->count[len];
    if (code - count < first)
        return h->symbol[index + (code - first)];
    index += count;
    first += count;
    first <<= 1;
    code <<= 1;
}
```

`Huff::decode` uses the same names (`code`, `first`, `index`, `count`) and the same update order. The comparison is rewritten as `code < first + count` with saturating arithmetic, which is the same test when it does not overflow. `Huff::from_lengths` follows the canonical Huffman construction in RFC 1951; its names (`counts`, `symbols`, `next`, `acc`) do not follow `puff`'s `construct()`, so that function stays (c). Stored-block and dynamic-code parsing follow RFC 1951 as well. The file header cites RFC 1950 and RFC 1951 and does not mention `puff`.

Confidence is medium because there is no copied comment or typo. The shared pedagogical variable names and the `first <<= 1` loop are the evidence. Many tutorials reproduce this loop from `puff`. If this is treated as a derivative, the zlib notice in `puff.h` has to stay with the source: altered versions must be marked, and the notice may not be removed.

#### Remedy (hdf5-rust)

Add a `NOTICE` (or a header on `src/filter.rs`) with the HDF5 3-clause BSD copyright block: The HDF Group (2006) and the University of Illinois (1998–2006), the three conditions, and the disclaimer. Name the functions it covers: atomic n-bit decompress, little-endian integer scale-offset unpack, Fletcher32 including the pre-1.6.3 pair swap.

For `Huff::decode`, add the zlib / Mark Adler notice from `puff.h` and mark the function as derived from `puff.c` 2.3, or rewrite that loop from RFC 1951 in different structure so it is no longer that translation. The length and distance tables can stay; they are the RFC.

`cargo package --list` already ships `LICENSE-MIT` and `LICENSE-APACHE`. A root `NOTICE` would ship too. There is no `include` whitelist to update.

### LaTeX-Rust

| Rank | Location | Suspected source | License | Cat. | Confidence |
| --- | --- | --- | --- | --- | --- |
| 1 | `data/dvipsnames.tsv` (68 data rows), loaded by `src/color.rs` | `dvipsnam.def`, the `%<*dvipsnames>` guard of the LaTeX graphics bundle `drivers.dtx`. Copyright (C) 1994 David Carlisle and Sebastian Rahtz; 1995–1999 David Carlisle; 2000–2026 LaTeX Project. LPPL-1.3 or later | LPPL-1.3c (file says "either version 1.3 or, at your option, any later version") | (a) | High |

All 68 dvips names are present. The TSV is alphabetical; `dvipsnam.def` is in hue order. 67 of 68 CMYK tuples match exactly, including the two-decimal spellings `0.40`, `0.10`, and `0.50` rather than `0.4` / `0.1` / `0.5`.

```tex
\DefineNamedColor{named}{Apricot}       {cmyk}{0,0.32,0.52,0}
\DefineNamedColor{named}{DarkOrchid}    {cmyk}{0.40,0.80,0.20,0}
\DefineNamedColor{named}{Goldenrod}     {cmyk}{0,0.10,0.84,0}
```

```text
Apricot	0	0.32	0.52	0
DarkOrchid	0.40	0.80	0.20	0
Goldenrod	0	0.10	0.84	0
```

The one mismatch is TealBlue: the TSV has `0.86, 0.09, 0.54, 0.10`; `drivers.dtx` has `0.86, 0, 0.34, 0.02`. One altered row does not make the other 67 an independent palette.

`data/dvipsnames.tsv` and `src/color.rs` have no LPPL notice, no LaTeX Project copyright, and no pointer at `dvipsnam.def`. The crate's `LICENSE-MIT` / `LICENSE-APACHE` do not replace that notice. A short factual list is a weaker copyright claim than source code (a compilation of names and numbers). The preserved spelling and the closed set of 68 names are why this is still listed as (a).

#### Remedy (LaTeX-Rust)

Put an LPPL header on `data/dvipsnames.tsv` naming `dvipsnam.def` / `drivers.dtx`, the copyright holders above, and LPPL-1.3 or later, and say the table was re-sorted and that TealBlue was changed. LPPL wants modified files not presented as the original. Alternatively, drop the file and ship a palette this project defines itself.

`Cargo.toml` `include` already lists `data/**`, `LICENSE-MIT`, `LICENSE-APACHE`, and `NOTICE`. A header inside the TSV ships with the crate. The existing `NOTICE` covers STIX Two Math only.

### accu_book

No (a) or (b) finding.

The on-disk format is a private line format (`accu 1`, `CELL`, `SOURCE <byte_len>`, `ENDCELL`), not nbformat JSON. Source search found no `nbformat`, `cell_type`, `execution_count`, `marimo`, or `pluto`. `blake3` is the crates.io crate (`use blake3::Hasher`), not a pasted implementation. "Jupyter" appears as a product contrast in the README and in one comment ("anti Jupyter-ghost"), not as copied notebook code. egui is mentioned as a future UI and is not a dependency of this crate.

## License files and what `cargo package` ships

GitHub's NOASSERTION on hdf5-rust and LaTeX-Rust matches a dual-license tree that has `LICENSE-MIT` and `LICENSE-APACHE` and no single `LICENSE` file. Both texts are the standard MIT and Apache-2.0 licenses. `Cargo.toml` says `license = "MIT OR Apache-2.0"` in all three crates. That SPDX expression is fine for crates.io. GitHub license detection often stops at NOASSERTION for this layout. Adding a `LICENSE` file is optional and was not done here.

| Repo | `license` field | Notice file | `cargo package --list` (this scan) |
| --- | --- | --- | --- |
| hdf5-rust 1.1.1 | `MIT OR Apache-2.0` | none | Ships `LICENSE-MIT` and `LICENSE-APACHE`. No `include` key, so a future root `NOTICE` would ship. |
| latex-rust 2.1.0 | `MIT OR Apache-2.0` | `NOTICE` names STIX Two Math, SIL OFL 1.1, and the STIX trademark | Ships `LICENSE-MIT`, `LICENSE-APACHE`, `NOTICE`, `fonts/stix-two-math/OFL.txt` (full OFL 1.1, 108 lines), and the OTF. `include` lists those paths. |
| accu_book 1.0.0 | `MIT OR Apache-2.0` | none | Ships `LICENSE-MIT` and `LICENSE-APACHE`. Copyright line is "Jeff Carr / McGee Resources". |

`cargo-deny` and `cargo-license` are not installed in this environment. Licenses below are the `license` field crates.io reports for the resolved version (`Cargo.lock` where one is committed; direct pins otherwise).

hdf5-rust 1.1.1 has no dependencies (`Cargo.lock` is only the package itself).

latex-rust direct dependencies (no `Cargo.lock` in the repo; versions are the ones crates.io currently resolves inside the `Cargo.toml` ranges used here):

| Crate | Version queried | License |
| --- | --- | --- |
| ttf-parser | 0.25.1 | MIT OR Apache-2.0 |
| tiny-skia | 0.11.4 | BSD-3-Clause |
| egui | 0.28.1 | MIT OR Apache-2.0 |

tiny-skia's BSD notice lives in that crate. latex-rust does not vendor its source. Transitive licenses were not frozen, because latex-rust ships no lockfile.

accu_book `Cargo.lock` (direct and transitive), noteworthy rows:

| Crate | Version | License |
| --- | --- | --- |
| blake3 | 1.8.7 | CC0-1.0 OR Apache-2.0 OR Apache-2.0 WITH LLVM-exception |
| uuid | 1.26.1 | Apache-2.0 OR MIT |
| serde / serde_json | 1.0.229 / 1.0.151 | MIT OR Apache-2.0 |
| thiserror | 2.0.20 | MIT OR Apache-2.0 |
| unicode-ident | 1.0.26 | (MIT OR Apache-2.0) AND Unicode-3.0 |
| r-efi | 6.0.0 | MIT OR Apache-2.0 OR LGPL-2.1-or-later |

`r-efi` offers LGPL-2.1-or-later as one alternative. Choosing MIT or Apache-2.0 from that OR-expression does not pull LGPL onto accu_book. `unicode-ident` adds the Unicode-3.0 terms; those travel inside the dependency crate. Nothing else in the lockfile is copyleft-only.

## Checked, and no notice is required

### hdf5-rust

| Location | Why it is (c), or why the notice is already satisfied |
| --- | --- |
| `src/checksum.rs` `lookup3` / `mix` / `final_mix` | Close translation of Bob Jenkins `lookup3.c` (May 2006), the `hashlittle` byte path: `0xdeadbeef`, mix rotates 4, 6, 8, 16, 19, 4, final rotates 14, 11, 25, 16, 4, 14, 24. Jenkins's header says the file is public domain and may be used for any purpose. The rust file already names `hashlittle` and Jenkins. HDF5 vendors the same macros inside `H5checksum.c` and keeps Jenkins's grant paragraph; the rust code follows the public-domain reference. No notice is required. |
| `src/filter.rs` `unshuffle` | The obvious inverse of the HDF5 shuffle (`out[j * elem + i] = data[i * n + j]`). `H5Zshuffle.c` is a larger, different loop. Category (c). |
| `src/btree.rs`, `src/btree2.rs`, `src/earray.rs`, `src/chunk_index.rs`, `src/messages.rs`, `src/superblock.rs`, `src/ohdr.rs`, `src/decode.rs` | No 8-token run against `H5B2int.c`, `H5EAcache.c`, `H5FAcache.c`, or pyfive. These files follow the HDF5 file-format layout (signatures, version bytes, checksummed headers). Category (c). A citation of the HDF5 file-format spec would be courtesy. The README already says this is an independent implementation and is not affiliated with The HDF Group. |
| `src/inflate.rs` length/distance tables and `from_lengths` | RFC 1951. The tables also match `puff.c` and `miniz`. Category (c). |
| h5py | Used to generate fixtures (`scripts/gen_h5py_fixtures.py`). Not copied into the crate. |
| `hdf5` / `hdf5-metno` | libhdf5 bindings, not a pure-Rust format implementation. Not cloned. No FFI wrapper of those crates is present. |
| netCDF | Not cloned. No netCDF API or error-string overlap showed up in the hdf5-rust sources that were read. |

### LaTeX-Rust

8-token shingles of `src/**/*.rs` against KaTeX 0.19.0, MathJax `ts/input/tex`, pulldown-latex, latex2mathml, and unicode-math produced no run long enough to record. That includes `src/layout/engine.rs` (3661 lines), `src/parser/parse.rs`, and `src/render/egui/tessellate.rs`.

| Location | Why it is (c), or why the notice is already satisfied |
| --- | --- |
| `fonts/stix-two-math/` | STIX Two Math is embedded. `OFL.txt` is the full SIL OFL 1.1 with the STIX Fonts Project copyright (2001–2021) and the reserved-name / trademark language. `NOTICE` points at it. `Cargo.toml` `include` ships both. This is the case that was done correctly. The font bytes were not re-hashed against the upstream STIX 2.13 tarball. |
| `src/style_map.rs` | Mathematical Alphanumeric Symbols, Unicode block U+1D400. Bases `0x1D400`, `0x1D434`, `0x1D468`, and the Letterlike holes (`ℎ` = U+210E, `ℂ` = U+2102, `ℒ` = U+2112, `ℭ` = U+212D, and the rest) are the Unicode chart. KaTeX `wide-character.ts` maps the other direction (code point to a KaTeX font) and cites `https://www.unicode.org/charts/PDF/U1D400.pdf`. It is not this file. |
| `src/layout/space.rs` | TeXbook Table 18, with negative entries meaning "omit in script style." KaTeX `spacingData.ts` uses separate `spacings` and `tightSpacings` objects and does not use this signed matrix. |
| `src/atoms.rs` | TeX atom classes for named commands (TeXbook Table 18). Command lists overlap KaTeX and MathJax because the command names are LaTeX's. No shared description strings or copied comments. |
| `data/symbols.tsv` | 476 rows. Descriptions are short glosses ("Pitchfork", "Normal subgroup", "Greek Alphabet (Lowercase/Uppercase)"). They do not match unicode-math's description field (`pitchfork`, not a repeated "Greek Alphabet..." boilerplate) and they did not match KaTeX or MathJax sources. Command-to-glyph pairs are the ordinary LaTeX mapping. Category (c). |
| `src/hash.rs` | SHA-256, FIPS 180-4, including the standard `K` and `H0` constants. Category (c). |
| `src/layout/metrics.rs` | Reads OpenType MATH constants from the face through `ttf-parser`. Field names follow the MATH table and TeX σ-numbers. Category (c). |
| Typst, STIX glyph tables beyond the font file, Latin Modern | Typst was not cloned (large). No Typst-specific identifier or comment showed up. Latin Modern is not embedded. |

### accu_book

`src/*.rs` was searched for attribution comments and for Jupyter / marimo / Pluto / nbformat vocabulary. The notebook model (cells, DAG, job queue, `.accu` text, Blake3 content hashes) does not reproduce those codebases. marimo and Pluto.jl were not fully cloned; see below.

## What this scan did not verify

- The rest of the HDF5 C tree beyond the files listed under Method. A later filter or a copied comment could still sit in a file that was only sampled (`messages.rs`, `btree2.rs`, `earray.rs` were read in part and did not match the indexed C files).
- `hdf5` / `hdf5-metno` Rust sources, and netCDF sources. Both were treated as out of scope once it was clear hdf5-rust does not wrap them. They were not diffed.
- Typst, a full MathJax checkout (only `ts/input/tex`), marimo, and Pluto.jl. accu_book's negative result is a string and format check, not an n-gram against those trees.
- The second source of the TealBlue tuple `0.86, 0.09, 0.54, 0.10`. It is not the `drivers.dtx` value.
- Byte identity of `STIXTwoMath-Regular.otf` with the upstream STIX Two Math 2.13 release. The OFL notice is present either way.
- `cargo deny check licenses`. The tool is not installed. Dependency licenses were read from crates.io for the versions above.
- latex-rust's transitive dependency closure. There is no `Cargo.lock` in that repo.
- Whether a court would treat the 68-color TSV as a copyrightable compilation. It is listed because the values and the spelling match `dvipsnam.def`.
- Legal advice. This is an evidence report.
