//! Parse an HDF5 file image into [`FileModel`].

use alloc::string::String;
use alloc::vec::Vec;

use crate::btree::{
    read_gheap_object, read_local_heap, walk_chunk_btree, walk_group_btree, ChunkRef,
};
use crate::buf::Reader;
use crate::chunk_index::{self, ChunkIndex};
use crate::error::{HDF5Error, Result};
use crate::filter::{apply_filters, parse_filters, FilterDesc};
use crate::messages::{
    integer_bits, parse_attribute, parse_dataspace, parse_datatype, parse_layout, parse_link,
    parse_link_info_heap, parse_symbol_table, CompoundField, IntegerBits, ParsedDType,
    ParsedLayout,
};
use crate::model::{DTypeKind, DatasetRec, FileModel};
use crate::ohdr::{parse_ohdr, RawMsg};
use crate::superblock;
use crate::{
    HDF5_MAX_WALK_DEPTH, HDF5_MSG_ATTRIBUTE, HDF5_MSG_DATASPACE, HDF5_MSG_DATATYPE,
    HDF5_MSG_FILTER, HDF5_MSG_LAYOUT, HDF5_MSG_LINK, HDF5_MSG_LINK_INFO, HDF5_MSG_SYMBOL_TABLE,
};

pub fn decode(data: &[u8]) -> Result<FileModel> {
    let sb = superblock::parse(data)?;
    let mut model = FileModel::new();
    walk(&mut WalkCtx {
        data,
        sb: &sb,
        ohdr: sb.root_ohdr,
        path: "",
        model: &mut model,
        depth: 0,
        btree: sb.root_btree,
        heap: sb.root_heap,
    })?;
    Ok(model)
}

struct WalkCtx<'a> {
    data: &'a [u8],
    sb: &'a superblock::Superblock,
    ohdr: u64,
    path: &'a str,
    model: &'a mut FileModel,
    depth: usize,
    btree: Option<u64>,
    heap: Option<u64>,
}

fn walk(ctx: &mut WalkCtx<'_>) -> Result<()> {
    if ctx.depth > HDF5_MAX_WALK_DEPTH {
        return Err(HDF5Error::InvalidHeader);
    }
    let msgs = parse_ohdr(ctx.data, ctx.ohdr, ctx.sb.offset_size, ctx.sb.length_size)?;
    if let Some(ds) = try_dataset(ctx.data, ctx.sb, &msgs)? {
        if !ctx.path.is_empty() {
            ctx.model.put_dataset(ctx.path, ds)?;
        }
        return Ok(());
    }
    if !ctx.path.is_empty() {
        ctx.model.create_group(ctx.path)?;
    }
    let mut kids: Vec<(String, u64)> = Vec::new();
    if let Some(h) = msgs.iter().find(|m| m.ty == HDF5_MSG_LINK_INFO) {
        if parse_link_info_heap(&h.body, ctx.sb.offset_size)?.is_some() {
            return Err(HDF5Error::DenseGroupsNotSupported);
        }
    }
    for m in &msgs {
        if m.ty == HDF5_MSG_LINK {
            let l = parse_link(&m.body, ctx.sb.offset_size)?;
            kids.push((l.name, l.ohdr));
        }
    }
    if kids.is_empty() {
        if let Some(m) = msgs.iter().find(|x| x.ty == HDF5_MSG_SYMBOL_TABLE) {
            let (bt, hp) = parse_symbol_table(&m.body, ctx.sb.offset_size)?;
            let heap_bytes = read_local_heap(ctx.data, hp, ctx.sb.offset_size, ctx.sb.length_size)?;
            for c in walk_group_btree(
                ctx.data,
                bt,
                heap_bytes,
                ctx.sb.offset_size,
                ctx.sb.length_size,
                0,
            )? {
                kids.push((c.name, c.ohdr));
            }
        } else if let (Some(bt), Some(hp)) = (ctx.btree, ctx.heap) {
            let heap_bytes = read_local_heap(ctx.data, hp, ctx.sb.offset_size, ctx.sb.length_size)?;
            for c in walk_group_btree(
                ctx.data,
                bt,
                heap_bytes,
                ctx.sb.offset_size,
                ctx.sb.length_size,
                0,
            )? {
                kids.push((c.name, c.ohdr));
            }
        }
    }
    for (name, child) in kids {
        let child_path = if ctx.path.is_empty() {
            name
        } else {
            alloc::format!("{}/{name}", ctx.path)
        };
        walk(&mut WalkCtx {
            data: ctx.data,
            sb: ctx.sb,
            ohdr: child,
            path: &child_path,
            model: ctx.model,
            depth: ctx.depth + 1,
            btree: None,
            heap: None,
        })?;
    }
    Ok(())
}

fn try_dataset(
    data: &[u8],
    sb: &superblock::Superblock,
    msgs: &[RawMsg],
) -> Result<Option<DatasetRec>> {
    let dtype_m = msgs.iter().find(|m| m.ty == HDF5_MSG_DATATYPE);
    let space_m = msgs.iter().find(|m| m.ty == HDF5_MSG_DATASPACE);
    let layout_m = msgs.iter().find(|m| m.ty == HDF5_MSG_LAYOUT);
    let (Some(dtype_m), Some(space_m), Some(layout_m)) = (dtype_m, space_m, layout_m) else {
        return Ok(None);
    };
    let dtype = parse_datatype(&dtype_m.body)?;
    let space = parse_dataspace(&space_m.body, sb.length_size)?;
    let layout = parse_layout(&layout_m.body, sb.offset_size, sb.length_size)?;
    let mut attrs = Vec::new();
    for m in msgs {
        if m.ty == HDF5_MSG_ATTRIBUTE {
            if let Ok(a) = parse_attribute(&m.body, sb.offset_size, sb.length_size) {
                if let Some(s) = attr_to_string(data, sb, &a.dtype, &a.data) {
                    attrs.push((a.name, s));
                }
            }
        }
    }
    let mut filters: Vec<FilterDesc> = Vec::new();
    for m in msgs {
        if m.ty == HDF5_MSG_FILTER {
            match parse_filters(&m.body) {
                Ok(f) => filters = f,
                Err(_) => {
                    // Malformed pipeline: treat as filtered so reads fail honestly.
                    filters = alloc::vec![FilterDesc {
                        id: 0xffff,
                        client_data: Vec::new(),
                    }];
                }
            }
        }
    }
    let fields = compound_fields_of(&dtype);
    let (kind, needs_swap) = dtype_kind_swap(dtype);
    let bits = integer_bits(&dtype_m.body);
    match layout {
        ParsedLayout::Chunked { chunk_dims, index } => {
            let assembled = assemble_chunks(
                data,
                sb,
                &index,
                &space.dims,
                &chunk_dims,
                kind.elem_size(),
                &filters,
            );
            match assembled {
                Ok(mut raw) => {
                    finish_numeric(&mut raw, needs_swap, kind.elem_size(), bits.as_ref());
                    Ok(Some(DatasetRec {
                        shape: space.dims,
                        kind,
                        data: raw,
                        attrs,
                        chunked: false,
                        filtered: false,
                        compound_fields: fields.clone(),
                    }))
                }
                Err(HDF5Error::FilteredNotSupported) => Ok(Some(DatasetRec {
                    shape: space.dims,
                    kind,
                    data: Vec::new(),
                    attrs,
                    chunked: true,
                    filtered: true,
                    compound_fields: fields.clone(),
                })),
                Err(HDF5Error::ChunkedNotSupported) => Ok(Some(DatasetRec {
                    shape: space.dims,
                    kind,
                    data: Vec::new(),
                    attrs,
                    chunked: true,
                    filtered: !filters.is_empty(),
                    compound_fields: fields.clone(),
                })),
                Err(e) => Err(e),
            }
        }
        ParsedLayout::Compact { data: raw } => {
            let mut raw = raw;
            finish_numeric(&mut raw, needs_swap, kind.elem_size(), bits.as_ref());
            Ok(Some(make_rec(kind, space.dims, raw, attrs, fields.clone())))
        }
        ParsedLayout::Contiguous { addr, size } => {
            let n = if size == 0 {
                elem_count(&space.dims).saturating_mul(kind.elem_size() as u64)
            } else {
                size
            };
            let mut raw = if n == 0 {
                Vec::new()
            } else {
                Reader::new(data, sb.offset_size, sb.length_size)?
                    .slice_at(addr, n)?
                    .to_vec()
            };
            finish_numeric(&mut raw, needs_swap, kind.elem_size(), bits.as_ref());
            Ok(Some(make_rec(kind, space.dims, raw, attrs, fields)))
        }
    }
}

fn dtype_kind_swap(dtype: ParsedDType) -> (DTypeKind, bool) {
    match dtype {
        ParsedDType::Float64Le => (DTypeKind::Float64, false),
        ParsedDType::Float32Le => (DTypeKind::Float32, false),
        ParsedDType::Int8Le => (DTypeKind::Int8, false),
        ParsedDType::Int16Le => (DTypeKind::Int16, false),
        ParsedDType::Int32Le => (DTypeKind::Int32, false),
        ParsedDType::Int64Le => (DTypeKind::Int64, false),
        ParsedDType::UInt8Le => (DTypeKind::UInt8, false),
        ParsedDType::UInt16Le => (DTypeKind::UInt16, false),
        ParsedDType::UInt32Le => (DTypeKind::UInt32, false),
        ParsedDType::UInt64Le => (DTypeKind::UInt64, false),
        ParsedDType::Float64Be => (DTypeKind::Float64, true),
        ParsedDType::Float32Be => (DTypeKind::Float32, true),
        ParsedDType::Int8Be => (DTypeKind::Int8, true),
        ParsedDType::Int16Be => (DTypeKind::Int16, true),
        ParsedDType::Int32Be => (DTypeKind::Int32, true),
        ParsedDType::Int64Be => (DTypeKind::Int64, true),
        ParsedDType::UInt8Be => (DTypeKind::UInt8, true),
        ParsedDType::UInt16Be => (DTypeKind::UInt16, true),
        ParsedDType::UInt32Be => (DTypeKind::UInt32, true),
        ParsedDType::UInt64Be => (DTypeKind::UInt64, true),
        ParsedDType::Opaque { size } => (DTypeKind::Opaque(size), false),
        ParsedDType::Compound { size, .. } => (DTypeKind::Compound { size }, false),
        ParsedDType::FixedString { size } | ParsedDType::Other { size } => (
            DTypeKind::Other {
                size: if size == 0 { 1 } else { size },
            },
            false,
        ),
        ParsedDType::VlenString => (DTypeKind::Other { size: 1 }, false),
    }
}

fn compound_fields_of(dtype: &ParsedDType) -> Vec<CompoundField> {
    match dtype {
        ParsedDType::Compound { fields, .. } => fields.clone(),
        _ => Vec::new(),
    }
}

fn finish_numeric(raw: &mut [u8], needs_swap: bool, elem: usize, bits: Option<&IntegerBits>) {
    if needs_swap {
        bswap_inplace(raw, elem);
    }
    if let Some(bits) = bits {
        narrow_integer_lanes(raw, elem, bits.precision, bits.offset, bits.signed);
    }
}

/// Shift a narrowed integer down to bit 0 and sign-extend when it is signed.
/// N-bit (and any short-precision integer) leaves padding bits clear.
fn narrow_integer_lanes(
    data: &mut [u8],
    elem: usize,
    precision: usize,
    offset: usize,
    signed: bool,
) {
    let width = elem.saturating_mul(8);
    if elem == 0 || elem > 8 || precision == 0 || precision > width || offset + precision > width {
        return;
    }
    if precision == width && offset == 0 {
        return;
    }
    for slot in data.chunks_exact_mut(elem) {
        let raw = read_uint_le(slot);
        let shifted = raw >> offset;
        let mask = if precision >= 64 {
            u64::MAX
        } else {
            (1u64 << precision) - 1
        };
        let bits = shifted & mask;
        let value = if signed {
            sign_extend_bits(bits, precision)
        } else {
            bits
        };
        write_uint_le(slot, value);
    }
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

fn sign_extend_bits(v: u64, precision: usize) -> u64 {
    if precision == 0 || precision >= 64 {
        return v;
    }
    let sign = 1u64 << (precision - 1);
    let mask = (1u64 << precision) - 1;
    let x = v & mask;
    if x & sign != 0 {
        x | !mask
    } else {
        x
    }
}

fn bswap_inplace(data: &mut [u8], elem: usize) {
    if elem <= 1 {
        return;
    }
    for chunk in data.chunks_exact_mut(elem) {
        chunk.reverse();
    }
}

fn make_rec(
    kind: DTypeKind,
    shape: Vec<u64>,
    data: Vec<u8>,
    attrs: Vec<(String, String)>,
    compound_fields: Vec<CompoundField>,
) -> DatasetRec {
    DatasetRec {
        shape,
        kind,
        data,
        attrs,
        chunked: false,
        filtered: false,
        compound_fields,
    }
}

fn assemble_chunks(
    data: &[u8],
    sb: &superblock::Superblock,
    index: &ChunkIndex,
    shape: &[u64],
    chunk_dims: &[u32],
    elem: usize,
    filters: &[FilterDesc],
) -> Result<Vec<u8>> {
    if elem == 0 {
        return Err(HDF5Error::InvalidHeader);
    }
    let ndims = chunk_dims.len();
    if ndims == 0 {
        return Err(HDF5Error::ChunkedNotSupported);
    }
    let full_chunk_nbytes = chunk_dims
        .iter()
        .try_fold(1u64, |a, &b| a.checked_mul(u64::from(b)))
        .ok_or(HDF5Error::ShapeMismatch)?;
    let full_chunk_u32 = u32::try_from(full_chunk_nbytes).map_err(|_| HDF5Error::ShapeMismatch)?;

    let chunks: Vec<ChunkRef> = match index {
        ChunkIndex::BTreeV1 { addr } => {
            let r = Reader::new(data, sb.offset_size, sb.length_size)?;
            if r.is_undef(*addr) {
                return Err(HDF5Error::ChunkedNotSupported);
            }
            walk_chunk_btree(data, *addr, ndims, sb.offset_size, sb.length_size, 0)?
        }
        ChunkIndex::Single {
            addr,
            nbytes,
            filter_mask,
        } => {
            let r = Reader::new(data, sb.offset_size, sb.length_size)?;
            if r.is_undef(*addr) {
                return Err(HDF5Error::ChunkedNotSupported);
            }
            alloc::vec![chunk_index::single_chunk_ref(
                *addr,
                *nbytes,
                *filter_mask,
                ndims,
                full_chunk_u32,
            )]
        }
        ChunkIndex::FixedArray { addr } => {
            let r = Reader::new(data, sb.offset_size, sb.length_size)?;
            if r.is_undef(*addr) {
                return Err(HDF5Error::ChunkedNotSupported);
            }
            chunk_index::walk_fixed_array(
                data,
                *addr,
                sb.offset_size,
                sb.length_size,
                shape,
                chunk_dims,
                0,
            )?
        }
        ChunkIndex::Implicit { addr } => {
            if !filters.is_empty() {
                return Err(HDF5Error::ChunkedNotSupported);
            }
            let r = Reader::new(data, sb.offset_size, sb.length_size)?;
            if r.is_undef(*addr) {
                return Err(HDF5Error::ChunkedNotSupported);
            }
            chunk_index::walk_implicit(*addr, shape, chunk_dims, full_chunk_u32)?
        }
        ChunkIndex::ExtensibleArray { addr } => {
            let r = Reader::new(data, sb.offset_size, sb.length_size)?;
            if r.is_undef(*addr) {
                return Err(HDF5Error::ChunkedNotSupported);
            }
            crate::earray::walk_extensible_array(
                data,
                *addr,
                sb.offset_size,
                sb.length_size,
                shape,
                chunk_dims,
                full_chunk_u32,
            )?
        }
        ChunkIndex::BTreeV2 { addr } => crate::btree2::walk_btree_v2(
            data,
            *addr,
            sb.offset_size,
            sb.length_size,
            shape,
            chunk_dims,
            full_chunk_u32,
        )?,
    };

    let nbytes = elem_count(shape)
        .saturating_mul(elem as u64)
        .try_into()
        .map_err(|_| HDF5Error::ShapeMismatch)?;
    let mut out = alloc::vec![0u8; nbytes];
    let spatial: Vec<u32> = if chunk_dims.len() == shape.len() + 1 {
        chunk_dims[..shape.len()].to_vec()
    } else if chunk_dims.len() == shape.len() {
        chunk_dims.to_vec()
    } else if shape.is_empty() {
        Vec::new()
    } else {
        return Err(HDF5Error::ShapeMismatch);
    };
    for ch in chunks {
        let n = if ch.size == 0 {
            continue;
        } else {
            ch.size as u64
        };
        let raw = Reader::new(data, sb.offset_size, sb.length_size)?.slice_at(ch.addr, n)?;
        let decoded = if filters.is_empty() && ch.filter_mask == 0 {
            raw.to_vec()
        } else if filters.is_empty() && ch.filter_mask != 0 {
            return Err(HDF5Error::FilteredNotSupported);
        } else {
            apply_filters(
                raw.to_vec(),
                filters,
                ch.filter_mask,
                full_chunk_u32 as usize,
            )?
        };
        copy_chunk(&mut out, shape, elem, &ch.offset, &spatial, &decoded)?;
    }
    Ok(out)
}

fn copy_chunk(
    dest: &mut [u8],
    shape: &[u64],
    elem: usize,
    origin: &[u64],
    chunk: &[u32],
    src: &[u8],
) -> Result<()> {
    let rank = shape.len();
    if rank == 0 {
        let n = elem.min(src.len()).min(dest.len());
        dest[..n].copy_from_slice(&src[..n]);
        return Ok(());
    }
    if origin.len() < rank || chunk.len() < rank {
        return Err(HDF5Error::ShapeMismatch);
    }
    let mut local = alloc::vec![0u64; rank];
    loop {
        let mut in_bounds = true;
        let mut g = 0u64;
        for i in 0..rank {
            let gi = origin[i].saturating_add(local[i]);
            if gi >= shape[i] {
                in_bounds = false;
                break;
            }
            g = g.saturating_mul(shape[i]).saturating_add(gi);
        }
        if in_bounds {
            let mut l = 0u64;
            for i in 0..rank {
                l = l
                    .saturating_mul(u64::from(chunk[i]))
                    .saturating_add(local[i]);
            }
            let ds = (g as usize).saturating_mul(elem);
            let ss = (l as usize).saturating_mul(elem);
            if ds.saturating_add(elem) <= dest.len() && ss.saturating_add(elem) <= src.len() {
                dest[ds..ds + elem].copy_from_slice(&src[ss..ss + elem]);
            }
        }
        let mut k = rank - 1;
        loop {
            local[k] = local[k].saturating_add(1);
            if local[k] < u64::from(chunk[k]) {
                break;
            }
            local[k] = 0;
            if k == 0 {
                return Ok(());
            }
            k -= 1;
        }
    }
}

fn elem_count(dims: &[u64]) -> u64 {
    dims.iter().copied().fold(1u64, |a, b| a.saturating_mul(b))
}

/// On-disk VL string: `u32` length, collection address, `u32` heap index.
/// Older files omit the length and store only the global-heap ID.
fn parse_vlen_heap_id(raw: &[u8], offset_size: u8) -> Option<(u64, u32)> {
    let off = offset_size as usize;
    let with_len = 4usize.checked_add(off)?.checked_add(4)?;
    let without = off.checked_add(4)?;
    if raw.len() >= with_len {
        let mut r = Reader::new(raw, offset_size, 8).ok()?;
        let _len = r.u32().ok()?;
        let coll = r.addr().ok()?;
        let idx = r.u32().ok()?;
        return Some((coll, idx));
    }
    if raw.len() >= without {
        let mut r = Reader::new(raw, offset_size, 8).ok()?;
        let coll = r.addr().ok()?;
        let idx = r.u32().ok()?;
        return Some((coll, idx));
    }
    None
}

fn attr_to_string(
    data: &[u8],
    sb: &superblock::Superblock,
    dtype: &ParsedDType,
    raw: &[u8],
) -> Option<String> {
    match dtype {
        ParsedDType::FixedString { .. } => {
            let end = raw.iter().position(|&b| b == 0).unwrap_or(raw.len());
            String::from_utf8(raw[..end].to_vec()).ok()
        }
        ParsedDType::VlenString => {
            let (coll, idx) = parse_vlen_heap_id(raw, sb.offset_size)?;
            if Reader::new(raw, sb.offset_size, sb.length_size)
                .ok()?
                .is_undef(coll)
            {
                return Some(String::new());
            }
            let obj = read_gheap_object(data, coll, idx, sb.offset_size, sb.length_size).ok()?;
            let end = obj.iter().position(|&b| b == 0).unwrap_or(obj.len());
            String::from_utf8(obj[..end].to_vec()).ok()
        }
        _ => None,
    }
}
