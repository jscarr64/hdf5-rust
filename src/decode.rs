//! Parse an HDF5 file image into [`FileModel`].

use alloc::string::String;
use alloc::vec::Vec;

use crate::btree::{read_gheap_object, read_local_heap, walk_chunk_btree, walk_group_btree};
use crate::buf::Reader;
use crate::error::{HDF5Error, Result};
use crate::messages::{
    parse_attribute, parse_dataspace, parse_datatype, parse_filter_count, parse_layout, parse_link,
    parse_link_info_heap, parse_symbol_table, ParsedDType, ParsedLayout,
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
    walk(
        data,
        &sb,
        sb.root_ohdr,
        "",
        &mut model,
        0,
        sb.root_btree,
        sb.root_heap,
    )?;
    Ok(model)
}

fn walk(
    data: &[u8],
    sb: &superblock::Superblock,
    ohdr: u64,
    path: &str,
    model: &mut FileModel,
    depth: usize,
    btree: Option<u64>,
    heap: Option<u64>,
) -> Result<()> {
    if depth > HDF5_MAX_WALK_DEPTH {
        return Err(HDF5Error::InvalidHeader);
    }
    let msgs = parse_ohdr(data, ohdr, sb.offset_size, sb.length_size)?;
    if let Some(ds) = try_dataset(data, sb, &msgs)? {
        if !path.is_empty() {
            model.put_dataset(path, ds)?;
        }
        return Ok(());
    }
    if !path.is_empty() {
        model.create_group(path)?;
    }
    let mut kids: Vec<(String, u64)> = Vec::new();
    if let Some(h) = msgs.iter().find(|m| m.ty == HDF5_MSG_LINK_INFO) {
        if parse_link_info_heap(&h.body, sb.offset_size)?.is_some() {
            return Err(HDF5Error::DenseGroupsNotSupported);
        }
    }
    for m in &msgs {
        if m.ty == HDF5_MSG_LINK {
            let l = parse_link(&m.body, sb.offset_size)?;
            kids.push((l.name, l.ohdr));
        }
    }
    if kids.is_empty() {
        if let Some(m) = msgs.iter().find(|x| x.ty == HDF5_MSG_SYMBOL_TABLE) {
            let (bt, hp) = parse_symbol_table(&m.body, sb.offset_size)?;
            let heap_bytes = read_local_heap(data, hp, sb.offset_size, sb.length_size)?;
            for c in walk_group_btree(data, bt, heap_bytes, sb.offset_size, sb.length_size, 0)? {
                kids.push((c.name, c.ohdr));
            }
        } else if let (Some(bt), Some(hp)) = (btree, heap) {
            let heap_bytes = read_local_heap(data, hp, sb.offset_size, sb.length_size)?;
            for c in walk_group_btree(data, bt, heap_bytes, sb.offset_size, sb.length_size, 0)? {
                kids.push((c.name, c.ohdr));
            }
        }
    }
    for (name, child) in kids {
        let child_path = if path.is_empty() {
            name
        } else {
            alloc::format!("{path}/{name}")
        };
        walk(data, sb, child, &child_path, model, depth + 1, None, None)?;
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
    let mut nfilters = 0u8;
    for m in msgs {
        if m.ty == HDF5_MSG_FILTER {
            match parse_filter_count(&m.body) {
                Ok(n) => nfilters = n,
                Err(_) => nfilters = 1,
            }
        }
    }
    let filtered = nfilters > 0;
    let kind = dtype_kind(dtype);
    match layout {
        ParsedLayout::Chunked { addr, chunk_dims } => {
            if filtered {
                return Ok(Some(DatasetRec {
                    shape: space.dims,
                    kind,
                    data: Vec::new(),
                    attrs,
                    chunked: true,
                    filtered: true,
                }));
            }
            let assembled =
                assemble_chunks(data, sb, addr, &space.dims, &chunk_dims, kind.elem_size());
            match assembled {
                Ok(raw) => Ok(Some(DatasetRec {
                    shape: space.dims,
                    kind,
                    data: raw,
                    attrs,
                    chunked: false,
                    filtered: false,
                })),
                Err(_) => Ok(Some(DatasetRec {
                    shape: space.dims,
                    kind,
                    data: Vec::new(),
                    attrs,
                    chunked: true,
                    filtered: false,
                })),
            }
        }
        ParsedLayout::Compact { data: raw } => Ok(Some(make_rec(kind, space.dims, raw, attrs))),
        ParsedLayout::Contiguous { addr, size } => {
            let n = if size == 0 {
                elem_count(&space.dims).saturating_mul(kind.elem_size() as u64)
            } else {
                size
            };
            let raw = if n == 0 {
                Vec::new()
            } else {
                Reader::new(data, sb.offset_size, sb.length_size)?
                    .slice_at(addr, n)?
                    .to_vec()
            };
            Ok(Some(make_rec(kind, space.dims, raw, attrs)))
        }
    }
}

fn dtype_kind(dtype: ParsedDType) -> DTypeKind {
    match dtype {
        ParsedDType::Float64Le => DTypeKind::Float64,
        ParsedDType::Float32Le => DTypeKind::Float32,
        ParsedDType::Int8Le => DTypeKind::Int8,
        ParsedDType::Int16Le => DTypeKind::Int16,
        ParsedDType::Int32Le => DTypeKind::Int32,
        ParsedDType::Int64Le => DTypeKind::Int64,
        ParsedDType::UInt8Le => DTypeKind::UInt8,
        ParsedDType::UInt16Le => DTypeKind::UInt16,
        ParsedDType::UInt32Le => DTypeKind::UInt32,
        ParsedDType::UInt64Le => DTypeKind::UInt64,
        ParsedDType::Opaque { size } => DTypeKind::Opaque(size),
        ParsedDType::FixedString { size } | ParsedDType::Other { size } => DTypeKind::Other {
            size: if size == 0 { 1 } else { size },
        },
        ParsedDType::VlenString => DTypeKind::Other { size: 1 },
    }
}

fn make_rec(
    kind: DTypeKind,
    shape: Vec<u64>,
    data: Vec<u8>,
    attrs: Vec<(String, String)>,
) -> DatasetRec {
    DatasetRec {
        shape,
        kind,
        data,
        attrs,
        chunked: false,
        filtered: false,
    }
}

fn assemble_chunks(
    data: &[u8],
    sb: &superblock::Superblock,
    btree_addr: u64,
    shape: &[u64],
    chunk_dims: &[u32],
    elem: usize,
) -> Result<Vec<u8>> {
    if elem == 0 {
        return Err(HDF5Error::InvalidHeader);
    }
    let r = Reader::new(data, sb.offset_size, sb.length_size)?;
    if r.is_undef(btree_addr) {
        return Err(HDF5Error::ChunkedNotSupported);
    }
    let ndims = chunk_dims.len();
    if ndims == 0 {
        return Err(HDF5Error::ChunkedNotSupported);
    }
    let chunks = walk_chunk_btree(data, btree_addr, ndims, sb.offset_size, sb.length_size, 0)?;
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
        if ch.filter_mask != 0 {
            return Err(HDF5Error::FilteredNotSupported);
        }
        let n = if ch.size == 0 {
            continue;
        } else {
            ch.size as u64
        };
        let raw = Reader::new(data, sb.offset_size, sb.length_size)?.slice_at(ch.addr, n)?;
        copy_chunk(&mut out, shape, elem, &ch.offset, &spatial, raw)?;
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
