//! Parse an HDF5 file image into [`FileModel`].

use alloc::string::String;
use alloc::vec::Vec;

use crate::btree::{read_gheap_object, read_local_heap, walk_group_btree};
use crate::buf::Reader;
use crate::error::{HDF5Error, Result};
use crate::messages::{
    parse_attribute, parse_dataspace, parse_datatype, parse_layout, parse_link,
    parse_link_info_heap, parse_symbol_table, ParsedDType, ParsedLayout,
};
use crate::model::{DTypeKind, DatasetRec, FileModel};
use crate::ohdr::{parse_ohdr, RawMsg};
use crate::superblock;
use crate::{
    HDF5_MAX_WALK_DEPTH, HDF5_MSG_ATTRIBUTE, HDF5_MSG_DATASPACE, HDF5_MSG_DATATYPE,
    HDF5_MSG_LAYOUT, HDF5_MSG_LINK, HDF5_MSG_LINK_INFO, HDF5_MSG_SYMBOL_TABLE,
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
    if space.dims.len() > 2 {
        return Err(HDF5Error::RankNotSupported);
    }
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
    match layout {
        ParsedLayout::Chunked => {
            let kind = match dtype {
                ParsedDType::Float64Le => DTypeKind::Float64,
                ParsedDType::Float32Le => DTypeKind::Float32,
                ParsedDType::Opaque { size } => DTypeKind::Opaque(size),
                _ => DTypeKind::Float64,
            };
            Ok(Some(DatasetRec {
                shape: space.dims,
                kind,
                data: Vec::new(),
                attrs,
                chunked: true,
            }))
        }
        ParsedLayout::Compact { data: raw } => Ok(Some(make_rec(dtype, space.dims, raw, attrs)?)),
        ParsedLayout::Contiguous { addr, size } => {
            let n = if size == 0 {
                elem_count(&space.dims).saturating_mul(dtype.element_size() as u64)
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
            Ok(Some(make_rec(dtype, space.dims, raw, attrs)?))
        }
    }
}

fn make_rec(
    dtype: ParsedDType,
    shape: Vec<u64>,
    data: Vec<u8>,
    attrs: Vec<(String, String)>,
) -> Result<DatasetRec> {
    let kind = match dtype {
        ParsedDType::Float64Le => DTypeKind::Float64,
        ParsedDType::Float32Le => DTypeKind::Float32,
        ParsedDType::Opaque { size } => DTypeKind::Opaque(size),
        _ => {
            return Ok(DatasetRec {
                shape,
                kind: DTypeKind::Opaque(dtype.element_size()),
                data,
                attrs,
                chunked: false,
            });
        }
    };
    Ok(DatasetRec {
        shape,
        kind,
        data,
        attrs,
        chunked: false,
    })
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
