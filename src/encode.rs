//! Serialize [`FileModel`] to a superblock-v2 HDF5 image.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

use crate::buf::align8;
use crate::error::{HDF5Error, Result};
use crate::messages::{
    encode_dataspace, encode_fill, encode_group_info, encode_hard_link, encode_ieee_f32le,
    encode_ieee_f64le, encode_integer, encode_layout_contiguous, encode_link_info, encode_opaque,
    encode_string_attr,
};
use crate::model::{DTypeKind, FileModel};
use crate::ohdr::{encode_ohdr_v2, RawMsg};
use crate::superblock;
use crate::{
    HDF5_MSG_ATTRIBUTE, HDF5_MSG_DATASPACE, HDF5_MSG_DATATYPE, HDF5_MSG_FILL,
    HDF5_MSG_FLAG_CONSTANT, HDF5_MSG_GROUP_INFO, HDF5_MSG_LAYOUT, HDF5_MSG_LINK,
    HDF5_MSG_LINK_INFO, HDF5_SUPERBLOCK_V2_SIZE,
};

pub fn encode(model: &FileModel) -> Result<Vec<u8>> {
    let mut data_addrs: BTreeMap<String, u64> = BTreeMap::new();
    let mut pos = HDF5_SUPERBLOCK_V2_SIZE as u64;
    for (path, ds) in &model.datasets {
        pos = align8(pos);
        data_addrs.insert(path.clone(), pos);
        pos = pos.saturating_add(ds.data.len() as u64);
    }

    let mut ohdr_addrs: BTreeMap<String, u64> = BTreeMap::new();
    let mut ohdr_bytes: BTreeMap<String, Vec<u8>> = BTreeMap::new();

    for (path, ds) in &model.datasets {
        if ds.filtered {
            return Err(HDF5Error::FilteredNotSupported);
        }
        if ds.chunked {
            return Err(HDF5Error::ChunkedNotSupported);
        }
        let addr = *data_addrs.get(path).ok_or(HDF5Error::InvalidHeader)?;
        let mut msgs = Vec::new();
        msgs.push(RawMsg {
            ty: HDF5_MSG_DATASPACE,
            flags: HDF5_MSG_FLAG_CONSTANT,
            body: encode_dataspace(&ds.shape, true)?,
        });
        let dtype = match ds.kind {
            DTypeKind::Float64 => encode_ieee_f64le(),
            DTypeKind::Float32 => encode_ieee_f32le(),
            DTypeKind::Int8 => encode_integer(1, true),
            DTypeKind::Int16 => encode_integer(2, true),
            DTypeKind::Int32 => encode_integer(4, true),
            DTypeKind::Int64 => encode_integer(8, true),
            DTypeKind::UInt8 => encode_integer(1, false),
            DTypeKind::UInt16 => encode_integer(2, false),
            DTypeKind::UInt32 => encode_integer(4, false),
            DTypeKind::UInt64 => encode_integer(8, false),
            DTypeKind::Opaque(n) => encode_opaque(n as u32),
            DTypeKind::Other { .. } => return Err(HDF5Error::UnsupportedDtype),
        };
        msgs.push(RawMsg {
            ty: HDF5_MSG_DATATYPE,
            flags: HDF5_MSG_FLAG_CONSTANT,
            body: dtype,
        });
        msgs.push(RawMsg {
            ty: HDF5_MSG_FILL,
            flags: 0,
            body: encode_fill(),
        });
        msgs.push(RawMsg {
            ty: HDF5_MSG_LAYOUT,
            flags: 0,
            body: encode_layout_contiguous(addr, ds.data.len() as u64),
        });
        for (n, v) in &ds.attrs {
            msgs.push(RawMsg {
                ty: HDF5_MSG_ATTRIBUTE,
                flags: 0,
                body: encode_string_attr(n, v)?,
            });
        }
        let bytes = encode_ohdr_v2(&msgs);
        pos = align8(pos);
        ohdr_addrs.insert(path.clone(), pos);
        pos = pos.saturating_add(bytes.len() as u64);
        ohdr_bytes.insert(path.clone(), bytes);
    }

    let mut group_paths: Vec<String> = model.groups.keys().cloned().collect();
    group_paths
        .sort_by_key(|p| core::cmp::Reverse(p.matches('/').count() + usize::from(!p.is_empty())));

    for gpath in &group_paths {
        let kids = model.groups.get(gpath).ok_or(HDF5Error::InvalidHeader)?;
        let mut msgs = Vec::new();
        msgs.push(RawMsg {
            ty: HDF5_MSG_LINK_INFO,
            flags: 0,
            body: encode_link_info(),
        });
        msgs.push(RawMsg {
            ty: HDF5_MSG_GROUP_INFO,
            flags: 0,
            body: encode_group_info(),
        });
        for name in kids {
            let child_path = if gpath.is_empty() {
                name.clone()
            } else {
                alloc::format!("{gpath}/{name}")
            };
            let child_addr = *ohdr_addrs
                .get(&child_path)
                .ok_or(HDF5Error::InvalidHeader)?;
            msgs.push(RawMsg {
                ty: HDF5_MSG_LINK,
                flags: 0,
                body: encode_hard_link(name, child_addr)?,
            });
        }
        let bytes = encode_ohdr_v2(&msgs);
        pos = align8(pos);
        ohdr_addrs.insert(gpath.clone(), pos);
        pos = pos.saturating_add(bytes.len() as u64);
        ohdr_bytes.insert(gpath.clone(), bytes);
    }

    let root = *ohdr_addrs.get("").ok_or(HDF5Error::InvalidHeader)?;
    let mut file = alloc::vec![0u8; pos as usize];
    let sb = superblock::encode_v2(root, pos);
    file[..HDF5_SUPERBLOCK_V2_SIZE].copy_from_slice(&sb);
    for (path, ds) in &model.datasets {
        let a = *data_addrs.get(path).ok_or(HDF5Error::InvalidHeader)? as usize;
        file[a..a + ds.data.len()].copy_from_slice(&ds.data);
    }
    for (path, bytes) in &ohdr_bytes {
        let a = *ohdr_addrs.get(path).ok_or(HDF5Error::InvalidHeader)? as usize;
        if a + bytes.len() > file.len() {
            file.resize(a + bytes.len(), 0);
        }
        file[a..a + bytes.len()].copy_from_slice(bytes);
    }
    let eof = file.len() as u64;
    let sb = superblock::encode_v2(root, eof);
    file[..HDF5_SUPERBLOCK_V2_SIZE].copy_from_slice(&sb);
    Ok(file)
}

/// Build a minimal valid file whose only dataset is chunked (for the error gold).
pub fn encode_chunked_stub() -> Result<Vec<u8>> {
    use crate::{HDF5_LAYOUT_CHUNKED, HDF5_LAYOUT_VERSION};
    let mut layout = Vec::new();
    layout.push(HDF5_LAYOUT_VERSION);
    layout.push(HDF5_LAYOUT_CHUNKED);
    layout.push(1);
    crate::buf::push_u64(&mut layout, crate::HDF5_UNDEF_ADDR_8);
    crate::buf::push_u32(&mut layout, 1);
    crate::buf::push_u32(&mut layout, 8);

    let mut msgs = Vec::new();
    msgs.push(RawMsg {
        ty: HDF5_MSG_DATASPACE,
        flags: HDF5_MSG_FLAG_CONSTANT,
        body: encode_dataspace(&[1], false)?,
    });
    msgs.push(RawMsg {
        ty: HDF5_MSG_DATATYPE,
        flags: HDF5_MSG_FLAG_CONSTANT,
        body: encode_ieee_f64le(),
    });
    msgs.push(RawMsg {
        ty: HDF5_MSG_LAYOUT,
        flags: 0,
        body: layout,
    });
    let ds_ohdr = encode_ohdr_v2(&msgs);
    let ds_addr = HDF5_SUPERBLOCK_V2_SIZE as u64;

    let mut gmsgs = Vec::new();
    gmsgs.push(RawMsg {
        ty: HDF5_MSG_LINK_INFO,
        flags: 0,
        body: encode_link_info(),
    });
    gmsgs.push(RawMsg {
        ty: HDF5_MSG_GROUP_INFO,
        flags: 0,
        body: encode_group_info(),
    });
    gmsgs.push(RawMsg {
        ty: HDF5_MSG_LINK,
        flags: 0,
        body: encode_hard_link("data", ds_addr)?,
    });
    let g_bytes = encode_ohdr_v2(&gmsgs);
    let g_addr = align8(ds_addr + ds_ohdr.len() as u64);
    let eof = g_addr + g_bytes.len() as u64;
    let mut file = alloc::vec![0u8; eof as usize];
    file[..HDF5_SUPERBLOCK_V2_SIZE].copy_from_slice(&superblock::encode_v2(g_addr, eof));
    file[ds_addr as usize..ds_addr as usize + ds_ohdr.len()].copy_from_slice(&ds_ohdr);
    file[g_addr as usize..g_addr as usize + g_bytes.len()].copy_from_slice(&g_bytes);
    Ok(file)
}
