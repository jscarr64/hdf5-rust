//! Named constants. No other magic numbers in the crate.

/// HDF5 format signature (`\211HDF\r\n\032\n`).
pub const HDF5_SIGNATURE: [u8; 8] = [0x89, 0x48, 0x44, 0x46, 0x0d, 0x0a, 0x1a, 0x0a];

/// Superblock version this crate writes.
pub const HDF5_SUPERBLOCK_V2: u8 = 2;

/// Maximum dataspace rank (HDF5 format `H5S_MAX_RANK`). Rank 2 is not a cap.
pub const HDF5_MAX_DIMS: usize = 32;

/// Maximum link / attribute / path component length in bytes.
pub const HDF5_MAX_NAME_LEN: usize = 65535;

/// Dataset attribute written on every dataset this crate creates.
pub const HDF5_WRITER_VERSION_ATTR: &str = "hdf5-rust-version";

/// Second dataset attribute written on every dataset this crate creates.
pub const HDF5_CREATED_ATTR: &str = "created";

/// Value stored in [`HDF5_WRITER_VERSION_ATTR`] (this crate’s version).
pub const HDF5_WRITER_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Object-header signature for version 2 headers.
pub const HDF5_OHDR_SIGNATURE: [u8; 4] = *b"OHDR";

/// Object-header continuation chunk signature.
pub const HDF5_OCHK_SIGNATURE: [u8; 4] = *b"OCHK";

/// Version-1 B-tree node signature.
pub const HDF5_TREE_SIGNATURE: [u8; 4] = *b"TREE";

/// Symbol-table node signature.
pub const HDF5_SNOD_SIGNATURE: [u8; 4] = *b"SNOD";

/// Local heap signature.
pub const HDF5_HEAP_SIGNATURE: [u8; 4] = *b"HEAP";

/// Global heap collection signature.
pub const HDF5_GCOL_SIGNATURE: [u8; 4] = *b"GCOL";

/// Size of offsets this crate writes (bytes).
pub const HDF5_OFFSET_SIZE: u8 = 8;

/// Size of lengths this crate writes (bytes).
pub const HDF5_LENGTH_SIZE: u8 = 8;

/// Undefined address for 8-byte offsets.
pub const HDF5_UNDEF_ADDR_8: u64 = u64::MAX;

/// Maximum first-dimension extent (HDF5 unlimited).
pub const HDF5_UNLIMITED: u64 = u64::MAX;

/// Superblock v2 on-disk size with 8-byte offsets (including checksum).
pub const HDF5_SUPERBLOCK_V2_SIZE: usize = 48;

/// Object-header v2 version byte.
pub const HDF5_OHDR_VERSION: u8 = 2;

/// Object-header flags: 4-byte chunk-0 size, no timestamps.
pub const HDF5_OHDR_FLAGS_CHUNK4: u8 = 0x02;

/// Header message type: dataspace.
pub const HDF5_MSG_DATASPACE: u8 = 0x01;
/// Header message type: link info.
pub const HDF5_MSG_LINK_INFO: u8 = 0x02;
/// Header message type: datatype.
pub const HDF5_MSG_DATATYPE: u8 = 0x03;
/// Header message type: fill value.
pub const HDF5_MSG_FILL: u8 = 0x05;
/// Header message type: link.
pub const HDF5_MSG_LINK: u8 = 0x06;
/// Header message type: layout.
pub const HDF5_MSG_LAYOUT: u8 = 0x08;
/// Header message type: group info.
pub const HDF5_MSG_GROUP_INFO: u8 = 0x0A;
/// Header message type: attribute.
pub const HDF5_MSG_ATTRIBUTE: u8 = 0x0C;
/// Header message type: continuation.
pub const HDF5_MSG_CONTINUATION: u8 = 0x10;
/// Header message type: symbol table (old-style group).
pub const HDF5_MSG_SYMBOL_TABLE: u8 = 0x11;
/// Header message type: nil.
pub const HDF5_MSG_NIL: u8 = 0x00;

/// Layout class: contiguous.
pub const HDF5_LAYOUT_CONTIGUOUS: u8 = 1;
/// Layout class: chunked.
pub const HDF5_LAYOUT_CHUNKED: u8 = 2;
/// Layout class: compact.
pub const HDF5_LAYOUT_COMPACT: u8 = 0;

/// Datatype class: floating-point.
pub const HDF5_CLASS_FLOAT: u8 = 1;
/// Datatype class: string.
pub const HDF5_CLASS_STRING: u8 = 3;
/// Datatype class: opaque.
pub const HDF5_CLASS_OPAQUE: u8 = 5;
/// Datatype class: variable-length.
pub const HDF5_CLASS_VLEN: u8 = 9;

/// Dataspace type: scalar.
pub const HDF5_SPACE_SCALAR: u8 = 0;
/// Dataspace type: simple.
pub const HDF5_SPACE_SIMPLE: u8 = 1;

/// Dataspace message version written by this crate.
pub const HDF5_DATASPACE_VERSION: u8 = 2;
/// Layout message version written by this crate.
pub const HDF5_LAYOUT_VERSION: u8 = 3;
/// Link message version.
pub const HDF5_LINK_VERSION: u8 = 1;
/// Attribute message version written by this crate.
pub const HDF5_ATTR_VERSION: u8 = 2;
/// Datatype message version for IEEE and opaque.
pub const HDF5_DTYPE_VERSION: u8 = 1;
/// Fill-value message version.
pub const HDF5_FILL_VERSION: u8 = 3;

/// Opaque ASCII tag written on opaque datatypes (NUL-padded to 8).
pub const HDF5_OPAQUE_TAG: [u8; 8] = *b"hdf5rust";

/// Object-header v1 message flags: data is constant.
pub const HDF5_MSG_FLAG_CONSTANT: u8 = 0x01;

/// B-tree node type: group (symbol table).
pub const HDF5_BTREE_GROUP: u8 = 0;
/// B-tree node type: raw data chunks.
pub const HDF5_BTREE_CHUNK: u8 = 1;

/// Header message type: filter pipeline.
pub const HDF5_MSG_FILTER: u8 = 0x0B;

/// Datatype class: fixed-point integer.
pub const HDF5_CLASS_INTEGER: u8 = 0;

/// Maximum object-header walk depth (continuation + nested groups).
pub const HDF5_MAX_WALK_DEPTH: usize = 64;

/// Superblock search: first candidate after offset 0.
pub const HDF5_SUPERBLOCK_SEARCH_START: u64 = 512;
