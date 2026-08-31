//! In-memory HDF5 graph used for both encode and the public file object.

use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use crate::error::{HDF5Error, Result};
use crate::HDF5DType;
use crate::{HDF5_CREATED_ATTR, HDF5_MAX_NAME_LEN, HDF5_WRITER_VERSION, HDF5_WRITER_VERSION_ATTR};

#[derive(Clone, Debug)]
pub struct DatasetRec {
    pub shape: Vec<u64>,
    pub kind: DTypeKind,
    pub data: Vec<u8>,
    pub attrs: Vec<(String, String)>,
    pub chunked: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DTypeKind {
    Float64,
    Float32,
    Opaque(usize),
}

impl DTypeKind {
    pub fn elem_size(self) -> usize {
        match self {
            Self::Float64 => 8,
            Self::Float32 => 4,
            Self::Opaque(n) => n,
        }
    }

    pub fn to_public(self) -> HDF5DType {
        match self {
            Self::Float64 => HDF5DType::Float64,
            Self::Float32 => HDF5DType::Float32,
            Self::Opaque(n) => HDF5DType::Opaque(n),
        }
    }
}

#[derive(Clone, Debug)]
pub struct FileModel {
    pub groups: BTreeMap<String, Vec<String>>,
    pub datasets: BTreeMap<String, DatasetRec>,
}

impl FileModel {
    pub fn new() -> Self {
        let mut groups = BTreeMap::new();
        groups.insert(String::new(), Vec::new());
        Self {
            groups,
            datasets: BTreeMap::new(),
        }
    }

    pub fn normalize(path: &str) -> Result<String> {
        let mut parts = Vec::new();
        for s in path.split('/') {
            if s.is_empty() || s == "." {
                continue;
            }
            if s == ".." {
                return Err(HDF5Error::InvalidHeader);
            }
            if s.len() > HDF5_MAX_NAME_LEN {
                return Err(HDF5Error::NameTooLong);
            }
            parts.push(s);
        }
        Ok(parts.join("/"))
    }

    pub fn parent_name(path: &str) -> Result<(String, String)> {
        let n = Self::normalize(path)?;
        match n.rfind('/') {
            Some(i) => Ok((n[..i].to_string(), n[i + 1..].to_string())),
            None => Ok((String::new(), n)),
        }
    }

    pub fn create_group(&mut self, path: &str) -> Result<()> {
        let p = Self::normalize(path)?;
        if p.is_empty() {
            return Ok(());
        }
        if self.datasets.contains_key(&p) {
            return Err(HDF5Error::InvalidHeader);
        }
        let (parent, name) = Self::parent_name(&p)?;
        if !self.groups.contains_key(&parent) {
            return Err(HDF5Error::NotFound(parent));
        }
        if !self.groups.contains_key(&p) {
            self.groups.insert(p.clone(), Vec::new());
            let kids = self
                .groups
                .get_mut(&parent)
                .ok_or_else(|| HDF5Error::NotFound(parent.clone()))?;
            if !kids.iter().any(|k| k == &name) {
                kids.push(name);
            }
        }
        Ok(())
    }

    pub fn ensure_parent(&mut self, path: &str) -> Result<(String, String)> {
        let (parent, name) = Self::parent_name(path)?;
        if !parent.is_empty() && !self.groups.contains_key(&parent) {
            self.create_group(&parent)?;
        }
        Ok((parent, name))
    }

    pub fn put_dataset(&mut self, path: &str, rec: DatasetRec) -> Result<()> {
        let p = Self::normalize(path)?;
        if p.is_empty() {
            return Err(HDF5Error::InvalidHeader);
        }
        let (parent, name) = self.ensure_parent(&p)?;
        if let Some(kids) = self.groups.get_mut(&parent) {
            if !kids.iter().any(|k| k == &name) {
                kids.push(name);
            }
        }
        self.groups.remove(&p);
        self.datasets.insert(p, rec);
        Ok(())
    }

    pub fn default_attrs() -> Vec<(String, String)> {
        let created = created_stamp();
        alloc::vec![
            (
                HDF5_WRITER_VERSION_ATTR.to_string(),
                HDF5_WRITER_VERSION.to_string()
            ),
            (HDF5_CREATED_ATTR.to_string(), created),
        ]
    }

    pub fn list_datasets(&self) -> Vec<String> {
        self.datasets.keys().cloned().collect()
    }

    pub fn list_groups(&self) -> Vec<String> {
        let mut v: Vec<String> = self
            .groups
            .keys()
            .map(|k| {
                if k.is_empty() {
                    "/".to_string()
                } else {
                    k.clone()
                }
            })
            .collect();
        v.sort();
        v
    }
}

fn created_stamp() -> String {
    #[cfg(feature = "std")]
    {
        match std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
            Ok(d) => alloc::format!("{}", d.as_secs()),
            Err(_) => String::from("0"),
        }
    }
    #[cfg(not(feature = "std"))]
    {
        String::from("0")
    }
}
