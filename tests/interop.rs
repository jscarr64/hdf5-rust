//! Interop golds: files written by h5py, files we write that h5py (and MATLAB if present) must open.

#![cfg(feature = "std")]

use hdf5_rust::{HDF5DType, HDF5Error, Hdf5File, HDF5_WRITER_VERSION_ATTR};
use std::path::{Path, PathBuf};
use std::process::Command;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn interop_dir() -> PathBuf {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/interop_tmp");
    std::fs::create_dir_all(&p).expect("tmpdir");
    p
}

/// IEEE binary64 bits for integer values 0..=11 (no hardware float).
fn ieee64_ints_0_to_11() -> [u64; 12] {
    [
        0x0000_0000_0000_0000,
        0x3FF0_0000_0000_0000,
        0x4000_0000_0000_0000,
        0x4008_0000_0000_0000,
        0x4010_0000_0000_0000,
        0x4014_0000_0000_0000,
        0x4018_0000_0000_0000,
        0x401C_0000_0000_0000,
        0x4020_0000_0000_0000,
        0x4022_0000_0000_0000,
        0x4024_0000_0000_0000,
        0x4026_0000_0000_0000,
    ]
}

#[test]
fn gold_read_h5py_default_f64() {
    let f = Hdf5File::open(fixture("h5py_f64_default.h5")).expect("open h5py default");
    let (shape, bits) = f.read_f64("data").expect("read_f64");
    assert_eq!(shape, vec![2, 3]);
    let want = &ieee64_ints_0_to_11()[1..7];
    assert_eq!(bits, want);
}

#[test]
fn gold_read_h5py_string_attr() {
    let f = Hdf5File::open(fixture("h5py_f64_attr.h5")).expect("open attr file");
    let units = f.read_attr_str("data", "units").expect("units attr");
    assert_eq!(units, "kelvin");
    let (shape, bits) = f.read_f64("data").expect("read");
    assert_eq!(shape, vec![4, 3]);
    assert_eq!(bits, ieee64_ints_0_to_11());
}

#[test]
fn gold_read_h5py_three_datasets() {
    let f = Hdf5File::open(fixture("h5py_three.h5")).expect("open three");
    let mut names = f.list_datasets();
    names.sort();
    assert_eq!(names, vec!["a", "b", "c"]);
    assert_eq!(f.dataset_shape("b").expect("shape"), vec![2]);
}

#[test]
fn gold_read_h5py_chunked_uncompressed() {
    let f = Hdf5File::open(fixture("h5py_chunked.h5")).expect("open chunked file");
    assert_eq!(f.dataset_dtype("data").expect("dtype"), HDF5DType::Float64);
    assert_eq!(f.dataset_shape("data").expect("shape"), vec![5, 4]);
    let (shape, bits) = f.read_f64("data").expect("read uncompressed chunks");
    assert_eq!(shape, vec![5, 4]);
    // Integer values 0..=19 as IEEE binary64 bits (C order).
    let want = [
        0x0000_0000_0000_0000u64,
        0x3FF0_0000_0000_0000,
        0x4000_0000_0000_0000,
        0x4008_0000_0000_0000,
        0x4010_0000_0000_0000,
        0x4014_0000_0000_0000,
        0x4018_0000_0000_0000,
        0x401C_0000_0000_0000,
        0x4020_0000_0000_0000,
        0x4022_0000_0000_0000,
        0x4024_0000_0000_0000,
        0x4026_0000_0000_0000,
        0x4028_0000_0000_0000,
        0x402A_0000_0000_0000,
        0x402C_0000_0000_0000,
        0x402E_0000_0000_0000,
        0x4030_0000_0000_0000,
        0x4031_0000_0000_0000,
        0x4032_0000_0000_0000,
        0x4033_0000_0000_0000,
    ];
    assert_eq!(bits, want);
}

#[test]
fn gold_read_h5py_int32() {
    let f = Hdf5File::open(fixture("h5py_int32.h5")).expect("open int32");
    assert_eq!(f.dataset_dtype("data").expect("dtype"), HDF5DType::Int32);
    let (shape, bits) = f.read_i32("data").expect("read_i32");
    assert_eq!(shape, vec![2, 3]);
    assert_eq!(bits, vec![0, 1, 2, 3, 4, 5]);
    match f.read_f64("data") {
        Err(HDF5Error::TypeMismatch) => {}
        other => panic!("expected TypeMismatch, got {other:?}"),
    }
}

#[test]
fn gold_read_h5py_rank3_and_mixed() {
    let cube = Hdf5File::open(fixture("h5py_rank3.h5")).expect("open rank3");
    let (shape, bits) = cube.read_f64("cube").expect("read cube");
    assert_eq!(shape, vec![2, 3, 4]);
    assert_eq!(bits.len(), 24);
    assert_eq!(bits[0], 0);
    assert_eq!(bits[23], 0x4037_0000_0000_0000); // 23.0

    let mixed = Hdf5File::open(fixture("h5py_mixed.h5")).expect("open mixed");
    let mut names = mixed.list_datasets();
    names.sort();
    assert_eq!(names, vec!["cube", "table"]);
    let (tshape, tbits) = mixed.read_f64("table").expect("table");
    assert_eq!(tshape, vec![2, 3]);
    assert_eq!(&tbits, &ieee64_ints_0_to_11()[1..7]);
    let (cshape, _) = mixed.read_f64("cube").expect("cube beside table");
    assert_eq!(cshape, vec![2, 3, 4]);
}

#[test]
fn gold_read_h5py_rank4() {
    let f = Hdf5File::open(fixture("h5py_rank4.h5")).expect("open rank4");
    let (shape, bits) = f.read_f64("vol").expect("read vol");
    assert_eq!(shape, vec![2, 2, 3, 4]);
    assert_eq!(bits.len(), 48);
    assert_eq!(bits[0], 0);
    assert_eq!(bits[47], 0x4047_8000_0000_0000); // 47.0
}

#[test]
fn gold_read_h5py_rank5() {
    let f = Hdf5File::open(fixture("h5py_rank5.h5")).expect("open rank5");
    assert_eq!(f.dataset_dtype("stack").expect("dtype"), HDF5DType::Int32);
    let (shape, bits) = f.read_i32("stack").expect("read stack");
    assert_eq!(shape, vec![2, 2, 2, 2, 3]);
    assert_eq!(bits.len(), 48);
    assert_eq!(bits[0], 0);
    assert_eq!(bits[47], 47);
}

#[test]
fn gold_read_h5py_gzip_is_filtered() {
    let f = Hdf5File::open(fixture("h5py_gzip.h5")).expect("open gzip");
    assert_eq!(f.dataset_dtype("data").expect("dtype"), HDF5DType::Float64);
    assert_eq!(f.dataset_shape("data").expect("shape"), vec![5, 4]);
    match f.read_f64("data") {
        Err(HDF5Error::FilteredNotSupported) => {}
        other => panic!("expected FilteredNotSupported, got {other:?}"),
    }
}

#[test]
fn gold_read_h5py_complex_is_other() {
    let f = Hdf5File::open(fixture("h5py_complex.h5")).expect("open complex");
    assert_eq!(f.dataset_dtype("data").expect("dtype"), HDF5DType::Other);
    match f.read_f64("data") {
        Err(HDF5Error::TypeMismatch) => {}
        other => panic!("expected TypeMismatch, got {other:?}"),
    }
    match f.read_opaque("data") {
        Err(HDF5Error::TypeMismatch) => {}
        other => panic!("opaque must not swallow compound, got {other:?}"),
    }
}

fn write_ours() -> (PathBuf, Vec<u64>) {
    let bits = ieee64_ints_0_to_11()[1..7].to_vec();
    let mut f = Hdf5File::create();
    f.write_f64("data", &[2, 3], &bits).expect("write");
    assert_eq!(f.dataset_dtype("data").expect("dtype"), HDF5DType::Float64);
    let path = interop_dir().join("ours.h5");
    f.save(&path).expect("save");
    (path, bits)
}

#[test]
fn gold_h5py_opens_our_i32() {
    let mut f = Hdf5File::create();
    f.write_i32("data", &[2, 3], &[0, 1, 2, 3, 4, 5])
        .expect("write");
    let path = interop_dir().join("ours_i32.h5");
    f.save(&path).expect("save");
    let py = r#"
import h5py, numpy as np, sys
p = sys.argv[1]
with h5py.File(p, "r") as hf:
    d = hf["data"]
    got = np.array(d)
    want = np.array([[0,1,2],[3,4,5]], dtype=np.int32)
    assert got.shape == (2,3), got.shape
    assert got.dtype == np.int32, got.dtype
    assert np.array_equal(got, want), got
print("ok")
"#;
    let out = Command::new("python3")
        .arg("-c")
        .arg(py)
        .arg(&path)
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "h5py failed to read our int32:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn gold_h5py_opens_our_write() {
    let (path, _) = write_ours();
    let py = r#"
import h5py, numpy as np, sys
p = sys.argv[1]
with h5py.File(p, "r") as hf:
    d = hf["data"]
    got = np.array(d)
    want = np.array([[1.0,2.0,3.0],[4.0,5.0,6.0]])
    assert got.shape == (2,3), got.shape
    assert np.array_equal(got, want), got
    ver = d.attrs["hdf5-rust-version"]
    if isinstance(ver, bytes):
        ver = ver.decode()
    assert str(ver), ver
print("ok")
"#;
    let out = Command::new("python3")
        .arg("-c")
        .arg(py)
        .arg(&path)
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "h5py failed to read our file:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}

fn find_matlab() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("MATLAB") {
        let pb = PathBuf::from(p);
        if pb.exists() {
            return Some(pb);
        }
    }
    let which = Command::new("sh")
        .args(["-c", "command -v matlab"])
        .output()
        .ok()?;
    if which.status.success() {
        let s = String::from_utf8_lossy(&which.stdout).trim().to_string();
        if !s.is_empty() {
            return Some(PathBuf::from(s));
        }
    }
    for cand in [
        "/usr/local/MATLAB/R2026a/bin/matlab",
        "/usr/local/MATLAB/R2025a/bin/matlab",
        "/usr/local/MATLAB/R2024b/bin/matlab",
        "/usr/local/MATLAB/R2024a/bin/matlab",
        "/opt/MATLAB/R2026a/bin/matlab",
        "/opt/MATLAB/R2025a/bin/matlab",
        "/opt/MATLAB/R2024b/bin/matlab",
    ] {
        if Path::new(cand).exists() {
            return Some(PathBuf::from(cand));
        }
    }
    None
}

#[test]
#[ignore = "requires MATLAB on PATH or MATLAB=/path/to/matlab"]
fn gold_matlab_h5read_our_i32() {
    let matlab = find_matlab().expect(
        "MATLAB not found. Export MATLAB=/absolute/path/to/matlab (the binary, not the toolbox).",
    );
    let mut f = Hdf5File::create();
    f.write_i32("data", &[2, 3], &[0, 1, 2, 3, 4, 5])
        .expect("write");
    let path = interop_dir().join("ours_i32_ml.h5");
    f.save(&path).expect("save");
    let mfile = interop_dir().join("check_h5read_i32.m");
    let posix = path.display().to_string().replace('\\', "/");
    std::fs::write(
        &mfile,
        format!(
            "x = h5read('{posix}', '/data');\n\
             assert(isa(x, 'int32'));\n\
             assert(isequal(size(x), [2 3]) || isequal(size(x), [3 2]));\n\
             assert(isequal(sort(double(x(:))), (0:5)'));\n\
             disp('ok');\n"
        ),
    )
    .expect("mfile");
    let out = Command::new(&matlab)
        .args(["-batch", &format!("run('{}')", mfile.display())])
        .current_dir(interop_dir())
        .output()
        .expect("matlab spawn");
    assert!(
        out.status.success(),
        "MATLAB h5read int32 failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
#[ignore = "requires MATLAB on PATH or MATLAB=/path/to/matlab"]
fn gold_matlab_h5read_our_write() {
    let matlab = find_matlab().expect(
        "MATLAB not found. Export MATLAB=/absolute/path/to/matlab (the binary, not the toolbox).",
    );
    let (path, _) = write_ours();
    let mfile = interop_dir().join("check_h5read.m");
    let posix = path.display().to_string().replace('\\', "/");
    std::fs::write(
        &mfile,
        format!(
            "x = h5read('{posix}', '/data');\n\
             assert(isequal(size(x), [2 3]) || isequal(size(x), [3 2]));\n\
             v = h5readatt('{posix}', '/data', '{HDF5_WRITER_VERSION_ATTR}');\n\
             assert(~isempty(v));\n\
             disp('ok');\n"
        ),
    )
    .expect("mfile");
    let out = Command::new(&matlab)
        .args(["-batch", &format!("run('{}')", mfile.display())])
        .current_dir(interop_dir())
        .output()
        .expect("matlab spawn");
    assert!(
        out.status.success(),
        "MATLAB h5read failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}
