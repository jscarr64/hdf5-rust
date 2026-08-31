//! Format golds. Values are bit patterns, not hardware floats.

use hdf5_rust::{
    fixture_chunked_dataset, HDF5DType, HDF5Error, Hdf5File, HDF5_SIGNATURE,
    HDF5_WRITER_VERSION_ATTR,
};

fn bits_u64(n: usize) -> Vec<u64> {
    (0..n as u64)
        .map(|i| 0x3FF0_0000_0000_0000u64.wrapping_add(i))
        .collect()
}

fn bits_u32(n: usize) -> Vec<u32> {
    (0..n as u32)
        .map(|i| 0x3F80_0000u32.wrapping_add(i))
        .collect()
}

fn roundtrip(file: &Hdf5File) -> Hdf5File {
    let bytes = file.to_bytes().expect("encode");
    assert_eq!(&bytes[..8], &HDF5_SIGNATURE);
    Hdf5File::from_bytes(&bytes).expect("decode")
}

#[test]
fn gold_ieee64_100x3_bit_identical() {
    let bits = bits_u64(300);
    let mut f = Hdf5File::create();
    f.write_f64("data", &[100, 3], &bits).expect("write");
    let g = roundtrip(&f);
    let (shape, got) = g.read_f64("data").expect("read");
    assert_eq!(shape, vec![100, 3]);
    assert_eq!(got, bits);
    assert_eq!(g.dataset_dtype("data").expect("dtype"), HDF5DType::Float64);
    let ver = g
        .read_attr_str("data", HDF5_WRITER_VERSION_ATTR)
        .expect("version attr");
    assert!(!ver.is_empty());
}

#[test]
fn gold_opaque_50x50() {
    let elem = 16usize;
    let n = 50 * 50;
    let mut data = vec![0u8; n * elem];
    for i in 0..n {
        data[i * elem] = (i % 251) as u8;
        data[i * elem + 1] = 1;
    }
    let mut f = Hdf5File::create();
    f.write_opaque("exact", &[50, 50], elem, &data)
        .expect("write");
    let g = roundtrip(&f);
    let (shape, sz, got) = g.read_opaque("exact").expect("read");
    assert_eq!(shape, vec![50, 50]);
    assert_eq!(sz, elem);
    assert_eq!(got, data);
    assert_eq!(
        g.dataset_dtype("exact").expect("dtype"),
        HDF5DType::Opaque(elem)
    );
}

#[test]
fn gold_ieee32_1d_1000_bit_identical() {
    let bits = bits_u32(1000);
    let mut f = Hdf5File::create();
    f.write_f32("vec", &[1000], &bits).expect("write");
    let g = roundtrip(&f);
    let (shape, got) = g.read_f32("vec").expect("read");
    assert_eq!(shape, vec![1000]);
    assert_eq!(got, bits);
}

#[test]
fn gold_nested_group() {
    let bits = bits_u64(6);
    let mut f = Hdf5File::create();
    f.create_group("results").expect("group");
    f.write_f64("results/data", &[2, 3], &bits).expect("write");
    let g = roundtrip(&f);
    let (shape, got) = g.read_f64("results/data").expect("read");
    assert_eq!(shape, vec![2, 3]);
    assert_eq!(got, bits);
    let groups = g.list_groups();
    assert!(groups.iter().any(|p| p == "results" || p == "/results"));
}

#[test]
fn gold_append_f64_10x3() {
    let a = bits_u64(30);
    let b = bits_u64(30)
        .into_iter()
        .map(|x| x.wrapping_add(1000))
        .collect::<Vec<_>>();
    let mut f = Hdf5File::create();
    f.write_f64("t", &[10, 3], &a).expect("write");
    f.append_f64("t", &[10, 3], &b).expect("append");
    let g = roundtrip(&f);
    let (shape, got) = g.read_f64("t").expect("read");
    assert_eq!(shape, vec![20, 3]);
    assert_eq!(&got[..30], a.as_slice());
    assert_eq!(&got[30..], b.as_slice());
}

#[test]
fn gold_list_three_datasets() {
    let mut f = Hdf5File::create();
    f.write_f64("a", &[2], &bits_u64(2)).expect("a");
    f.write_f64("b", &[2], &bits_u64(2)).expect("b");
    f.write_f64("c", &[2], &bits_u64(2)).expect("c");
    let g = roundtrip(&f);
    let mut names = g.list_datasets();
    names.sort();
    assert_eq!(names, vec!["a", "b", "c"]);
    assert_eq!(g.dataset_shape("b").expect("shape"), vec![2]);
}

#[test]
fn gold_error_invalid_signature() {
    let err = Hdf5File::from_bytes(b"not hdf5!!").expect_err("sig");
    assert_eq!(err, HDF5Error::InvalidSignature);
}

#[test]
fn gold_error_not_found() {
    let f = Hdf5File::create();
    match f.read_f64("missing") {
        Err(HDF5Error::NotFound(_)) => {}
        other => panic!("expected NotFound, got {other:?}"),
    }
}

#[test]
fn gold_error_shape_mismatch_append() {
    let mut f = Hdf5File::create();
    f.write_f64("t", &[10, 3], &bits_u64(30)).expect("write");
    match f.append_f64("t", &[10, 4], &bits_u64(40)) {
        Err(HDF5Error::ShapeMismatch) => {}
        other => panic!("expected ShapeMismatch, got {other:?}"),
    }
}

#[test]
fn gold_error_chunked() {
    let bytes = fixture_chunked_dataset().expect("fixture");
    let f = Hdf5File::from_bytes(&bytes).expect("parse chunked file");
    match f.read_f64("data") {
        Err(HDF5Error::ChunkedNotSupported) => {}
        other => panic!("expected ChunkedNotSupported, got {other:?}"),
    }
}

#[test]
fn gold_slice_rows() {
    let bits = bits_u64(30);
    let mut f = Hdf5File::create();
    f.write_f64("t", &[10, 3], &bits).expect("write");
    let g = roundtrip(&f);
    let (shape, got) = g.read_f64_slice("t", 2, 5).expect("slice");
    assert_eq!(shape, vec![3, 3]);
    assert_eq!(got, bits[6..15]);
}
