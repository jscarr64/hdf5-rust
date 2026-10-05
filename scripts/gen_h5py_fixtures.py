#!/usr/bin/env python3
"""Regenerate tests/fixtures/*.h5 from h5py. Run from the crate root."""
from pathlib import Path
import h5py
import numpy as np

out = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
out.mkdir(parents=True, exist_ok=True)

with h5py.File(out / "h5py_f64_default.h5", "w") as f:
    f.create_dataset(
        "data",
        data=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64),
    )

with h5py.File(out / "h5py_f64_attr.h5", "w", libver="latest") as f:
    ds = f.create_dataset("data", data=np.arange(12, dtype=np.float64).reshape(4, 3))
    ds.attrs["units"] = "kelvin"

with h5py.File(out / "h5py_three.h5", "w") as f:
    f.create_dataset("a", data=np.array([1.0, 2.0], dtype=np.float64))
    f.create_dataset("b", data=np.array([3.0, 4.0], dtype=np.float64))
    f.create_dataset("c", data=np.array([5.0, 6.0], dtype=np.float64))

with h5py.File(out / "h5py_chunked.h5", "w") as f:
    f.create_dataset(
        "data",
        data=np.arange(20, dtype=np.float64).reshape(5, 4),
        chunks=(2, 4),
    )

with h5py.File(out / "h5py_int32.h5", "w") as f:
    f.create_dataset("data", data=np.arange(6, dtype=np.int32).reshape(2, 3))

with h5py.File(out / "h5py_rank3.h5", "w") as f:
    f.create_dataset("cube", data=np.arange(24, dtype=np.float64).reshape(2, 3, 4))

with h5py.File(out / "h5py_rank4.h5", "w") as f:
    f.create_dataset("vol", data=np.arange(48, dtype=np.float64).reshape(2, 2, 3, 4))

with h5py.File(out / "h5py_rank5.h5", "w") as f:
    f.create_dataset(
        "stack", data=np.arange(48, dtype=np.int32).reshape(2, 2, 2, 2, 3)
    )

with h5py.File(out / "h5py_mixed.h5", "w") as f:
    f.create_dataset(
        "table",
        data=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64),
    )
    f.create_dataset("cube", data=np.arange(24, dtype=np.float64).reshape(2, 3, 4))

with h5py.File(out / "h5py_gzip.h5", "w") as f:
    f.create_dataset(
        "data",
        data=np.arange(20, dtype=np.float64).reshape(5, 4),
        compression="gzip",
    )

with h5py.File(out / "h5py_complex.h5", "w") as f:
    f.create_dataset(
        "data",
        data=np.array([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]], dtype=np.complex128),
    )

with h5py.File(out / "h5py_gzip_chunked.h5", "w") as f:
    f.create_dataset(
        "data",
        data=np.arange(20, dtype=np.float64).reshape(5, 4),
        chunks=(2, 4),
        compression="gzip",
        compression_opts=4,
    )

with h5py.File(out / "h5py_layout_v4_chunked.h5", "w", libver="latest") as f:
    f.create_dataset(
        "data",
        data=np.arange(20, dtype=np.float64).reshape(5, 4),
        chunks=(2, 4),
    )

with h5py.File(out / "h5py_layout_v4_single.h5", "w", libver="latest") as f:
    f.create_dataset(
        "data",
        data=np.arange(12, dtype=np.float64).reshape(3, 4),
        chunks=(3, 4),
    )

with h5py.File(out / "h5py_layout_v4_gzip.h5", "w", libver="latest") as f:
    f.create_dataset(
        "data",
        data=np.arange(20, dtype=np.float64).reshape(5, 4),
        chunks=(2, 4),
        compression="gzip",
    )

dt = np.dtype([("x", "<i4"), ("y", "<f8")])
arr = np.zeros(3, dtype=dt)
arr["x"] = [1, 2, 3]
arr["y"] = [1.5, 2.5, 3.5]
with h5py.File(out / "h5py_compound.h5", "w") as f:
    f.create_dataset("points", data=arr)

with h5py.File(out / "h5py_f64be.h5", "w") as f:
    f.create_dataset("data", data=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=">f8"))

with h5py.File(out / "h5py_i32be.h5", "w") as f:
    f.create_dataset("data", data=np.array([[1, 2, 3], [4, 5, 6]], dtype=">i4"))

# Live files from this h5py/HDF5, not hand-built buffers.
# Extensible array: one unlimited dimension. B-tree v2: two unlimited dimensions.


def low_level(path, data, chunks, setup_dcpl, dtype=None):
    """Create one chunked dataset with a custom creation property list."""
    data = np.ascontiguousarray(data)
    tid = dtype if dtype is not None else h5py.h5t.py_create(data.dtype)
    with h5py.File(path, "w", libver="latest") as f:
        dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
        dcpl.set_chunk(chunks)
        setup_dcpl(dcpl)
        space = h5py.h5s.create_simple(data.shape, data.shape)
        dset = h5py.h5d.create(f.id, b"data", tid, space, dcpl)
        dset.write(h5py.h5s.ALL, h5py.h5s.ALL, data)


with h5py.File(out / "h5py_earray_small.h5", "w", libver="latest") as f:
    f.create_dataset(
        "data", data=np.arange(3, dtype=np.int32), chunks=(1,), maxshape=(None,)
    )

with h5py.File(out / "h5py_earray_blocks.h5", "w", libver="latest") as f:
    f.create_dataset(
        "data", data=np.arange(100, dtype=np.int32), chunks=(1,), maxshape=(None,)
    )

with h5py.File(out / "h5py_earray_super.h5", "w", libver="latest") as f:
    f.create_dataset(
        "data", data=np.arange(400, dtype=np.int32), chunks=(1,), maxshape=(None,)
    )

with h5py.File(out / "h5py_earray_2d.h5", "w", libver="latest") as f:
    f.create_dataset(
        "data",
        data=np.arange(48, dtype=np.int32).reshape(6, 8),
        chunks=(2, 2),
        maxshape=(None, 8),
    )

with h5py.File(out / "h5py_earray_gzip.h5", "w", libver="latest") as f:
    f.create_dataset(
        "data",
        data=np.arange(16, dtype=np.int32),
        chunks=(4,),
        maxshape=(None,),
        compression="gzip",
        shuffle=True,
        compression_opts=4,
    )

with h5py.File(out / "h5py_bt2_small.h5", "w", libver="latest") as f:
    f.create_dataset(
        "data",
        data=np.arange(4, dtype=np.int32).reshape(2, 2),
        chunks=(1, 1),
        maxshape=(None, None),
    )

with h5py.File(out / "h5py_bt2_internal.h5", "w", libver="latest") as f:
    f.create_dataset(
        "data",
        data=np.arange(100, dtype=np.int32).reshape(10, 10),
        chunks=(1, 1),
        maxshape=(None, None),
    )

with h5py.File(out / "h5py_bt2_depth2.h5", "w", libver="latest") as f:
    f.create_dataset(
        "data",
        data=np.arange(73 * 73).astype(np.uint8).reshape(73, 73),
        chunks=(1, 1),
        maxshape=(None, None),
    )

with h5py.File(out / "h5py_bt2_gzip.h5", "w", libver="latest") as f:
    f.create_dataset(
        "data",
        data=np.arange(16, dtype=np.int32).reshape(4, 4),
        chunks=(2, 2),
        maxshape=(None, None),
        compression="gzip",
        shuffle=True,
    )

with h5py.File(out / "h5py_bt2_fixeddim.h5", "w", libver="latest") as f:
    f.create_dataset(
        "data",
        data=np.arange(32, dtype=np.int32).reshape(4, 4, 2),
        chunks=(2, 2, 2),
        maxshape=(None, None, 2),
    )

low_level(
    out / "h5py_implicit.h5",
    np.arange(32, dtype=np.int32).reshape(8, 4),
    (2, 4),
    lambda dcpl: dcpl.set_alloc_time(h5py.h5d.ALLOC_TIME_EARLY),
)

low_level(
    out / "h5py_shuffle.h5",
    np.arange(8, dtype=np.int32),
    (4,),
    lambda dcpl: dcpl.set_filter(h5py.h5z.FILTER_SHUFFLE, h5py.h5z.FLAG_OPTIONAL, (4,)),
)

with h5py.File(out / "h5py_fletcher32.h5", "w", libver="latest") as f:
    f.create_dataset(
        "data",
        data=np.arange(8, dtype=np.int32),
        chunks=(4,),
        fletcher32=True,
    )

with h5py.File(out / "h5py_fletcher_shuffle_gzip.h5", "w", libver="latest") as f:
    f.create_dataset(
        "data",
        data=np.arange(8, dtype=np.int32),
        chunks=(4,),
        compression="gzip",
        shuffle=True,
        fletcher32=True,
    )

nbit_dt = h5py.h5t.NATIVE_INT32.copy()
nbit_dt.set_precision(12)
nbit_dt.set_offset(0)
low_level(
    out / "h5py_nbit.h5",
    np.array([0, 1, 100, -5, 2047], dtype=np.int32),
    (5,),
    lambda dcpl: dcpl.set_filter(h5py.h5z.FILTER_NBIT, h5py.h5z.FLAG_OPTIONAL, ()),
    dtype=nbit_dt,
)

low_level(
    out / "h5py_scaleoffset_i32.h5",
    np.array([10, 12, 15, 11, -3, -1, 4, 8], dtype=np.int32),
    (4,),
    lambda dcpl: dcpl.set_filter(
        h5py.h5z.FILTER_SCALEOFFSET, h5py.h5z.FLAG_OPTIONAL, (2, 0)
    ),
)

low_level(
    out / "h5py_scaleoffset_f64.h5",
    np.array([1.5, 2.25, 3.125, 4.0], dtype=np.float64),
    (4,),
    lambda dcpl: dcpl.set_filter(
        h5py.h5z.FILTER_SCALEOFFSET, h5py.h5z.FLAG_OPTIONAL, (0, 2)
    ),
)

try:
    with h5py.File(out / "h5py_szip.h5", "w", libver="latest") as f:
        f.create_dataset(
            "data",
            data=np.arange(16, dtype=np.int32),
            chunks=(8,),
            compression="szip",
            compression_opts=("nn", 8),
        )
except Exception as exc:
    print("szip fixture skipped:", exc)

print("wrote", sorted(p.name for p in out.glob("*.h5")))
