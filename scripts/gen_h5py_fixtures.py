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

print("wrote", sorted(p.name for p in out.glob("*.h5")))
