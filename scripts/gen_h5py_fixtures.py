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

print("wrote", list(out.glob("*.h5")))
