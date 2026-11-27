"""Tiny self-contained XDS recordings for running the public examples."""
from pathlib import Path

import h5py
import numpy as np


def write_recording(path: Path, *, dataset: str) -> Path:
    n_samples = 800
    ticks = np.arange(n_samples)
    target_field = "curs_v" if dataset == "chewie" else "force"
    with h5py.File(path, "w") as file:
        xds = file.create_group("xds")
        xds["bin_width"] = [[0.001]]
        xds["time_frame"] = (ticks * 0.001)[None, :]
        xds["spike_counts"] = np.stack((ticks % 4, (ticks + 1) % 5, (ticks + 2) % 6)).astype(np.uint16)
        for name, offset in (("curs_v", 0), ("force", 10)):
            xds[name] = np.stack((offset + ticks * 0.1, offset - ticks * 0.05))
        xds["unit_names"] = np.empty((3, 1), dtype=h5py.ref_dtype)
        refs = file.create_group("#refs#")
        for index, label in enumerate(("unit-a", "unit-b", "unit-c")):
            name = refs.create_dataset(str(index), data=np.frombuffer(label.encode("utf-16-le"), dtype="<u2")[:, None])
            name.attrs["MATLAB_class"] = np.bytes_("char")
            xds["unit_names"][index, 0] = name.ref
        xds["unit_names"].attrs["MATLAB_class"] = np.bytes_("cell")
        xds["trial_start_time"] = np.arange(10, dtype=float)[None, :] * 0.08
        xds["trial_end_time"] = np.arange(1, 11, dtype=float)[None, :] * 0.08
    return path
