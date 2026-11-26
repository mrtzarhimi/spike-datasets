"""Synthetic channel-major MATLAB v7.3-style XDS files."""

from pathlib import Path

import h5py
import numpy as np


def write_xds(
    path: Path,
    *,
    units: tuple[str, ...] = ("elec9", "elec2", "elec7"),
    include_force: bool = False,
    timestamps: np.ndarray | None = None,
    bin_width: float | np.floating = 0.001,
) -> Path:
    if timestamps is None:
        timestamps = np.arange(8) * 0.001
    ticks = np.arange(len(timestamps))
    with h5py.File(path, "w") as file:
        xds = file.create_group("xds")
        xds.attrs["MATLAB_class"] = np.bytes_("struct")
        xds["bin_width"] = np.asarray(bin_width).reshape(1, 1)
        xds["time_frame"] = timestamps[None, :]
        xds["spike_counts"] = (np.arange(len(units))[:, None] * 10 + ticks).astype(
            np.uint16
        )
        xds["curs_v"] = np.stack((ticks + 0.5, -ticks - 0.5))
        if include_force:
            target = xds.create_dataset("force", data=np.stack((100 + ticks, 200 + ticks)))
            target.attrs["units"] = np.bytes_("fixture force units")

        references = file.create_group("#refs#")
        names = xds.create_dataset("unit_names", (len(units), 1), dtype=h5py.ref_dtype)
        names.attrs["MATLAB_class"] = np.bytes_("cell")
        for i, unit in enumerate(units):
            text = references.create_dataset(str(i), data=_utf16(unit))
            text.attrs["MATLAB_class"] = np.bytes_("char")
            names[i, 0] = text.ref

        xds["trial_start_time"] = [[0.0, 0.003, 0.004]]
        xds["trial_end_time"] = [[0.002, 0.002, 0.008]]
        xds["trial_gocue_time"] = [[0.001, np.nan, 0.005]]
        outcome = xds.create_dataset("trial_result", data=_utf16("RFA"))
        outcome.attrs["MATLAB_class"] = np.bytes_("char")
        xds["trial_target_dir"] = [[0.0, 90.0, 180.0]]
    return path


def _utf16(value: str) -> np.ndarray:
    return np.frombuffer(value.encode("utf-16-le"), dtype="<u2")[:, None]
