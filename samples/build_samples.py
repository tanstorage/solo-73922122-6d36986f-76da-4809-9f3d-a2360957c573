#!/usr/bin/env python3
"""生成三份示例 ZIP，覆盖任务要求的关键场景。

用法（在仓库根目录执行）：
    .venv/bin/python -I samples/build_samples.py samples/data

生成：
    samples/data/shared_calib.zip   共享标定 + 包内相对外链 + 对象引用属性
    samples/data/cycles.zip         硬链接组循环 + 跨文件外链共享身份
    samples/data/same_path_diff.zip 两个源文件同路径、不同对象（禁止合并）
"""

from __future__ import annotations

import argparse
import os
import tempfile
import zipfile

import h5py
import numpy as np


def build_shared_calib(work: str) -> str:
    """仪器主文件通过两个外链入口共享同一份标定对象，并带引用属性。"""
    calib_path = os.path.join(work, "calibration.h5")
    inst_path = os.path.join(work, "instrument.h5")

    with h5py.File(calib_path, "w") as f:
        g = f.create_group("calib")
        g.create_dataset("coeffs", data=np.array([1.0, 2.0, 3.0], dtype="float32"))
        g.create_dataset("offset", data=np.array(0.25, dtype="float64"))
        g.attrs.create("version", "2026.11", dtype=h5py.string_dtype("utf-8"))
        g.attrs.create("serial", b"CAL-0001")  # 定长/ASCII 字符串
        g.attrs.create("tolerance", np.array([0.01, 0.02], dtype="float64"))
        # 标量对象引用属性 + 空引用属性
        g.attrs.create("coeff_ref", g["coeffs"].ref, dtype=h5py.ref_dtype)
        g.attrs.create("null_ref", h5py.Reference(), dtype=h5py.ref_dtype)

    with h5py.File(inst_path, "w") as f:
        m = f.create_group("measurements")
        m.create_dataset(
            "spectrum",
            data=np.arange(8, dtype="int16").reshape(2, 4),
        )
        m.create_dataset("temp", data=np.linspace(20.0, 21.0, 4, dtype="float32"))
        m.attrs.create("site", "lab-a", dtype=h5py.string_dtype("utf-8"))
        # 两个外链入口指向标定文件中的同一个组对象
        f["shared_calib"] = h5py.ExternalLink("calibration.h5", "/calib")
        f["calib_alias"] = h5py.ExternalLink("calibration.h5", "/calib")
        # 组内硬链接：同一数据集的两个入口
        m["spectrum_copy"] = m["spectrum"]
        # 标量对象引用属性（指向本组数据集）与空引用
        f.attrs.create("spectrum_ref", m["spectrum"].ref, dtype=h5py.ref_dtype)
        f.attrs.create("empty_ref", h5py.Reference(), dtype=h5py.ref_dtype)
        f.attrs.create("run_id", 42)

    return _zip(work, "shared_calib.zip", ["calibration.h5", "instrument.h5"])


def build_cycles(work: str) -> str:
    """硬链接形成的组循环，以及跨文件外链回到同一组。"""
    cyc_path = os.path.join(work, "cycles.h5")
    partner_path = os.path.join(work, "partner.h5")

    with h5py.File(cyc_path, "w") as f:
        g1 = f.create_group("g1")
        g2 = g1.create_group("g2")
        g2.create_dataset("values", data=np.array([10, 20, 30], dtype="int32"))
        # 循环：/g1/g2/back 指回 /g1；/g1/self 指向自身
        g2["back"] = g1
        g1["self"] = g1
        # 同一数据集的第二个硬链接入口
        g1["values_alias"] = g2["values"]
        g1.attrs.create("depth", 3)

    with h5py.File(partner_path, "w") as f:
        # 两个外链入口指向 cycles.h5 的同一个组，输出中必须共享身份
        f["via_ext"] = h5py.ExternalLink("cycles.h5", "/g1")
        f["via_ext_again"] = h5py.ExternalLink("cycles.h5", "/g1")
        f.attrs.create("note", "two entries, one object",
                       dtype=h5py.string_dtype("utf-8"))

    return _zip(work, "cycles.zip", ["cycles.h5", "partner.h5"])


def build_same_path_diff(work: str) -> str:
    """两个源文件含同路径对象，但值/类型不同，输出中不得合并。"""
    master_path = os.path.join(work, "master.h5")
    slave_path = os.path.join(work, "slave.h5")

    with h5py.File(master_path, "w") as f:
        g = f.create_group("sensors")
        g.create_dataset("channel_0", data=np.array([1, 2, 3], dtype="int32"))
        g.attrs.create("role", "master", dtype=h5py.string_dtype("utf-8"))

    with h5py.File(slave_path, "w") as f:
        g = f.create_group("sensors")
        g.create_dataset("channel_0", data=np.array([9.5], dtype="float64"))
        g.attrs.create("role", "slave", dtype=h5py.string_dtype("utf-8"))

    return _zip(work, "same_path_diff.zip", ["master.h5", "slave.h5"])


def _zip(work: str, zip_name: str, members: list[str]) -> str:
    out = os.path.join(work, zip_name)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in members:
            zf.write(os.path.join(work, name), arcname=name)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="生成示例 HDF5 ZIP 包")
    parser.add_argument("out_dir", help="示例 ZIP 的输出目录")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    for builder in (build_shared_calib, build_cycles, build_same_path_diff):
        with tempfile.TemporaryDirectory(prefix="sample_") as work:
            zip_path = builder(work)
            dest = os.path.join(args.out_dir, os.path.basename(zip_path))
            with open(zip_path, "rb") as src, open(dest, "wb") as dst:
                dst.write(src.read())
            print(f"written: {dest} ({os.path.getsize(dest)} bytes)")


if __name__ == "__main__":
    main()
