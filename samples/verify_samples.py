#!/usr/bin/env python3
"""对示例 ZIP 跑合并并断言关键性质。

默认直接调用本地合并核心（无需启动服务）；加 --url 时改为通过 HTTP API 验证：

    .venv/bin/python -I samples/verify_samples.py samples/data
    .venv/bin/python -I samples/verify_samples.py samples/data --url http://127.0.0.1:8000

检查项：
    1. 根挂载路径 /<文件名> 存在，形状/类型/数值/属性保留；
    2. 硬链接与外链的多个入口共享同一对象身份；
    3. 组循环在输出中可遍历且会终止；
    4. 对象引用属性指向输出内对象，空引用仍为空；
    5. 不同源文件的同路径异对象未被合并；
    6. 改一处标定值，其他硬链接入口与引用读到的都是新值；
    7. 输出文件自包含（把源 HDF5 移走后仍可读取）。
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile

import h5py
import numpy as np

# 允许从仓库根直接运行本脚本。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.merger import merge_sources  # noqa: E402
from app.package import extract_package  # noqa: E402


def addr(obj: h5py.HLObject) -> int:
    return h5py.h5o.get_info(obj.id).addr


def merge_zip_locally(zip_path: str, out_dir: str) -> str:
    extract_dir = tempfile.mkdtemp(dir=out_dir)
    sources = extract_package(zip_path, extract_dir)
    output_path = os.path.join(out_dir, "merged.h5")
    merge_sources(sources, output_path)
    shutil.rmtree(extract_dir, ignore_errors=True)
    return output_path


def merge_zip_via_http(zip_path: str, url: str, out_dir: str) -> str:
    import httpx

    with open(zip_path, "rb") as fh:
        resp = httpx.post(
            f"{url.rstrip('/')}/merge",
            files={"file": (os.path.basename(zip_path), fh, "application/zip")},
            timeout=60.0,
        )
    if resp.status_code != 200:
        raise AssertionError(f"HTTP {resp.status_code}: {resp.text}")
    output_path = os.path.join(out_dir, "merged.h5")
    with open(output_path, "wb") as dst:
        dst.write(resp.content)
    return output_path


def check_shared_calib(path: str) -> None:
    with h5py.File(path, "r+") as f:
        assert "/calibration/calib/coeffs" in f
        assert "/instrument/shared_calib/coeffs" in f
        assert "/instrument/calib_alias/coeffs" in f
        coeff = f["/calibration/calib/coeffs"]
        a1 = f["/instrument/shared_calib/coeffs"]
        a2 = f["/instrument/calib_alias/coeffs"]
        assert addr(coeff) == addr(a1) == addr(a2), "外链两入口未共享身份"
        assert a1.dtype == np.dtype("float32") and a1.shape == (3,)
        np.testing.assert_array_equal(a1[...], [1.0, 2.0, 3.0])

        # 硬链接身份
        s1 = f["/instrument/measurements/spectrum"]
        s2 = f["/instrument/measurements/spectrum_copy"]
        assert addr(s1) == addr(s2), "硬链接未保持身份"
        assert s1.dtype == np.dtype("int16") and s1.shape == (2, 4)

        # 标量与数组数值/字符串属性（变长 ASCII/UTF-8 回读均为 str）
        cg = f["/calibration/calib"]
        assert cg.attrs["version"] == "2026.11"
        assert cg.attrs["serial"] == "CAL-0001"
        np.testing.assert_array_equal(cg.attrs["tolerance"], [0.01, 0.02])
        assert f["/instrument"].attrs["run_id"] == 42

        # 引用属性：指向输出内对应对象；空引用仍为空
        ref_obj = f[cg.attrs["coeff_ref"]]
        assert addr(ref_obj) == addr(coeff), "引用属性未指向输出内对应对象"
        spec_ref = f[f["/instrument"].attrs["spectrum_ref"]]
        assert addr(spec_ref) == addr(s1)
        assert not bool(cg.attrs["null_ref"]), "空引用未保持为空"
        assert not bool(f["/instrument"].attrs["empty_ref"])

        # 核心业务性质：离线调整标定后，其他入口与引用读到新值
        coeff[0] = 99.0
        assert a1[0] == 99.0 and a2[0] == 99.0, "其他硬链接入口未读到新值"
        via_ref = f[cg.attrs["coeff_ref"]]
        assert via_ref[0] == 99.0, "引用入口未读到新值"


def check_cycles(path: str) -> None:
    with h5py.File(path, "r") as f:
        g1 = f["/cycles/g1"]
        assert addr(g1["self"]) == addr(g1), "自循环硬链接身份错误"
        assert addr(g1["g2/back"]) == addr(g1), "组循环硬链接身份错误"
        v1 = g1["g2/values"]
        v2 = g1["values_alias"]
        assert addr(v1) == addr(v2)
        # 跨文件外链的两个入口也指向同一输出对象
        e1 = f["/partner/via_ext"]
        e2 = f["/partner/via_ext_again"]
        assert addr(e1) == addr(e2) == addr(g1), "跨文件外链未共享身份"

        # 显式栈可终止地遍历循环组
        seen: set[int] = set()
        stack = [g1]
        steps = 0
        while stack and steps < 1000:
            cur = stack.pop()
            steps += 1
            a = addr(cur)
            if a in seen:
                continue
            seen.add(a)
            for child in cur.values():
                if isinstance(child, h5py.Group):
                    stack.append(child)
        assert steps < 1000, "循环遍历未终止"
        assert g1.attrs["depth"] == 3


def check_same_path_diff(path: str) -> None:
    with h5py.File(path, "r") as f:
        m = f["/master/sensors/channel_0"]
        s = f["/slave/sensors/channel_0"]
        assert addr(m) != addr(s), "同路径异对象被错误合并"
        assert m.dtype == np.dtype("int32") and list(m[...]) == [1, 2, 3]
        assert s.dtype == np.dtype("float64") and s[...] == 9.5
        assert f["/master/sensors"].attrs["role"] == "master"
        assert f["/slave/sensors"].attrs["role"] == "slave"


def check_self_contained(path: str) -> None:
    with h5py.File(path, "r") as f:
        # 完整读出所有数据集，确认没有任何外部存储依赖
        def walk(name: str, obj: h5py.HLObject) -> None:
            if isinstance(obj, h5py.Dataset):
                _ = obj[...]
        f.visititems(walk)
        assert f["/instrument/shared_calib/coeffs"].shape == (3,)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="build_samples.py 生成的 ZIP 所在目录")
    parser.add_argument("--url", default=None, help="若提供则走 HTTP API，例如 http://127.0.0.1:8000")
    args = parser.parse_args()

    cases = {
        "shared_calib.zip": check_shared_calib,
        "cycles.zip": check_cycles,
        "same_path_diff.zip": check_same_path_diff,
    }

    with tempfile.TemporaryDirectory(prefix="verify_") as out_dir:
        for zip_name, checker in cases.items():
            zip_path = os.path.join(args.data_dir, zip_name)
            if not os.path.exists(zip_path):
                raise SystemExit(f"缺少示例文件：{zip_path}，请先运行 build_samples.py")
            if args.url:
                merged = merge_zip_via_http(zip_path, args.url, out_dir)
            else:
                merged = merge_zip_locally(zip_path, out_dir)
            checker(merged)
            if zip_name == "shared_calib.zip":
                check_self_contained(merged)
            os.remove(merged)
            print(f"PASS {zip_name}")

    # 自包含性额外验证：合并完成后把所有源文件移走，再打开下载文件仍可读
    with tempfile.TemporaryDirectory(prefix="standalone_") as tmp:
        zip_path = os.path.join(args.data_dir, "shared_calib.zip")
        merged = merge_zip_locally(zip_path, tmp) if not args.url else \
            merge_zip_via_http(zip_path, args.url, tmp)
        moved = os.path.join(tmp, "downloaded.h5")
        os.rename(merged, moved)
        with h5py.File(moved, "r") as f:
            assert f["/calibration/calib/coeffs"][...].sum() > 0
        print("PASS standalone-read-after-source-removed")

    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
