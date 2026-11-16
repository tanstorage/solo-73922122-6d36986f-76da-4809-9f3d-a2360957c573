"""pytest 共用夹具与构造辅助。"""

from __future__ import annotations

import zipfile

import h5py
import pytest


@pytest.fixture()
def work_dir(tmp_path):
    return tmp_path


def make_h5(path, builder):
    """在 path 创建 HDF5 文件，builder(h5py.File) 负责填充内容。"""
    with h5py.File(path, "w") as f:
        builder(f)


def zip_files(out_zip, members: dict[str, str]):
    """把 {归档名: 磁盘路径} 打成 ZIP。"""
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for arcname, disk in members.items():
            zf.write(disk, arcname=arcname)


def zip_from_bytes(out_zip, entries: dict[str, bytes]):
    """直接用字节内容构造 ZIP（用于伪造扩展名/魔数等场景）。"""
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_STORED) as zf:
        for arcname, data in entries.items():
            zf.writestr(arcname, data)


def addr(obj):
    return h5py.h5o.get_info(obj.id).addr


@pytest.fixture()
def helpers():
    class H:
        make_h5 = staticmethod(make_h5)
        zip_files = staticmethod(zip_files)
        zip_from_bytes = staticmethod(zip_from_bytes)
        addr = staticmethod(addr)

    return H
