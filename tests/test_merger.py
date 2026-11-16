"""合并核心 app.merger 的单元测试（不经过 Web 层）。"""

from __future__ import annotations

import os
import zipfile

import h5py
import numpy as np
import pytest

from app.merger import MAX_TOTAL_BYTES, MergeError, merge_sources
from app.package import PackageError, extract_package


def _extract_and_merge(zip_path, tmp_path, limit=MAX_TOTAL_BYTES):
    extract_dir = tmp_path / "extract"
    extract_dir.mkdir()
    sources = extract_package(str(zip_path), str(extract_dir))
    out = tmp_path / "merged.h5"
    merge_sources(sources, str(out), limit)
    return str(out)


def _build_basic(tmp_path, helpers):
    a = tmp_path / "a.h5"
    helpers.make_h5(
        str(a),
        lambda f: (
            f.create_group("g"),
            f["g"].create_dataset("d", data=np.arange(6, dtype="int32").reshape(2, 3)),
            f["g"].create_dataset("f", data=np.array([1.5, 2.5], dtype="float32")),
            f.attrs.create("run", 7),
        ),
    )
    z = tmp_path / "p.zip"
    helpers.zip_files(str(z), {"a.h5": str(a)})
    return z


def test_root_mount_and_value_preservation(tmp_path, helpers):
    z = _build_basic(tmp_path, helpers)
    out = _extract_and_merge(z, tmp_path)
    with h5py.File(out, "r") as f:
        assert "/a/g/d" in f and "/a/g/f" in f
        d = f["/a/g/d"]
        assert d.shape == (2, 3) and d.dtype == np.dtype("int32")
        np.testing.assert_array_equal(d[...], np.arange(6).reshape(2, 3))
        assert f["/a/g/f"].dtype == np.dtype("float32")
        assert f["/a"].attrs["run"] == 7


def test_dtype_variants_and_byteorder(tmp_path, helpers):
    a = tmp_path / "a.h5"
    def build(f):
        g = f.create_group("g")
        for i, dt in enumerate(["i1", "u8", "f8", np.dtype(">f4"), np.dtype("<i2")]):
            g.create_dataset(f"d{i}", data=np.ones((2,), dtype=dt))
        g.create_dataset("scalar", data=np.array(42, dtype="i4"))
        g.create_dataset("empty", shape=(0, 2), dtype="f4")
    helpers.make_h5(str(a), build)
    z = tmp_path / "p.zip"
    helpers.zip_files(str(z), {"a.h5": str(a)})
    out = _extract_and_merge(z, tmp_path)
    with h5py.File(out, "r") as f:
        g = f["/a/g"]
        for i, dt in enumerate(["i1", "u8", "f8", np.dtype(">f4"), np.dtype("<i2")]):
            ds = g[f"d{i}"]
            assert ds.dtype == np.dtype(dt), (ds.dtype, dt)
        assert g["scalar"].shape == () and g["scalar"][()] == 42
        assert g["empty"].shape == (0, 2)


def test_hard_link_identity_shared(tmp_path, helpers):
    a = tmp_path / "a.h5"
    def build(f):
        g = f.create_group("g")
        ds = g.create_dataset("d", data=np.array([1, 2, 3], dtype="i4"))
        g["d_alias"] = ds  # 数据集硬链接
        sub = g.create_group("sub")
        sub["back"] = g    # 组循环硬链接
        g["self"] = g
    helpers.make_h5(str(a), build)
    z = tmp_path / "p.zip"
    helpers.zip_files(str(z), {"a.h5": str(a)})
    out = _extract_and_merge(z, tmp_path)
    with h5py.File(out, "r+") as f:
        g = f["/a/g"]
        assert helpers.addr(g["d"]) == helpers.addr(g["d_alias"])
        assert helpers.addr(g["self"]) == helpers.addr(g)
        assert helpers.addr(g["sub/back"]) == helpers.addr(g)
        # 改一处，另一入口读到新值
        g["d"][0] = 99
        assert g["d_alias"][0] == 99


def test_group_cycle_terminates(tmp_path, helpers):
    a = tmp_path / "a.h5"
    def build(f):
        g1 = f.create_group("g1")
        g2 = g1.create_group("g2")
        g2.create_dataset("v", data=np.array([1], dtype="i4"))
        g2["back"] = g1
        g1["self"] = g1
    helpers.make_h5(str(a), build)
    z = tmp_path / "p.zip"
    helpers.zip_files(str(z), {"a.h5": str(a)})
    out = _extract_and_merge(z, tmp_path)
    with h5py.File(out, "r") as f:
        # visititems 必须能终止且不重复爆炸（返回相对名，无前导斜杠）
        names = []
        f.visititems(lambda n, o: names.append(n))
        assert "a/g1/g2/v" in names
        assert helpers.addr(f["/a/g1/self"]) == helpers.addr(f["/a/g1"])
        assert helpers.addr(f["/a/g1/g2/back"]) == helpers.addr(f["/a/g1"])


def test_soft_link_rejected(tmp_path, helpers):
    a = tmp_path / "a.h5"
    helpers.make_h5(
        str(a),
        lambda f: (f.create_group("g"), f.__setitem__("soft", h5py.SoftLink("/g"))),
    )
    z = tmp_path / "p.zip"
    helpers.zip_files(str(z), {"a.h5": str(a)})
    with pytest.raises(MergeError, match="软链接"):
        _extract_and_merge(z, tmp_path)
    assert not (tmp_path / "merged.h5").exists()


def test_unsupported_dataset_dtype_rejected(tmp_path, helpers):
    a = tmp_path / "a.h5"
    def build(f):
        f.create_dataset("s", data=np.array([b"x", b"yy"]))  # 字符串数据集
    helpers.make_h5(str(a), build)
    z = tmp_path / "p.zip"
    helpers.zip_files(str(z), {"a.h5": str(a)})
    with pytest.raises(MergeError, match="整数或浮点"):
        _extract_and_merge(z, tmp_path)


def test_unsupported_attr_types_rejected(tmp_path, helpers):
    a = tmp_path / "a.h5"
    def build(f):
        g = f.create_group("g")
        ds = g.create_dataset("d", data=np.array([1, 2], dtype="i4"))
        g.attrs.create("region", ds.regionref[:], dtype=h5py.regionref_dtype)
    helpers.make_h5(str(a), build)
    z = tmp_path / "p.zip"
    helpers.zip_files(str(z), {"a.h5": str(a)})
    with pytest.raises(MergeError, match="区域引用"):
        _extract_and_merge(z, tmp_path)


def test_ref_array_attr_rejected(tmp_path, helpers):
    a = tmp_path / "a.h5"
    def build(f):
        g = f.create_group("g")
        g.create_dataset("d1", data=np.array([1], dtype="i4"))
        g.create_dataset("d2", data=np.array([2], dtype="i4"))
        g.attrs.create(
            "refs", np.array([g["d1"].ref, g["d2"].ref]), dtype=h5py.ref_dtype
        )
    helpers.make_h5(str(a), build)
    z = tmp_path / "p.zip"
    helpers.zip_files(str(z), {"a.h5": str(a)})
    with pytest.raises(MergeError, match="引用数组"):
        _extract_and_merge(z, tmp_path)


def test_scalar_object_ref_and_null_ref(tmp_path, helpers):
    a = tmp_path / "a.h5"
    def build(f):
        g = f.create_group("g")
        d = g.create_dataset("d", data=np.array([3, 4], dtype="i4"))
        g.attrs.create("ref", d.ref, dtype=h5py.ref_dtype)
        g.attrs.create("null", h5py.Reference(), dtype=h5py.ref_dtype)
    helpers.make_h5(str(a), build)
    z = tmp_path / "p.zip"
    helpers.zip_files(str(z), {"a.h5": str(a)})
    out = _extract_and_merge(z, tmp_path)
    with h5py.File(out, "r+") as f:
        g = f["/a/g"]
        target = f[g.attrs["ref"]]
        assert helpers.addr(target) == helpers.addr(f["/a/g/d"])
        assert not bool(g.attrs["null"])
        # 经引用改值，直接入口同步
        target[0] = 77
        assert f["/a/g/d"][0] == 77


def test_string_and_numeric_attrs_preserved(tmp_path, helpers):
    a = tmp_path / "a.h5"
    def build(f):
        g = f.create_group("g")
        g.attrs.create("u8", "标定", dtype=h5py.string_dtype("utf-8"))
        g.attrs.create("ascii", b"CAL", dtype=h5py.string_dtype("ascii"))
        g.attrs.create("arr", np.array([1.0, 2.0], dtype="f8"))
        g.attrs.create("scalar_i", np.array(5, dtype="i2"))
    helpers.make_h5(str(a), build)
    z = tmp_path / "p.zip"
    helpers.zip_files(str(z), {"a.h5": str(a)})
    out = _extract_and_merge(z, tmp_path)
    with h5py.File(out, "r") as f:
        g = f["/a/g"]
        assert g.attrs["u8"] == "标定"
        assert g.attrs["ascii"] == "CAL"
        np.testing.assert_array_equal(g.attrs["arr"], [1.0, 2.0])
        assert g.attrs["scalar_i"] == 5
        assert g.attrs.get_id("scalar_i").dtype == np.dtype("i2")


def test_external_link_relative_shared_identity(tmp_path, helpers):
    b = tmp_path / "b.h5"
    a = tmp_path / "a.h5"
    helpers.make_h5(
        str(b),
        lambda f: (
            f.create_group("shared"),
            f["shared"].create_dataset("v", data=np.array([9], dtype="i4")),
        ),
    )
    def build_a(f):
        f["e1"] = h5py.ExternalLink("b.h5", "/shared")
        f["e2"] = h5py.ExternalLink("b.h5", "/shared")
    helpers.make_h5(str(a), build_a)
    z = tmp_path / "p.zip"
    helpers.zip_files(str(z), {"a.h5": str(a), "b.h5": str(b)})
    out = _extract_and_merge(z, tmp_path)
    with h5py.File(out, "r+") as f:
        canonical = f["/b/shared"]
        assert helpers.addr(f["/a/e1"]) == helpers.addr(canonical)
        assert helpers.addr(f["/a/e2"]) == helpers.addr(canonical)
        canonical["v"][0] = 55
        assert f["/a/e1/v"][0] == 55 and f["/a/e2/v"][0] == 55


def test_external_link_missing_target_fails(tmp_path, helpers):
    a = tmp_path / "a.h5"
    helpers.make_h5(
        str(a), lambda f: f.__setitem__("x", h5py.ExternalLink("gone.h5", "/g"))
    )
    z = tmp_path / "p.zip"
    helpers.zip_files(str(z), {"a.h5": str(a)})
    with pytest.raises(MergeError, match="不在包内或缺失"):
        _extract_and_merge(z, tmp_path)
    assert not (tmp_path / "merged.h5").exists()


def test_external_link_missing_object_fails(tmp_path, helpers):
    b = tmp_path / "b.h5"
    a = tmp_path / "a.h5"
    helpers.make_h5(str(b), lambda f: f.create_group("g"))
    helpers.make_h5(
        str(a), lambda f: f.__setitem__("x", h5py.ExternalLink("b.h5", "/nope"))
    )
    z = tmp_path / "p.zip"
    helpers.zip_files(str(z), {"a.h5": str(a), "b.h5": str(b)})
    with pytest.raises(MergeError, match="目标对象不存在"):
        _extract_and_merge(z, tmp_path)


def test_external_link_escapes_package_fails(tmp_path, helpers):
    a = tmp_path / "a.h5"
    helpers.make_h5(
        str(a), lambda f: f.__setitem__("x", h5py.ExternalLink("../outside.h5", "/g"))
    )
    z = tmp_path / "p.zip"
    helpers.zip_files(str(z), {"a.h5": str(a)})
    with pytest.raises(MergeError, match="越出包边界"):
        _extract_and_merge(z, tmp_path)


def test_external_link_absolute_path_fails(tmp_path, helpers):
    a = tmp_path / "a.h5"
    helpers.make_h5(
        str(a), lambda f: f.__setitem__("x", h5py.ExternalLink("/etc/secret.h5", "/g"))
    )
    z = tmp_path / "p.zip"
    helpers.zip_files(str(z), {"a.h5": str(a)})
    with pytest.raises(MergeError, match="绝对路径"):
        _extract_and_merge(z, tmp_path)


def test_same_path_different_objects_not_merged(tmp_path, helpers):
    m = tmp_path / "master.h5"
    s = tmp_path / "slave.h5"
    helpers.make_h5(
        str(m),
        lambda f: (
            f.create_group("sensors"),
            f["sensors"].create_dataset("c0", data=np.array([1, 2], dtype="i4")),
        ),
    )
    helpers.make_h5(
        str(s),
        lambda f: (
            f.create_group("sensors"),
            f["sensors"].create_dataset("c0", data=np.array([9.5], dtype="f8")),
        ),
    )
    z = tmp_path / "p.zip"
    helpers.zip_files(str(z), {"master.h5": str(m), "slave.h5": str(s)})
    out = _extract_and_merge(z, tmp_path)
    with h5py.File(out, "r") as f:
        dm = f["/master/sensors/c0"]
        ds = f["/slave/sensors/c0"]
        assert helpers.addr(dm) != helpers.addr(ds)
        assert dm.dtype == np.dtype("i4") and ds.dtype == np.dtype("f8")


def test_named_datatype_rejected(tmp_path, helpers):
    a = tmp_path / "a.h5"
    with h5py.File(str(a), "w") as f:
        t = h5py.h5t.TypeID.copy(h5py.h5t.NATIVE_INT32)
        t.commit(f.id, b"mytype")
    z = tmp_path / "p.zip"
    helpers.zip_files(str(z), {"a.h5": str(a)})
    with pytest.raises(MergeError):
        _extract_and_merge(z, tmp_path)


def test_too_many_files_rejected(tmp_path, helpers):
    z = tmp_path / "p.zip"
    entries = {}
    for i in range(4):
        buf = __import__("io").BytesIO()
        with h5py.File(buf, "w") as f:
            f.create_group("g")
        entries[f"f{i}.h5"] = buf.getvalue()
    helpers.zip_from_bytes(str(z), entries)
    with pytest.raises(PackageError, match="最多包含 3 个"):
        _extract_and_merge(z, tmp_path)


def test_non_h5_extension_rejected(tmp_path, helpers):
    z = tmp_path / "p.zip"
    helpers.zip_from_bytes(str(z), {"a.txt": b"hello"})
    with pytest.raises(PackageError, match="只接受 .h5"):
        _extract_and_merge(z, tmp_path)


def test_bad_h5_magic_rejected(tmp_path, helpers):
    z = tmp_path / "p.zip"
    helpers.zip_from_bytes(str(z), {"a.h5": b"not an hdf5 file at all"})
    with pytest.raises(PackageError, match="魔数"):
        _extract_and_merge(z, tmp_path)


def test_nested_path_rejected(tmp_path, helpers):
    a = tmp_path / "a.h5"
    helpers.make_h5(str(a), lambda f: f.create_group("g"))
    z = tmp_path / "p.zip"
    with zipfile.ZipFile(str(z), "w") as zf:
        zf.write(str(a), arcname="subdir/a.h5")
    with pytest.raises(PackageError, match="平铺"):
        _extract_and_merge(z, tmp_path)


def test_directory_entry_rejected(tmp_path, helpers):
    z = tmp_path / "p.zip"
    with zipfile.ZipFile(str(z), "w") as zf:
        zf.writestr("dir/", "")
    with pytest.raises(PackageError, match="目录"):
        _extract_and_merge(z, tmp_path)


def test_total_size_limit_enforced(tmp_path, helpers):
    a = tmp_path / "a.h5"
    helpers.make_h5(
        str(a),
        lambda f: f.create_dataset("big", data=np.zeros(100, dtype="f8")),
    )
    z = tmp_path / "p.zip"
    helpers.zip_files(str(z), {"a.h5": str(a)})
    with pytest.raises(MergeError, match="超过上限"):
        _extract_and_merge(z, tmp_path, limit=10)


def test_duplicate_stem_rejected(tmp_path, helpers, monkeypatch):
    a = tmp_path / "a.h5"
    b = tmp_path / "b.h5"
    helpers.make_h5(str(a), lambda f: f.create_group("g"))
    helpers.make_h5(str(b), lambda f: f.create_group("h"))
    # 入口按文件名主干去重；用 monkeypatch 强制两个文件得到相同主干，
    # 覆盖根组名冲突的校验分支。
    import app.merger as merger_mod

    def fake_splitext(path):
        return (os.path.join(os.path.dirname(path), "same"), ".h5")

    monkeypatch.setattr(merger_mod.os.path, "splitext", fake_splitext)
    with pytest.raises(MergeError, match="根组名冲突"):
        merge_sources([str(a), str(b)], str(tmp_path / "o.h5"))
    assert not (tmp_path / "o.h5").exists()


def test_output_self_contained_after_source_removed(tmp_path, helpers):
    a = tmp_path / "a.h5"
    helpers.make_h5(
        str(a),
        lambda f: f.create_dataset("d", data=np.arange(100, dtype="f4")),
    )
    z = tmp_path / "p.zip"
    helpers.zip_files(str(z), {"a.h5": str(a)})
    out = _extract_and_merge(z, tmp_path)
    downloaded = tmp_path / "downloaded.h5"
    os.rename(out, downloaded)
    # 解压目录与原始 h5 全部删除后，下载文件仍可完整读取
    import shutil

    shutil.rmtree(tmp_path / "extract", ignore_errors=True)
    a.unlink()
    with h5py.File(str(downloaded), "r") as f:
        np.testing.assert_array_equal(f["/a/d"][...], np.arange(100, dtype="f4"))


def test_three_files_and_root_attrs(tmp_path, helpers):
    z = tmp_path / "p.zip"
    paths = {}
    for n in ("a", "b", "c"):
        p = tmp_path / f"{n}.h5"
        helpers.make_h5(
            str(p),
            lambda f, n=n: (
                f.attrs.create("root_attr", n),
                f.create_dataset("d", data=np.array([ord(n)], dtype="i4")),
            ),
        )
        paths[f"{n}.h5"] = str(p)
    helpers.zip_files(str(z), paths)
    out = _extract_and_merge(z, tmp_path)
    with h5py.File(out, "r") as f:
        assert set(f.keys()) == {"a", "b", "c"}
        for n in ("a", "b", "c"):
            assert f[f"/{n}"].attrs["root_attr"] == n
            assert f[f"/{n}/d"][...] == ord(n)


def test_nested_external_links_resolved(tmp_path, helpers):
    """a 外链到 b，b 该位置又是外链到 c 的对象，最终应解析到 c 的规范对象。"""
    c = tmp_path / "c.h5"
    b = tmp_path / "b.h5"
    a = tmp_path / "a.h5"
    helpers.make_h5(
        str(c),
        lambda f: f.create_dataset("leaf", data=np.array([5, 6], dtype="i4")),
    )
    with h5py.File(str(b), "w") as f:
        f["mid"] = h5py.ExternalLink("c.h5", "/leaf")
    with h5py.File(str(a), "w") as f:
        f["entry"] = h5py.ExternalLink("b.h5", "/mid")
    z = tmp_path / "p.zip"
    helpers.zip_files(str(z), {"a.h5": str(a), "b.h5": str(b), "c.h5": str(c)})
    out = _extract_and_merge(z, tmp_path)
    with h5py.File(out, "r") as f:
        assert helpers.addr(f["/a/entry"]) == helpers.addr(f["/c/leaf"])
        np.testing.assert_array_equal(f["/a/entry"][...], [5, 6])


def test_external_link_to_dataset(tmp_path, helpers):
    b = tmp_path / "b.h5"
    a = tmp_path / "a.h5"
    helpers.make_h5(
        str(b), lambda f: f.create_dataset("d", data=np.array([3.0], dtype="f4"))
    )
    with h5py.File(str(a), "w") as f:
        f["alias"] = h5py.ExternalLink("b.h5", "/d")
    z = tmp_path / "p.zip"
    helpers.zip_files(str(z), {"a.h5": str(a), "b.h5": str(b)})
    out = _extract_and_merge(z, tmp_path)
    with h5py.File(out, "r") as f:
        assert helpers.addr(f["/a/alias"]) == helpers.addr(f["/b/d"])


def test_cross_file_object_reference_attribute(tmp_path, helpers):
    """对象引用属性借外链指向另一文件对象时，输出引用应指向对应规范对象。"""
    b = tmp_path / "b.h5"
    a = tmp_path / "a.h5"
    with h5py.File(str(b), "w") as f:
        f.create_group("shared")
        f["shared"].create_dataset("v", data=np.array([2], dtype="i4"))
    with h5py.File(str(a), "w") as f:
        f["ext"] = h5py.ExternalLink("b.h5", "/shared/v")
        f.attrs.create("cross_ref", f["ext"].ref, dtype=h5py.ref_dtype)
    z = tmp_path / "p.zip"
    helpers.zip_files(str(z), {"a.h5": str(a), "b.h5": str(b)})
    out = _extract_and_merge(z, tmp_path)
    with h5py.File(out, "r") as f:
        target = f[f["/a"].attrs["cross_ref"]]
        assert helpers.addr(target) == helpers.addr(f["/b/shared/v"])


def test_hidden_file_rejected(tmp_path, helpers):
    z = tmp_path / "p.zip"
    helpers.zip_from_bytes(str(z), {".DS_Store": b"junk"})
    with pytest.raises(PackageError, match="隐藏"):
        _extract_and_merge(z, tmp_path)
