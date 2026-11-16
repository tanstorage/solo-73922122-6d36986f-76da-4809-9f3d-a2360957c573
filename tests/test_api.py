"""/merge HTTP 接口测试（FastAPI TestClient，不绑定端口）。"""

from __future__ import annotations

import io
import zipfile

import h5py
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import MAX_UPLOAD_BYTES, app

from conftest import addr


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _zip_bytes(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _h5_bytes(builder) -> bytes:
    buf = io.BytesIO()
    with h5py.File(buf, "w") as f:
        builder(f)
    return buf.getvalue()


def test_health_and_docs(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/").status_code == 200
    assert client.get("/docs").status_code == 200
    spec = client.get("/openapi.json").json()
    assert "/merge" in spec["paths"]


def test_merge_basic_roundtrip(client, tmp_path):
    payload = _zip_bytes({"a.h5": _h5_bytes(
        lambda f: f.create_dataset("d", data=np.array([1, 2, 3], dtype="i4"))
    )})
    resp = client.post(
        "/merge",
        files={"file": ("p.zip", payload, "application/zip")},
    )
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"] == "application/x-hdf5"
    assert "merged.h5" in resp.headers["content-disposition"]
    out = tmp_path / "out.h5"
    out.write_bytes(resp.content)
    with h5py.File(out, "r") as f:
        np.testing.assert_array_equal(f["/a/d"][...], [1, 2, 3])


def test_merge_hard_and_external_links(client, tmp_path):
    b = _h5_bytes(lambda f: (
        f.create_group("cal"),
        f["cal"].create_dataset("v", data=np.array([7.0], dtype="f4")),
    ))
    def build_a(f):
        g = f.create_group("m")
        ds = g.create_dataset("s", data=np.array([1], dtype="i2"))
        g["s2"] = ds
        f["link1"] = h5py.ExternalLink("b.h5", "/cal")
        f["link2"] = h5py.ExternalLink("b.h5", "/cal")
    a = _h5_bytes(build_a)
    resp = client.post("/merge", files={"file": ("p.zip", _zip_bytes(
        {"a.h5": a, "b.h5": b}
    ))})
    assert resp.status_code == 200, resp.text
    out = tmp_path / "out.h5"
    out.write_bytes(resp.content)
    with h5py.File(out, "r") as f:
        assert addr(f["/a/m/s"]) == addr(f["/a/m/s2"])
        assert addr(f["/a/link1"]) == addr(f["/b/cal"])
        assert addr(f["/a/link2"]) == addr(f["/b/cal"])


def test_reject_non_zip_extension(client):
    resp = client.post("/merge", files={"file": ("p.tar", b"x", "application/x-tar")})
    assert resp.status_code == 415


def test_reject_missing_file_field(client):
    resp = client.post("/merge", files={})
    assert resp.status_code == 422


def test_reject_corrupt_zip(client):
    resp = client.post(
        "/merge", files={"file": ("p.zip", b"PK not really a zip", "application/zip")}
    )
    assert resp.status_code == 400
    assert "ZIP" in resp.json()["detail"]


def test_reject_soft_link(client):
    def build(f):
        f.create_group("g")
        f["soft"] = h5py.SoftLink("/g")
    resp = client.post(
        "/merge", files={"file": ("p.zip", _zip_bytes({"a.h5": _h5_bytes(build)}))}
    )
    assert resp.status_code == 400
    assert "软链接" in resp.json()["detail"]


def test_reject_broken_external_link(client):
    def build(f):
        f["x"] = h5py.ExternalLink("gone.h5", "/g")
    resp = client.post(
        "/merge", files={"file": ("p.zip", _zip_bytes({"a.h5": _h5_bytes(build)}))}
    )
    assert resp.status_code == 400
    assert "缺失" in resp.json()["detail"]


def test_reject_oversized_upload(client):
    big = b"0" * (MAX_UPLOAD_BYTES + 1)
    resp = client.post(
        "/merge", files={"file": ("p.zip", big, "application/zip")}
    )
    assert resp.status_code == 413


def test_reject_uncompressed_size_over_limit(client, tmp_path):
    # 高压缩比数据：压缩后很小，但解压后超过 20 MiB
    big = np.zeros(20 * 1024 * 1024 + 100, dtype="uint8")
    src = tmp_path / "big.h5"
    with h5py.File(src, "w") as f:
        f.create_dataset("d", data=big)
    z = tmp_path / "p.zip"
    with zipfile.ZipFile(z, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(src, arcname="big.h5")
    compressed = z.read_bytes()
    assert len(compressed) < MAX_UPLOAD_BYTES  # 上传体积没超限
    resp = client.post(
        "/merge",
        files={"file": ("p.zip", compressed, "application/zip")},
    )
    assert resp.status_code == 400
    assert "20 MiB" in resp.json()["detail"]
