"""光谱 HDF5 单文件合并服务。

上传包含 1~3 个平铺 ``.h5`` 文件的 ZIP，服务把各文件根对象挂到输出文件的
``/<文件名>`` 下，保留形状、类型、数值与属性，并重建硬链接、包内相对外链与
对象引用；合并结果是一个完全自包含的 HDF5 文件，源包删除后仍可独立读取。
"""

from __future__ import annotations

import os
import shutil
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from .merger import MAX_TOTAL_BYTES, MergeError, merge_sources
from .package import PackageError, extract_package

# 原始上传字节上限：略大于解压上限，为 ZIP 容器开销留出余量。
MAX_UPLOAD_BYTES = 25 * 1024 * 1024

TAGS_METADATA = [
    {"name": "合并", "description": "上传 ZIP 并下载合并后的单件 HDF5。"},
    {"name": "运维", "description": "健康检查与接口元信息。"},
]

app = FastAPI(
    title="光谱 HDF5 合并服务",
    description=(
        "把仪器主文件与共享标定文件（1~3 个平铺 .h5，打包为 ZIP）"
        "合并为单件自包含 HDF5，供离线调整标定后复算。\n\n"
        "- 每个源文件根挂到输出的 `/<文件名>`；\n"
        "- 仅支持组、普通整数/浮点数据集，数值/字符串/标量对象引用属性；\n"
        "- 软链接及其他类型拒收；硬链接与包内相对外链重建为硬链接，保留身份与循环；\n"
        "- 外链缺目标或越出包边界则整包失败，不产生半成品。"
    ),
    version="1.0.0",
    openapi_tags=TAGS_METADATA,
)


@app.get("/health", tags=["运维"], summary="健康检查")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", tags=["运维"], summary="服务简介")
def index() -> JSONResponse:
    return JSONResponse(
        {
            "service": "光谱 HDF5 合并服务",
            "docs": "/docs",
            "openapi": "/openapi.json",
            "merge_endpoint": "POST /merge",
        }
    )


@app.post(
    "/merge",
    tags=["合并"],
    summary="合并 ZIP 内的 HDF5 文件，返回单件 HDF5",
    responses={
        200: {"description": "合并成功，返回 application/x-hdf5 文件流。"},
        400: {"description": "ZIP 结构、大小或 HDF5 内容不满足约束。"},
        413: {"description": "上传载荷超过 25 MiB。"},
        415: {"description": "上传字段不是 ZIP 文件。"},
        422: {"description": "缺少上传文件字段。"},
    },
)
async def merge(file: UploadFile = File(..., description="含 1~3 个平铺 .h5 的 ZIP")):
    if file.filename is None or not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=415, detail="仅接受 .zip 上传包")
    content_type = (file.content_type or "").lower()
    if content_type and "zip" not in content_type and content_type != "application/octet-stream":
        raise HTTPException(status_code=415, detail=f"Content-Type 不是 ZIP：{content_type}")

    work_dir = tempfile.mkdtemp(prefix="h5merge_")
    zip_path = os.path.join(work_dir, "upload.zip")
    output_path = os.path.join(work_dir, "merged.h5")
    try:
        # 先落盘并限制原始上传大小，避免超大请求耗尽内存/磁盘。
        written = 0
        with open(zip_path, "wb") as dst:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"上传载荷超过 {MAX_UPLOAD_BYTES} 字节上限（25 MiB）",
                    )
                dst.write(chunk)
        if written == 0:
            raise HTTPException(status_code=400, detail="上传内容为空")

        try:
            source_paths = extract_package(zip_path, work_dir, MAX_TOTAL_BYTES)
            merge_sources(source_paths, output_path, MAX_TOTAL_BYTES)
        except PackageError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except MergeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            # 理论上不会发生：合并失败应直接抛异常。守住最后一道防线。
            raise HTTPException(status_code=500, detail="合并未生成输出文件")

        return FileResponse(
            output_path,
            media_type="application/x-hdf5",
            filename="merged.h5",
            background=_cleanup_workdir(work_dir),
        )
    except HTTPException:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise
    except Exception:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise


def _cleanup_workdir(work_dir: str):
    """生成响应发送完成后执行的后台清理任务。

    starlette 的 BackgroundTask 接受同步可调用对象并在响应流结束后执行，
    因此这里直接返回删除临时目录的任务，确保下载完成后工作区不残留。
    """
    from starlette.background import BackgroundTask

    def _remove() -> None:
        shutil.rmtree(work_dir, ignore_errors=True)

    return BackgroundTask(_remove)
