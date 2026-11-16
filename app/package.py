"""上传 ZIP 包的解压与静态校验。

约束（与任务说明一致）：

* 必须是合法 ZIP；
* 包内为 1~3 个平铺的 ``.h5`` 普通文件，不接受目录条目、嵌套路径、其他后缀；
* 解压后所有文件合计不超过 20 MiB（按解压后真实字节数计量，防压缩炸弹）；
* 文件名主干唯一，避免输出根组重名。

解压目标始终是调用方提供的全新空目录；本模块只负责往里面写文件。
"""

from __future__ import annotations

import os
import zipfile

from .merger import MAX_TOTAL_BYTES

_HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"


class PackageError(Exception):
    """ZIP 包结构或大小不满足要求时抛出（对应 HTTP 4xx）。"""


def extract_package(zip_path: str, dest_dir: str, max_total_bytes: int = MAX_TOTAL_BYTES) -> list[str]:
    """校验并解压 ZIP，返回解压出的源文件绝对路径列表（按文件名排序）。

    :raises PackageError: ZIP 非法、条目不合规或超出大小限制。
    """
    if not zipfile.is_zipfile(zip_path):
        raise PackageError("上传内容不是合法的 ZIP 文件")

    extracted: list[str] = []
    total = 0
    try:
        zf = zipfile.ZipFile(zip_path, "r")
    except zipfile.BadZipFile as exc:
        raise PackageError(f"无法读取 ZIP：{exc}") from exc

    with zf:
        # 先做全部条目的静态检查，再动手解压，避免边写边回滚。
        infos = zf.infolist()
        if not infos:
            raise PackageError("ZIP 包为空")
        if len(infos) > 3:
            raise PackageError(f"每个包最多包含 3 个文件，实际 {len(infos)} 个")

        seen_names: set[str] = set()
        seen_stems: set[str] = set()
        for info in infos:
            name = info.filename
            if name.endswith("/") or info.is_dir():
                raise PackageError(f"不接受目录条目，包内必须平铺：{name}")
            # 平铺：任何路径分隔符（含 Windows 反斜杠）都视为嵌套。
            if "/" in name or "\\" in name or os.path.basename(name) != name:
                raise PackageError(f"不接受嵌套路径，包内必须平铺：{name}")
            if name in (".", "..") or name.startswith("."):
                raise PackageError(f"不接受隐藏或特殊文件名：{name}")
            if not name.lower().endswith(".h5"):
                raise PackageError(f"只接受 .h5 文件，发现：{name}")
            if name in seen_names:
                raise PackageError(f"包内文件名重复：{name}")
            seen_names.add(name)
            stem = os.path.splitext(name)[0]
            if stem in seen_stems:
                raise PackageError(f"文件名主干重复，输出根组会冲突：{name}")
            seen_stems.add(stem)

            # 以解压后真实大小计量，压缩炸弹在这一步被拦下。
            if info.file_size < 0:
                raise PackageError(f"条目大小非法：{name}")
            total += info.file_size
            if total > max_total_bytes:
                raise PackageError(
                    f"解压后合计 {total} 字节，超过上限 {max_total_bytes} 字节（20 MiB）"
                )

        dest_real = os.path.realpath(dest_dir)
        for info in infos:
            target = os.path.realpath(os.path.join(dest_real, info.filename))
            if target != os.path.join(dest_real, info.filename) or os.path.dirname(
                target
            ) != dest_real:
                raise PackageError(f"检测到越界路径（zip-slip）：{info.filename}")
            # 流式解压并再次按真实写入字节计数，双重保险。
            with zf.open(info, "r") as src, open(target, "wb") as dst:
                written = 0
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > max_total_bytes:
                        dst.close()
                        os.remove(target)
                        raise PackageError(
                            f"解压数据超出 {max_total_bytes} 字节上限（20 MiB）"
                        )
                    dst.write(chunk)
            if not _looks_like_hdf5(target):
                raise PackageError(f"不是有效的 HDF5 文件（魔数校验失败）：{info.filename}")
            extracted.append(target)

    extracted.sort(key=lambda p: os.path.basename(p))
    return extracted


def _looks_like_hdf5(path: str) -> bool:
    try:
        with open(path, "rb") as fh:
            return fh.read(8) == _HDF5_MAGIC
    except OSError:
        return False
