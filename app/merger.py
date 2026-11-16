"""HDF5 包合并核心逻辑。

把 1~3 个平铺的 HDF5 源文件合并为单个自包含的 HDF5 文件：

* 每个源文件的根对象挂到输出文件的 ``/<文件主干名>`` 下；
* 只接受组（Group）与普通整数/浮点数据集（Dataset）；
* 只接受数值、字符串、标量对象引用三类属性；软链接与其他链接类型拒收；
* 硬链接与包内相对外链在输出中重建为同一对象的硬链接，共享对象身份；
* 允许组循环；不同源文件的同路径对象各自挂在自己的根下，不做合并；
* 对象引用属性重写为指向输出中的对应对象，空引用保持为空；
* 外链目标缺失或越出包边界时整包失败（抛出 :class:`MergeError`）。

本模块只做纯文件系统层面的处理，不依赖 Web 框架，便于单元测试与脚本复用。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import h5py
import numpy as np

# 解压后所有源文件的总大小上限（字节）。
MAX_TOTAL_BYTES = 20 * 1024 * 1024

# 用于放置“仅被对象引用指向、没有任何链接可达”的对象的隐藏组名。
UNLINKED_GROUP = "__unlinked__"

# 手动解析外链链时的最大跳转数，防止外部构造的链接环导致无限循环。
MAX_LINK_HOPS = 4096


class MergeError(Exception):
    """输入包不满足合并约束时抛出；调用方应整包拒绝，不产生半成品。"""


@dataclass
class _Source:
    """一个已打开的源 HDF5 文件及其在包内的相对路径信息。"""

    index: int
    rel_path: str          # 相对解压目录的路径，例如 "instrument.h5"
    abs_path: str          # 解压后的绝对路径
    stem: str              # 输出中的根组名（去掉 .h5 后缀）
    handle: h5py.File
    size: int


def _obj_addr(obj: h5py.HLObject) -> int:
    """HDF5 对象头地址：同一文件内硬链接指向同一对象时地址相同。"""
    return h5py.h5o.get_info(obj.id).addr


def _identity(src: _Source, obj: h5py.HLObject) -> tuple[int, int]:
    """源对象的全局身份键：(源文件序号, 对象头地址)。"""
    return src.index, _obj_addr(obj)


def _canonical_name(src: _Source, obj: h5py.HLObject) -> Optional[str]:
    """返回对象在自身源文件内的一个链接路径；没有任何链接可达时返回 None。"""
    name = obj.name  # 形如 "/g/d"，根为 "/"
    if name == "/":
        return "/"
    if name and name != "/":
        return name
    return None


class _Merger:
    """单次合并任务的有状态执行器。"""

    def __init__(self, sources: list[_Source], output_path: str) -> None:
        self.sources = sources
        self.output_path = output_path
        # (源文件序号, 对象头地址) -> 输出对象
        self.copied: dict[tuple[int, int], h5py.HLObject] = {}
        # 已经入队（= 只会被展开一次）的组身份。组在入队那一刻即登记，
        # 因此硬链接循环、同一组的多个入口都不会让它被重复展开。
        self._enqueued: set[tuple[int, int]] = set()
        # 输出中已占用的组内成员名，防止极端情况下合成名冲突
        self._used_synthetic: set[str] = set()
        self._out: Optional[h5py.File] = None
        # 阶段一登记、阶段二统一解析的外链入口
        self._deferred_externals: list[
            tuple[_Source, h5py.HLObject, str, h5py.ExternalLink]
        ] = []

    # ------------------------------------------------------------------ #
    # 入口
    # ------------------------------------------------------------------ #
    def run(self) -> None:
        try:
            with h5py.File(self.output_path, "w") as out:
                self._out = out
                # 第一步：先把所有源的根组建好并登记身份。必须先于任何属性
                # 复制完成，否则根属性里的跨文件对象引用在翻译时会找不到
                # 目标源的根组。
                queue: list[tuple[_Source, h5py.HLObject, h5py.Group]] = []
                for src in self.sources:
                    root_out = out.create_group(src.stem)
                    root_key = _identity(src, src.handle["/"])
                    self.copied[root_key] = root_out
                    self._enqueued.add(root_key)
                    queue.append((src, src.handle["/"], root_out))
                # 第二步：所有根组就绪后再复制根属性（其中可能含对象引用）。
                for src in self.sources:
                    self._copy_attrs(
                        src, src.handle["/"], self.copied[_identity(src, src.handle["/"])]
                    )
                # 阶段一：沿硬链接建立每个源的规范对象树，外链只登记。
                # 阶段二：所有规范树就绪后，再把外链目标以硬链接放到入口。
                self._build_canonical(queue)
                self._place_external_links()
        except Exception:
            # 输出文件在任何异常下都不保留，杜绝半成品。
            self._safe_remove()
            raise
        finally:
            self._out = None

    def _safe_remove(self) -> None:
        try:
            if os.path.exists(self.output_path):
                os.remove(self.output_path)
        except OSError:
            pass

    # ------------------------------------------------------------------ #
    # 遍历
    # ------------------------------------------------------------------ #
    def _build_canonical(
        self, queue: list[tuple[_Source, h5py.HLObject, h5py.Group]]
    ) -> None:
        """阶段一：仅沿硬链接展开各源的规范对象树。

        不变量：
        * 每个组身份（源序号 + 对象头地址）只展开一次，组循环因此终止；
        * 每个成员都按“遍历时的入口名”放置，硬链接的多个入口名都会保留；
        * 软链接/其他链接类型拒收，外链登记到阶段二处理。
        """
        while queue:
            cur_src, cur_group, out_parent = queue.pop()
            group_key = _identity(cur_src, cur_group)
            # 所有组在“入队时”即登记 _enqueued；未登记的（理论上不会出现）
            # 或已处理过的重复项直接跳过，保证每个组只展开一次。
            if group_key not in self._enqueued:
                continue

            for member_name in cur_group.keys():
                link = cur_group.get(member_name, getlink=True)
                if isinstance(link, h5py.SoftLink):
                    raise MergeError(
                        f"不支持软链接：{cur_src.rel_path}:{cur_group.name}/{member_name}"
                    )
                if isinstance(link, h5py.ExternalLink):
                    # 外链在阶段二统一解析，规范树阶段不跨文件展开。
                    self._deferred_externals.append(
                        (cur_src, cur_group, member_name, link)
                    )
                    continue
                if not isinstance(link, h5py.HardLink):
                    raise MergeError(
                        f"不支持的链接类型 {type(link).__name__}："
                        f"{cur_src.rel_path}:{cur_group.name}/{member_name}"
                    )

                # 跟随硬链接拿到目标对象；其身份由对象头地址决定，
                # 与用哪个入口名访问无关。
                target_obj = cur_group[member_name]
                target_key = _identity(cur_src, target_obj)
                # 始终用当前入口名 member_name 放置：新对象创建在该名下，
                # 已复制对象则在该名下再建一个硬链接，多个入口因此都保留。
                self._place(cur_src, target_obj, out_parent, member_name)
                # 组在入队时即登记 _enqueued，保证每个组只展开一次。
                if isinstance(target_obj, h5py.Group) and target_key not in self._enqueued:
                    self._enqueued.add(target_key)
                    queue.append((cur_src, target_obj, self.copied[target_key]))

    def _place_external_links(self) -> None:
        """阶段二：解析外链并在输出入口处建立到规范对象的硬链接。"""
        for cur_src, cur_obj, member_name, link in self._deferred_externals:
            target_src, target_obj = self._resolve_external(
                cur_src, cur_obj, member_name, link
            )
            key = _identity(target_src, target_obj)
            target_out = self.copied.get(key)
            if target_out is None:
                # 兜底：目标没有任何硬链接可达时，按规范路径补建。
                target_out = self._ensure_referenced(target_src, target_obj)
            out_parent = self._canonical_output_parent(cur_src, cur_obj)
            if member_name in out_parent:
                # 同名成员已存在时只能是同一对象，校验身份后复用。
                if _obj_addr(out_parent[member_name]) != _obj_addr(target_out):
                    raise MergeError(
                        f"外链入口 {member_name!r} 与已有不同对象冲突"
                    )
                continue
            out_parent[member_name] = target_out

    def _canonical_output_parent(
        self, src: _Source, obj: h5py.HLObject
    ) -> h5py.Group:
        """返回源对象在输出中的规范父组（阶段一后必然已存在）。"""
        root_out = self.copied[_identity(src, src.handle["/"])]
        name = obj.name
        if name == "/":
            return root_out
        parts = [p for p in name.split("/") if p]
        cur = root_out
        for part in parts:
            cur = cur[part]
        return cur

    # ------------------------------------------------------------------ #
    # 外链解析（只允许包内相对路径）
    # ------------------------------------------------------------------ #
    def _resolve_external(
        self,
        cur_src: _Source,
        cur_obj: h5py.HLObject,
        member_name: str,
        first_link: h5py.ExternalLink,
    ) -> tuple[_Source, h5py.HLObject]:
        """解析（可能嵌套的）外链，返回最终目标的 (源, 对象)。

        每一跳都重新检查链接类型：外链只允许指向包内文件，路径越界或目标
        缺失立即抛出 :class:`MergeError`。
        """
        base_dir = os.path.dirname(cur_src.abs_path)
        link = first_link
        seen: set[tuple[str, str]] = set()

        for _ in range(MAX_LINK_HOPS + 1):
            raw_file = link.filename
            raw_path = link.path
            if isinstance(raw_file, bytes):
                raw_file = raw_file.decode("utf-8")
            if isinstance(raw_path, bytes):
                raw_path = raw_path.decode("utf-8")

            if os.path.isabs(raw_file) or os.path.splitdrive(raw_file)[0]:
                raise MergeError(
                    f"外链使用了绝对路径或带盘符路径，越出包边界："
                    f"{cur_src.rel_path}:{cur_obj.name}/{member_name} -> {raw_file}"
                )
            # 规范化并限定在解压目录之内，拒绝任何 "../" 越界。
            base_real = os.path.realpath(base_dir)
            target_abs = os.path.realpath(os.path.join(base_real, raw_file))
            if target_abs != base_real and not target_abs.startswith(
                base_real + os.sep
            ):
                raise MergeError(
                    f"外链目标越出包边界：{cur_src.rel_path} 中的 {raw_file}"
                )
            target_src = self._find_source(target_abs)
            if target_src is None:
                raise MergeError(
                    f"外链目标文件不在包内或缺失："
                    f"{cur_src.rel_path}:{cur_obj.name}/{member_name} -> {raw_file}"
                )
            hop_key = (target_src.rel_path, raw_path)
            if hop_key in seen:
                raise MergeError(f"外链解析出现循环：{hop_key}")
            seen.add(hop_key)

            # 只看目标路径上的“链接本身”，不做隐式跟随。
            parent_path, _, tail = raw_path.rpartition("/")
            parent = target_src.handle["/"] if parent_path in ("", "/") else \
                self._open_path(target_src, parent_path)
            if parent is None or not isinstance(parent, h5py.Group) or tail not in parent:
                raise MergeError(
                    f"外链目标对象不存在：{target_src.rel_path}:{raw_path}"
                )
            next_link = parent.get(tail, getlink=True)
            if isinstance(next_link, h5py.ExternalLink):
                # 仍是外链：记录下一跳，基目录换成目标文件所在目录。
                link = next_link
                base_dir = os.path.dirname(target_src.abs_path)
                continue
            if not isinstance(next_link, h5py.HardLink):
                raise MergeError(
                    f"外链目标经由不支持的链接类型：{target_src.rel_path}:{raw_path}"
                )
            target_obj = parent[tail]
            return target_src, target_obj

        raise MergeError("外链跳转超过上限，疑似链接环")

    def _find_source(self, abs_path: str) -> Optional[_Source]:
        for src in self.sources:
            if os.path.realpath(src.abs_path) == abs_path:
                return src
        return None

    def _open_path(
        self, src: _Source, path: str
    ) -> Optional[h5py.HLObject]:
        """打开源文件内的对象；目标不存在（含断链）时返回 None。"""
        try:
            if path not in src.handle:
                return None
            return src.handle[path]
        except (KeyError, OSError):
            return None

    # ------------------------------------------------------------------ #
    # 对象放置与复制
    # ------------------------------------------------------------------ #
    def _place(
        self,
        src: _Source,
        obj: h5py.HLObject,
        local_parent: h5py.Group,
        local_name: str,
    ) -> h5py.HLObject:
        """把源对象放到输出中。

        规则：
        * 已复制过的同一源对象 -> 在当前位置再建一个硬链接（共享身份）；
        * 首次出现 -> 按对象类型在当前位置创建，并登记身份。
        不同源文件的对象身份键不同，因此同路径异对象绝不会被合并。
        """
        key = _identity(src, obj)
        existing = self.copied.get(key)
        if existing is not None:
            if local_name in local_parent:
                present = local_parent[local_name]
                if _obj_addr(present) == _obj_addr(existing):
                    # 同一对象的规范入口已经占用这个名字，直接复用即可。
                    return present
                # 名字被另一对象占用：源包同一组内本就不允许重名，
                # 走到这里说明合并逻辑出现冲突，显式失败而不是静默覆盖。
                raise MergeError(
                    f"输出成员名 {local_name!r} 已被不同对象占用"
                )
            # 同一源对象的后续入口：再建一个硬链接，共享对象身份。
            local_parent[local_name] = existing
            return existing

        if isinstance(obj, h5py.Group):
            new_obj = local_parent.create_group(local_name)
        elif isinstance(obj, h5py.Dataset):
            new_obj = self._create_dataset(local_parent, local_name, obj)
        else:
            raise MergeError(
                f"不支持的对象类型 {type(obj).__name__}：{src.rel_path}:{obj.name}"
            )

        self.copied[key] = new_obj
        self._copy_attrs(src, obj, new_obj)
        return new_obj

    def _create_dataset(
        self, parent: h5py.Group, name: str, ds: h5py.Dataset
    ) -> h5py.Dataset:
        """复制普通整数/浮点数据集，保留形状、dtype 与全部数值。"""
        kind = ds.dtype.kind
        if kind not in ("i", "u", "f"):
            raise MergeError(
                f"仅支持整数或浮点数据集，实际 dtype={ds.dtype!r}：{ds.file.filename}:{ds.name}"
            )
        if h5py.check_ref_dtype(ds.dtype) is not None or h5py.check_string_dtype(
            ds.dtype
        ) is not None:
            raise MergeError(
                f"数据集不能是字符串或引用类型：{ds.file.filename}:{ds.name}"
            )
        # 直接用源 dtype 与数据创建，h5py 会保留字节序、位宽与形状；
        # 数据被完整读入内存（整包解压后 ≤20MiB），输出因此完全自包含，
        # 源包删除后仍可独立读取。
        data = ds[()]
        return parent.create_dataset(name, data=np.asarray(data, dtype=ds.dtype))

    # ------------------------------------------------------------------ #
    # 属性复制
    # ------------------------------------------------------------------ #
    def _copy_attrs(
        self,
        src: _Source,
        src_obj: h5py.HLObject,
        out_obj: h5py.HLObject,
    ) -> None:
        for attr_name in src_obj.attrs.keys():
            attr_id = src_obj.attrs.get_id(attr_name)
            dtype = attr_id.dtype
            shape = attr_id.shape
            value = src_obj.attrs[attr_name]

            ref_cls = h5py.check_ref_dtype(dtype)
            if ref_cls is not None:
                # 只允许“标量对象引用”；区域引用、引用数组一律拒收。
                if ref_cls is not h5py.h5r.Reference:
                    raise MergeError(
                        f"属性 {attr_name!r} 使用了不支持的区域引用："
                        f"{src.rel_path}:{src_obj.name}"
                    )
                if shape != ():
                    raise MergeError(
                        f"属性 {attr_name!r} 是引用数组，仅支持标量对象引用："
                        f"{src.rel_path}:{src_obj.name}"
                    )
                out_ref = self._translate_ref(src, value)
                out_obj.attrs.create(attr_name, out_ref, dtype=h5py.ref_dtype)
                continue

            if h5py.check_string_dtype(dtype) is not None:
                # 定长/变长、ASCII/UTF-8 字符串按原 dtype 原样保留。
                out_obj.attrs.create(attr_name, value, dtype=dtype)
                continue

            if dtype.kind in ("i", "u", "f"):
                out_obj.attrs.create(attr_name, value, dtype=dtype, shape=shape)
                continue

            raise MergeError(
                f"属性 {attr_name!r} 的类型 {dtype!r} 不受支持"
                f"（仅允许数值、字符串、标量对象引用）：{src.rel_path}:{src_obj.name}"
            )

    def _translate_ref(self, src: _Source, ref: h5py.Reference) -> h5py.Reference:
        """把源文件中的对象引用翻译为输出文件中的对应对象引用。

        普通文件内引用直接在引用所属文件解析；跨文件引用（引用 token 实际
        属于另一个包内文件，常见于经外链取得 ``.ref`` 后写入的场景）在
        各源句柄中依次尝试解析。裸 HDF5 对象引用不含文件标识，因此跨文件
        解析基于“哪个包内文件能打开该 token”的尽力匹配。
        """
        if not ref:  # 空引用保持为空
            return h5py.Reference()

        target_src, target_obj = self._dereference(src, ref)
        out_target = self._ensure_referenced(target_src, target_obj)
        return out_target.ref

    def _dereference(
        self, referrer: _Source, ref: h5py.Reference
    ) -> tuple[_Source, h5py.HLObject]:
        """在包内各源文件中解析一个对象引用，返回 (所属源, 对象)。"""
        # 引用所属文件优先，覆盖最常见的文件内引用。
        ordered = [referrer] + [s for s in self.sources if s is not referrer]
        last_exc: Optional[Exception] = None
        for cand in ordered:
            try:
                obj = cand.handle[ref]
            except (KeyError, OSError, ValueError) as exc:
                last_exc = exc
                continue
            if not isinstance(obj, (h5py.Group, h5py.Dataset)):
                raise MergeError(
                    f"对象引用指向了不支持的对象类型：{cand.rel_path}:{getattr(obj, 'name', '?')}"
                )
            return cand, obj
        raise MergeError(
            f"对象引用无法在包内任何文件中解析（{referrer.rel_path}）：{last_exc}"
        )

    def _ensure_referenced(
        self, src: _Source, obj: h5py.HLObject
    ) -> h5py.HLObject:
        """确保被引用对象在输出中存在（普通遍历未覆盖到的兜底）。"""
        key = _identity(src, obj)
        existing = self.copied.get(key)
        if existing is not None:
            return existing

        root_out = self.copied[_identity(src, src.handle["/"])]
        canonical = _canonical_name(src, obj)
        if canonical is None:
            # 没有任何链接可达的对象，放到该源根下的隐藏组中，保证引用有效。
            orphan_parent = self._orphan_group(root_out)
            synth = f"obj_{_obj_addr(obj):x}"
            while synth in orphan_parent or synth in self._used_synthetic:
                synth = synth + "_"
            self._used_synthetic.add(synth)
            return self._place(src, obj, orphan_parent, synth)

        # 沿规范路径逐级确保父组存在，再放置对象本身。
        parts = [p for p in canonical.split("/") if p]
        parent_out = root_out
        cur = src.handle["/"]
        for part in parts[:-1]:
            child = cur[part]
            child_key = _identity(src, child)
            if child_key in self.copied:
                parent_out = self.copied[child_key]
            else:
                parent_out = self._place(src, child, parent_out, part)
            cur = child
        return self._place(src, obj, parent_out, parts[-1])

    def _orphan_group(self, root_out: h5py.Group) -> h5py.Group:
        if UNLINKED_GROUP in root_out:
            return root_out[UNLINKED_GROUP]
        grp = root_out.create_group(UNLINKED_GROUP)
        return grp


# ---------------------------------------------------------------------- #
# 对外入口
# ---------------------------------------------------------------------- #
def merge_sources(
    source_paths: list[str], output_path: str, total_size_limit: int = MAX_TOTAL_BYTES
) -> None:
    """把已解压在同一目录下的源 HDF5 文件合并为单个输出文件。

    :param source_paths: 源文件绝对路径列表（1~3 个，平铺）。
    :param output_path: 输出 HDF5 文件路径；失败时该路径不会留下文件。
    :param total_size_limit: 源文件总字节数上限。
    :raises MergeError: 任何不满足约束的情况，调用方应整包拒绝。
    """
    if not 1 <= len(source_paths) <= 3:
        raise MergeError("每个包必须包含 1~3 个 .h5 文件")

    total = 0
    for p in source_paths:
        total += os.path.getsize(p)
    if total > total_size_limit:
        raise MergeError(
            f"源文件合计 {total} 字节，超过上限 {total_size_limit} 字节（20 MiB）"
        )

    stems: set[str] = set()
    sources: list[_Source] = []
    handles: list[h5py.File] = []
    try:
        for idx, path in enumerate(sorted(source_paths)):
            stem = os.path.splitext(os.path.basename(path))[0]
            if not stem:
                raise MergeError(f"无法从文件名得到根组名：{path}")
            if stem in stems:
                raise MergeError(f"文件名主干重复，根组名冲突：{stem}")
            stems.add(stem)
            try:
                fh = h5py.File(path, "r")
            except OSError as exc:
                raise MergeError(f"无法作为 HDF5 打开：{os.path.basename(path)}: {exc}") from exc
            handles.append(fh)
            sources.append(
                _Source(
                    index=idx,
                    rel_path=os.path.basename(path),
                    abs_path=os.path.abspath(path),
                    stem=stem,
                    handle=fh,
                    size=os.path.getsize(path),
                )
            )
        _Merger(sources, output_path).run()
    finally:
        for fh in handles:
            try:
                fh.close()
            except Exception:
                pass
