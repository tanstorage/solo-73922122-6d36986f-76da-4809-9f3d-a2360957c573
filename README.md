# 光谱 HDF5 单文件合并服务

把仪器主文件与共享标定文件（1～3 个平铺 `.h5`，打包为一个 ZIP）合并成**单件、自包含**
的 HDF5，供离线调整标定后复算。基于 FastAPI + h5py 实现。

## 它做什么

上传一个 ZIP，服务返回一个 `merged.h5`：

- 每个源文件的根对象挂到输出的 `/<文件名主干>`，例如 `instrument.h5` → `/instrument`；
- 仅接受**组**与**普通整数/浮点数据集**（含各种位宽与字节序、标量、空数组），
  保留形状、dtype、全部数值与属性；
- 属性仅接受**数值、字符串（定长/变长、ASCII/UTF-8）、标量对象引用**；
  区域引用、引用数组、复合等其他属性类型拒收；
- **软链接及其他链接类型一律拒收**；
- **硬链接**与**包内相对外链**在输出中重建为同一对象的硬链接，多个入口共享对象身份，
  允许组循环；
- **对象引用属性**重写为指向输出中的对应对象，空引用保持为空；
- 不同源文件的同路径对象各自挂在自己的根下，**不会被合并**；
- 外链目标缺失、对象不存在或路径越出包边界（含绝对路径、`../`）时**整包失败**，
  不产生半成品；
- 输出完全自包含：源包/源文件删除后，下载文件仍可独立完整读取。

### 离线复算语义

合并后，标定对象只有一份物理拷贝、多个硬链接入口。因此修改共享标定（例如改写
`/calibration/calib/coeffs`）后，仪器文件一侧的所有入口
（`/instrument/shared_calib/...`、`/instrument/calib_alias/...`）以及指向它的
对象引用读到的都是新值。

## 目录结构

```
app/
  main.py       FastAPI 路由：POST /merge、/health、/docs
  merger.py     合并核心（纯文件系统逻辑，可脱离 Web 复用）
  package.py    ZIP 解压与静态校验（平铺、1~3 个 .h5、≤20MiB、zip-slip 防护）
samples/
  build_samples.py   生成三个示例 ZIP（共享标定 / 循环 / 同路径异对象）
  verify_samples.py  对示例跑关键性质断言（本地核心或 --url 走 HTTP）
tests/          pytest 套件（40 个用例）
Dockerfile
docker-compose.yml
requirements.txt
```

## 本地运行

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

打开交互式文档：<http://127.0.0.1:8000/docs>（可直接在页面上传 ZIP 试 `/merge`）。

## Docker Compose 运行

```bash
docker compose up --build
# 服务位于 http://127.0.0.1:8000 ，文档 http://127.0.0.1:8000/docs
```

## 调用示例

```bash
curl -f -o merged.h5 \
  -F "file=@samples/data/shared_calib.zip;type=application/zip" \
  http://127.0.0.1:8000/merge
```

| 场景 | HTTP 状态码 |
| --- | --- |
| 成功 | `200`，返回 `application/x-hdf5` |
| 非 `.zip` 上传 | `415` |
| ZIP 结构/大小/HDF5 内容不满足约束 | `400` |
| 上传载荷超过 25 MiB | `413` |
| 缺少上传文件字段 | `422` |

## 示例与自测

```bash
# 生成示例 ZIP 到 samples/data/
.venv/bin/python -I samples/build_samples.py samples/data

# 直接对本地合并核心跑断言
.venv/bin/python -I samples/verify_samples.py samples/data

# 对运行中的服务走真实 HTTP 跑同样断言
.venv/bin/python -I samples/verify_samples.py samples/data --url http://127.0.0.1:8000

# 完整 pytest 套件
.venv/bin/python -I -m pytest tests/ -q
```

三个示例覆盖：

- `shared_calib.zip`：共享标定 + 两个包内相对外链入口 + 硬链接 + 标量/空对象引用属性；
- `cycles.zip`：硬链接组循环、自引用、跨文件外链共享同一身份；
- `same_path_diff.zip`：两个源文件含同路径但类型/值不同的对象，禁止合并。

## 关键设计说明

- **对象身份**用 `(源文件序号, HDF5 对象头地址)` 表示。同一文件内硬链接指向同一对象时
  头地址相同；不同源文件即使头地址数值相同，也因序号不同而绝不会被合并。
- **两阶段合并**：阶段一只沿硬链接建立每个源的规范对象树（外链只登记）；阶段二所有规范
  树就绪后再解析外链并在入口处建立硬链接。这保证跨文件目标先在自己的规范位置完整落地。
- **每个组只展开一次**：组在入队时即登记到 `_enqueued`，硬链接循环因此必然终止，
  而每个成员仍按遍历时的入口名放置，硬链接的多个入口名都会保留。
- **链接判定**统一用 `group.get(name, getlink=True)`，区分 HardLink / SoftLink /
  ExternalLink，不依赖隐式跟随，断链也能被明确识别并拒绝。
- **外链边界**：相对路径经 `realpath` 规范化后必须仍位于解压目录内；绝对路径、盘符、
  `../` 越界一律拒绝。嵌套外链逐跳手动解析并检测链接环。
- **失败不留半成品**：合并在临时文件上进行，任何异常都会删除输出；HTTP 层在响应发送
  完成后清理整个临时工作目录。
- **大小限制**：以**解压后真实字节数**计量（≤20 MiB），防止高压缩比压缩炸弹；
  原始上传另设 25 MiB 上限。

## 已知边界

- HDF5 裸对象引用本身不携带“属于哪个文件”的标识。对于跨文件对象引用属性（把经外链
  取得的 `.ref` 写入另一文件），服务在各包内源文件中尽力解析（引用所属文件优先）；
  文件内引用这一常规场景不受影响。
- 每个包最多 3 个文件、解压后合计 ≤20 MiB；数据在合并时完整读入内存以保证输出自包含。
