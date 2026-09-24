# 向量化与索引构建 —— 提交材料

> 任务：为 27 349 个文本块生成嵌入向量，用 ChromaDB 构建持久化向量索引，
> 并完成质量验证（基础统计 / 相似性检索 / 边界情况 / 元数据过滤）。
>
> 日期：2026-09-24 ｜ 仓库：https://github.com/Chase-nuo/medical_rag
>
> 注：本目录中的文件是从仓库对应位置复制而来；若内容有出入，以仓库 `docs/`、
> `scripts/`、`data/chroma/`、`data/chunks/` 下的原始文件为准（06 号报告即
> `docs/向量化与索引构建报告.md`）。

## 一、文件清单

| 文件 | 对应任务书条目 | 内容 |
|---|---|---|
| `06_统计报告_向量化与索引构建.md` | 处理日志和统计报告 | 完整报告：硬件约束、模型选型排除链、batch 吞吐实测、余弦相似度推导、索引设计、质量验证、故障复盘 |
| `07_代码_embed_chunks.py` | 嵌入与索引构建实现 | 嵌入（含断点续跑 / memmap / 重试）→ 灌库 → 统计落盘 → 查询封装 |
| `08_代码_verify_index.py` | 质量验证实现 | 六类验证：基础统计 / 语义检索 / 指令对比 / 自相似性 / 元数据过滤 / 边界 |
| `09_索引统计_index_stats.json` | **正确向量数量的索引统计** | 机器可读：27 349 块 / 1024 维 / bge-m3 / 1006 篇 / 块长统计 |
| `10_质量验证结果.csv` | 质量验证结果 | 26 项检查的结果与判定（含自相似性、过滤、边界、指令对比） |
| `11_处理日志_verify_index.log` | 质量验证过程证据 | 验证脚本完整输出，含每条自相似查询的 top1 与相似度 |
| `12_处理日志_embed_首段.log` | 嵌入过程证据（0 → 12 480） | 含速率与 ETA 逐段记录，以及故障拐点（`ReadTimeout` ×2） |
| `13_处理日志_embed_续跑.log` | 嵌入过程证据（14 080 → 27 349） | 重启后从断点续跑至完成，含一次 502 重试 |

## 二、核心结果

| 项 | 值 |
|---|---|
| 输入文本块 | 27 349（来自 1 006 篇文献） |
| 嵌入模型 | `bge-m3`（Ollama GGUF，CPU 推理） |
| 向量维度 | 1 024 |
| 索引向量数 | **27 349**（与块数一致，零向量 0） |
| 距离度量 | cosine（`hnsw:space=cosine`） |
| id 格式 | `{doc_id}_{chunk_index:04d}`，如 `PMC10010752_0000` |
| 元数据字段 | 12 个（过滤：`lang` `pub_year` `is_table`；溯源：`pmid` `doi` `journal`） |
| 嵌入耗时 | 209.5 min（续跑段计时；含故障的总墙钟约 7.3 h） |
| 建索引耗时 | 2.1 min |
| 单次查询耗时 | 4.85 s（含查询向量化，纯 CPU） |

**质量验证**（详见 06 号报告第 6 节，逐项见 `10_质量验证结果.csv`）

| 检查项 | 结果 | 判定 |
|---|---|---|
| 向量数量 = 块数 | 27 349 = 27 349 | ✅ |
| 零向量行 | 0 | ✅ |
| 自相似性（exact / 同篇 / 跨篇） | 8/8 · 8/8 · **0/8** | ✅ |
| 自相似平均相似度 | 0.9690 | ✅ |
| 语义检索 top1 | sim 0.7422，命中二甲双胍心血管结局研究 | ✅ 切题 |
| 元数据过滤（4 种） | 返回值字段值全部满足条件 | ✅ |
| 边界（空 / 超长 / 乱码） | 均不报错 | ✅ |

**指令前缀实验**：任务书写"BGE 查询添加指令获得更好的检索效果"，该结论来自 bge v1.5 系列。bge-m3 实测相反——无指令 0.7422 > 加指令 0.6833，且 top1 被换掉。故**默认关闭** `--instruction`，开关保留供换模型后重测。

## 三、为什么向量库本体不在本目录

| 产物 | 体积 | 是否提交 | 理由 |
|---|---|---|---|
| `data/chroma/`（ChromaDB 库） | 466.5 MB | ❌ | 体积大且可重建：`--skip-embed` 用已有 npy 2 min 重建 |
| `data/chunks/embeddings_bge-m3.npy` | 106.8 MB | ❌ | 可重建（约 4.5 h）；已写入 `.gitignore` |
| `data/chunks/*.parquet` / `*.jsonl` | 17.9 / 50.5 MB | ❌ | 阶段③产出，同属可重建数据 |
| `09_索引统计_index_stats.json` | 0.6 KB | ✅ | **任务书要求的"索引统计"，必须可查证** |
| 运行日志 ×3 | 3.4–4.9 KB | ✅ | 过程证据（耗时、速率、故障），体积小 |

嵌入与索引**分两步**、中间落 npy，正是为了让"昂贵的向量（4.5 h）"与"便宜的索引（2 min）"解耦：换 collection 名、改元数据、换距离度量都不必重算向量。

### 向量文件的获取（GitHub Release）

`embeddings_bge-m3.npy` 是唯一"贵且不可现场生成"的资产（重跑需约 4.5 h），已作为 Release 资产发布：

| 项 | 值 |
|---|---|
| Release | `v0.2-embeddings-bge-m3` |
| 下载地址 | https://github.com/Chase-nuo/medical_rag/releases/tag/v0.2-embeddings-bge-m3 |
| 文件名 | `embeddings_bge-m3.npy` |
| 大小 | 112 021 632 字节（106.8 MB） |
| SHA256 | `1dcc2201b852a2b8e2b102ced02d73d98db293b62d4205d701163493b9f6e5bf` |

下载后放入 `data/chunks/`，执行 `python scripts/embed_chunks.py --skip-embed` 约 2 min 即可重建完整索引，**无需重跑嵌入**。校验完整性：

```bash
python -c "import hashlib;h=hashlib.sha256();f=open('data/chunks/embeddings_bge-m3.npy','rb');[h.update(c) for c in iter(lambda:f.read(1<<20),b'')];print(h.hexdigest())"
# 应输出 1dcc2201b852a2b8e2b102ced02d73d98db293b62d4205d701163493b9f6e5bf
```

## 四、选型与参数依据

| 参数 | 取值 | 依据 |
|---|---|---|
| 嵌入模型 | `bge-m3` | 排除链：en-v1.5 系列不支持中文（语料 en 997 + zh 9）；OpenAI 付费且离线不可复现；clinicalBERT 需 (query, 文档) 标注，本项目无标注 |
| batch size | 64 | 实测吞吐饱和点：1→0.44、32→1.65、64→1.70、256→1.85 块/s；256 仅快 9% 却抬高内存与失败成本 |
| 距离度量 | cosine | 嵌入训练目标让相关文本**方向**接近；$\cos(q,d)=\frac{q\cdot d}{\\|q\\|\\|d\\|}$ 消除模长干扰；L2 与内积均受文本长度/词频影响 |
| id | `doc_id + chunk_index` | 唯一性：两个局部唯一量的拼接 ⟹ 全局唯一；幂等性：显式 id 让重复 `add` 变覆盖，避免 UUID 堆重复条目 |
| timeout | 120 s（原 600 s） | 600 s 时一次服务挂起吃掉 2 h；120 s 时最多 6 min 即报错退出，便于人工介入 |
| `--instruction` | 关闭 | 实测无指令更优（见第二节） |

## 五、复现方式

```bash
git clone git@github.com:Chase-nuo/medical_rag.git
cd medical_rag
pip install -r requirements.txt

# 前置：已有 data/chunks/chunks_length.parquet（阶段③产出，见 deliverables/01-05）
# 前置：ollama pull bge-m3

# 1) 全量嵌入 + 建索引（约 4.5 h，支持断点续跑；中断后重跑同一条命令即可）
python scripts/embed_chunks.py

# 2) 仅重建索引（已有 npy，约 2 min）
python scripts/embed_chunks.py --skip-embed

# 3) 查询
python scripts/embed_chunks.py --query "SGLT2 inhibitor heart failure" --n 5
python scripts/embed_chunks.py --query "抗生素疗程" --n 3 --where '{"lang":"zh"}'

# 4) 质量验证（生成 10、11 号文件）
python scripts/verify_index.py --samples 20 --log
```

依赖环境：Python 3，主要包 `numpy` `pandas` `pyarrow` `chromadb` `ollama`；
嵌入由本机 Ollama 服务（`:11434`）提供，需先启动 `ollama serve`。

已知注意点：Ollama 连续运行 5 天后出现过一次退化（`ReadTimeout`、502），
表现为吞吐从 1.9 掉到 0.8 块/s。跑长任务前建议先重启 Ollama（详见 06 号报告第 7 节）。
