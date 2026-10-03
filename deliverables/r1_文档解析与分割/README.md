# 文档解析与分割 —— 提交材料

> 任务：将 1 020 篇 PMC OA 医学文献解析并分割为适合向量化与检索的文本单元，
> 输出包含所有文本块及其元数据的数据集。
>
> 日期：2026-09-20 ｜ 仓库：https://github.com/Chase-nuo/medical_rag
>
> 注：本目录中的文件是从仓库对应位置复制而来；若内容有出入，以仓库 `docs/`、
> `scripts/`、`data/chunks/` 下的原始文件为准（01 号报告由脚本自动生成）。

## 一、文件清单

| 文件 | 对应任务书条目 | 内容 |
|---|---|---|
| `01_统计报告_文档解析与分割.md` | 处理日志和统计报告 | 完整报告：处理配置、清洗明细、规模统计、块长分布（分位数+直方图）、质量验证、结论 |
| `02_代码_build_chunks.py` | 实现 | 全部处理逻辑：DataFrame 构建、基础清洗、智能分割（策略 a）与整体不分割（策略 b）、落盘、质检 |
| `03_质量验证结果.csv` | 质量验证结果 | 问题块清单：`chunk_id` / `doc_id` / `chunk_index` / `token_count` / `issues` / 块首 80 字符 |
| `04_处理统计_processing_stats.json` | 统计报告 | 机器可读：配置、剔除明细、块长分位数、总 token 数、语言分布、overlap 与质检计数 |
| `05_处理日志_build_chunks.log` | 处理日志 | 本次运行的完整控制台输出（加载 → 清洗 → 切块 → 预览 → 质检 → 落盘） |

## 二、核心结果

| 项 | 值 |
|---|---|
| 输入文献 | 1 020 篇 |
| 清洗后 | **1 006 篇**（极短文 12 + 硬编码名单 2） |
| 文本块总数 | **27 349** |
| 块/篇 | 均值 27.2，中位 25，最大 194 |
| chunk_size / overlap | 512 / 64 token（12.5%），bge-m3 tokenizer 计数 |
| 块长 | min 32 · p50 417 · p95 500 · **max 509**（无一块超 512） |
| 总 token 数 | 10 488 326 |

**质量验证**（详见 01 号报告第 5 节）

| 检查项 | 命中 | 判定 |
|---|---|---|
| 超模型上限（>8192） | 0 | ✅ |
| 超 chunk_size（>512） | 0 | ✅ |
| 空块 | 0 | ✅ |
| 过短块（<20 token） | 0 | ✅ 已自动并入相邻块 |
| 编码异常 mojibake | 0 | ✅ |
| 硬切在单词中间 | 1 | ⚠️ MathType 数学公式，占 0.004% |
| overlap 生效 | 98.0% | ✅ 抽样 1 410 对相邻块，重叠 ≥100 字符（中位 200） |

**剔除说明**：极短文 12 篇（<100 行，会议摘要/社论）；`PMC5444287` core 为空、`PMC4616690` 全篇 mojibake 不可修复。剔除后者后 mojibake 命中由 5 块降为 0，反证语料中仅此一篇全篇损坏。

## 三、数据集格式

任务书写明「文本块数据集本地存储无需上交」，故数据集本体（17.9 MB parquet / 50.5 MB jsonl）未包含在本目录。字段定义与一条样例如下：

| 字段 | 类型 | 说明 |
|---|---|---|
| `chunk_id` | str | 全局唯一，格式 `{doc_id}_{序号:04d}` |
| `text` | str | 块正文（首块之后每块前置上一块尾部 64 token，实现重叠） |
| `doc_id` | str | 归属文献 ID（用 pmcid，零缺失；pmid 缺 4 篇故仅作元数据） |
| `chunk_index` / `total_chunks` | int | 块在原文中的序号 / 原文被分成的总块数 |
| `source_title` | str | 原文标题，便于追溯 |
| `token_count` | int | bge-m3 tokenizer 计数的真实 token 数 |
| `pmid` / `doi` / `journal` / `pub_year` | str | 检索期元数据过滤字段（`journal`、`pub_year` 零缺失） |
| `lang` | str | en / zh / mixed，按 CJK 占比判定 |
| `is_table` | bool | 是否含表格（PMC 把表格展平成单段） |
| `tail_cut` | bool | 尾边界是否识别成功（118 篇为 False，仍含参考文献） |

样例（节选，text 已截断）：

```json
{
  "chunk_id": "PMC10010752_0000",
  "text": "Subjective cognitive decline (SCD), the self-reported experience of worsening ...",
  "doc_id": "PMC10010752",
  "chunk_index": 0,
  "total_chunks": 24,
  "source_title": "Racial and Ethnic Differences in Subjective Cognitive Decline — United States, 2015–2020",
  "token_count": 429,
  "pmid": "36893045",
  "doi": "10.15585/mmwr.mm7210a1",
  "journal": "MMWR Morb Mortal Wkly Rep",
  "pub_year": "2023",
  "lang": "en",
  "is_table": false,
  "tail_cut": true
}
```

## 四、参数依据

切块参数取自前期数据分析报告 `docs/RAG数据分析与设计说明.md`（已随仓库提交）：

| 参数 | 取值 | 依据 |
|---|---|---|
| chunk_size | 512 token | §4.3：超限段落占比 2.49%，是 ≤5% 的最小值 |
| chunk_overlap | 64 token（12.5%） | §4.3：建议区间 10~15% |
| 分割器 | RecursiveCharacterTextSplitter | 按段落 → 句子 → 空格逐级降级 |
| 切块对象 | core 正文 | §2.5：元数据头块 + 参考文献占 32.2%，必须先剔除 |
| length_function | bge-m3 tokenizer | 切块上限取决于嵌入模型的 tokenizer，不能用字符数近似 |
| doc_id | pmcid | §1.5：pmid 缺 4 篇，pmcid 零缺失 |

对照组（策略 b 整体不分割）：1 006 篇中 **491 篇超 8 192 token** 上限，证明必须按长度切块。

## 五、复现方式

```bash
git clone git@github.com:Chase-nuo/medical_rag.git
cd medical_rag
pip install -r requirements.txt

# 1) 语料（PMC OA 原始语料 51 MB 未入库，需先抓取）
python scripts/fetch_pmc_oa.py
python scripts/expand_corpus.py --target 1000 --workers 4

# 2) 切块（约 1 分钟产出 27 349 块与本报告）
python scripts/build_chunks.py --report-md docs/文档解析与分割报告.md
```

依赖环境：Python 3，主要包 `pandas` `pyarrow` `langchain-text-splitters` `tokenizers`；
tokenizer 缓存位于 `data/tokenizer/bge-m3/`，缺失时会自动下载。
