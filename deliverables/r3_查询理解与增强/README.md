# 检索系统开发第一部分 —— 查询理解与增强 提交材料

> 提交人：＿＿＿＿＿＿＿＿　　学号：＿＿＿＿＿＿＿＿　　日期：2026-10-03

> 任务：在完成向量化索引后，构建 RAG 系统的智能检索系统第一部分——查询理解与增强。
> 对包含医学缩写、同义词、专业术语的查询做增强处理，输出清洗后查询、医学实体、
> 同义扩展、向量查询、关键词查询、过滤条件与多版本查询，供后续多路召回使用。
>
> 日期：2026-10-03 ｜ 仓库：https://github.com/Chase-nuo/medical_rag
>
> 注：本目录中的文件是从仓库对应位置复制而来；若内容有出入，以仓库 `docs/`、
> `scripts/` 下的原始文件为准（14 号报告即 `docs/查询理解与增强报告.md`）。

## 一、文件清单

| 文件 | 对应任务书条目 | 内容 |
|---|---|---|
| `14_统计报告_查询理解与增强.md` | 结论与实验数据 | 完整报告 12 节：符号词典、失效分析、词典设计、处理流程、实体识别、查询版本生成、过滤提取、三个被实验纠正的判断、对照实验、环境问题、产出与复现、遗留项 |
| `15_代码_query_understanding.py` | **本周产出：查询理解与增强代码** | 全部处理逻辑：静态医学同义词词典（43 组）、`MEDICAL_PATTERNS`（4 类，含 `\b` 边界）、清洗、实体识别、同义扩展、向量/关键词查询生成、过滤条件提取 |
| `16_代码_bench_query_enhancement.py` | 实验 | 对照实验（6 条查询 baseline vs enhanced）与中文桥接策略消融（A/B/C 三策略） |
| `17_实验结果_对照实验.json` | 实验结果（机器可读） | 6 条查询 × 18 字段：top1 相似度、top-n 均值、过滤合规率、术语命中、最优变体、耗时、top1 文献标题 |
| `18_实验结果_中文桥接消融.txt` | 实验结果 | 4 条中文查询在 A 原句 / B 原句+英文术语 / C 纯英文术语串 三策略下的 top1 相似度 |
| `19_处理日志_bench.log` | 处理日志 | 本次对照实验的完整控制台输出 |

## 二、核心结果

**词典规模**

| 层 | 规模 | 来源 |
|---|---|---|
| 静态 curated | 43 个概念组 / 125 个表层别名 / 52 条中文映射 / 18 条缩写 | 人工精选，中英对照 |
| 数据驱动 corpus | 原始 43 334 条 → 四道闸门过滤后 **42** 条 | 从本项目 1 006 篇文献挖掘的 `corpus_acronyms.csv` |

**对照实验**（baseline = 原查询单路无过滤；enhanced = 多版本查询 + 过滤，取最优路）

| 查询 | baseline | enhanced | Δ | 过滤合规 |
|---|---|---|---|---|
| 二甲双胍对心血管疾病有何影响？ | 0.5953 | **0.6943** | +0.0990 | — |
| MI 患者的死亡率 2020年以后的研究 | 0.6513 | **0.7032** | +0.0519 | 80% → **100%** |
| Does metformin reduce cardiovascular mortality… | 0.7422 | **0.7499** | +0.0077 | — |
| SGLT2 inhibitor heart failure hospitalization | 0.6053 | **0.6122** | +0.0069 | — |
| 近五年关于慢性肾病 eGFR 的随机对照试验 | 0.6545 | **0.7010** | +0.0465 | 20% → **100%** |
| 抗生素治疗社区获得性肺炎的疗程 | 0.5509 | 0.5509 | +0.0000 | — |
| **平均** | **0.6332** | **0.6686** | **+0.0353** | 100% |

6/6 无负例；两条带时间约束的查询过滤合规率由 80% / 20% 提升至 100%。

**中文桥接消融**：4 条中文查询中 A 赢 1 次、B 赢 1 次、C 赢 2 次——**最优策略不可预测**。
取 $\max(A,C)$ 相比拼接策略 B 平均 **+0.0220**，故最终不拼接，两种策略各生成一个变体，
交由第二部分的多路召回融合挑优。

## 三、增强输出格式

`enhance_query()` 返回 `EnhancedQuery`，字段与下游用途：

| 字段 | 含义 | 下游用途 |
|---|---|---|
| `original` / `cleaned` | 原始 / 清洗后查询 | 日志与溯源 |
| `lang` | zh / mixed / en | 决定是否生成跨语言变体 |
| `entities` | 类型 → 实体（drug / condition / outcome / study_type / abbreviation / zh_term） | 解释性输出 |
| `canonical_terms` | 规范术语 | 跨语言桥接 |
| `synonyms` / `expansions` | 同义词 / 缩写展开 | 关键词路扩面 |
| `vector_query` | 向量检索入口 | Chroma 查询 |
| `keyword_query` | 关键词检索入口 | 第二部分 BM25 |
| `filters` / `filter_notes` | 过滤条件 / 人类可读解释 | Chroma `where` |
| `variants` | 多版本查询 | 多路召回 |
| `provenance` | 术语来源（curated / corpus） | 置信度加权 |

示例（`MI mortality after 2020`）：

```
向量查询   myocardial infarction (MI) mortality after 2020
关键词查询 myocardial infarction mortality death fatal mi ami heart attack
过滤条件   {'pub_year': {'$in': ['2020', ..., '2026']}}
```

## 四、设计依据

三处**没有照做**任务书 / 常规做法的判断，均以实测为准：

| 项 | 文档或直觉 | 实测 | 结论 |
|---|---|---|---|
| BGE 指令前缀 | 任务书要求加 `Represent this question…` | 0.7422 → 0.6833（下降） | bge-m3 已内化指令，**默认关闭**，`--instruction` 保留 |
| 清洗去标点 | 常规 NLP 做法 | 0.7422 → 0.7273（下降） | 疑问标点携带语义，**只做全角转半角与空白压缩** |
| 中文拼接英文术语 | 第一版实现 | 消融：4 条里 2 条最差 | **不拼接**，A 与 C 各成一路 |

另有一处环境依赖：Windows 系统代理写在注册表而非环境变量，`httpx` 默认 `trust_env=True`
会把 `localhost` 请求发往代理，导致 `ollama` 客户端全部 `WinError 10061`。修复见
`scripts/embed_chunks.py` 顶部的 `NO_PROXY` 设置（必须在创建 client 之前）。

## 五、复现方式

```bash
git clone git@github.com:Chase-nuo/medical_rag.git
cd medical_rag
pip install -r requirements.txt

# 前置：需先完成阶段④索引（data/chroma/），并启动 Ollama
ollama serve

# 1) 单条查询的结构化增强结果
python scripts/query_understanding.py "MI 患者的死亡率 2020年以后的研究"
python scripts/query_understanding.py "二甲双胍对心血管疾病有何影响？" --json

# 2) 对照实验 / 中文桥接消融（产出 17、18、19 号文件）
python scripts/bench_query_enhancement.py
python scripts/bench_query_enhancement.py --json
python scripts/bench_query_enhancement.py --ablation
```

依赖环境：Python 3，主要包 `pandas` `numpy` `chromadb` `ollama`；
嵌入模型 `bge-m3`（Ollama，1024 维），索引 `data/chroma/`（27 349 块）。

## 六、已知局限

1. 对照实验样本仅 6 条且为自编，存在挑选风险；「术语命中」指标 baseline 6/6 → enhanced 6/6，无区分度，不作判据。
2. top1 相似度不是相关性标注。要计算 Recall@K / MRR，需先建立 20–30 条 (问题, 相关块) 标注集。
3. 数据驱动词典存在释义噪声（如 `BP → biological process`），靠 curated 优先级覆盖已知缩写，未知缩写仍有风险。
