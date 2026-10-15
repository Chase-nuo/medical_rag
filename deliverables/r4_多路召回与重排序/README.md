# 检索系统开发第二部分 —— 多路召回、融合与重排序 提交材料

> 提交人：＿＿＿＿＿＿＿＿　　学号：＿＿＿＿＿＿＿＿　　日期：2026-10-16

> 任务：应用多路检索（MultiPathRetriever）提升召回质量、搭建重排序器（bge-reranker-base），
> 并结合第一部分完成完整检索流水线。
> 完整推导与实验过程见 `20_统计报告_多路召回与重排序.md`（11 节）。

---

## 文件清单

| 编号 | 文件 | 内容 |
|---|---|---|
| 20 | `20_统计报告_多路召回与重排序.md` | 完整报告：符号词典、环境约束、BM25 设计、融合策略、重排序、四项实验、工程决策、遗留项 |
| 21 | `21_代码_bm25_index.py` | BM25 倒排索引（构建 / 查询 / 存盘） |
| 22 | `22_代码_multipath_retriever.py` | 多路检索 + 三种融合（rrf / weighted / simple） |
| 23 | `23_代码_reranker.py` | cross-encoder 重排序 + 多准则加权 |
| 24 | `24_代码_retrieval_pipeline.py` | 完整流水线（理解 → 召回 → 融合 → 重排） |
| 25 | `25_代码_bench_retrieval.py` | 四项对照实验脚本 |
| 26 | `26_实验结果_对照实验.json` | 结构化实验结果 |
| 27 | `27_索引统计_bm25.json` | BM25 索引规模凭证 |
| 28 | `28_处理日志_bench_retrieval.log` | 主实验运行日志 |
| 29 | `29_处理日志_补充实验.txt` | 参数扫描、量化验证、BM25 增量价值、融合策略抹平效应 |

---

## 核心结果

**重排序收益**（6 条查询，同一候选池重排前后）

| 指标 | 结果 |
|---|---|
| 平均 Δtop1 relevance | **+0.1980** |
| 最大单条 Δtop1 | **+0.8619**（SGLT2 查询：0.1334 → 0.9953） |
| 多准则 top5 − 融合 top5 | **+0.2725** |

**融合策略对比**（候选池 top5 relevance 均值，6 条查询）

| rrf | weighted | simple |
|---|---|---|
| 0.4781 | 0.4263 | **0.4977** |

反直觉结果：`simple` 最高。原因不是它更好，而是它退化成"只保留向量 top20"；
补充实验证明 BM25 每条查询能带来 **2~3 条向量路漏掉的高相关候选**（最高 rel 0.998）。

**重排序抹平了策略差异**：融合阶段 top1 相差可达 0.84 的三种策略，重排后 top1 **完全相同**
（SGLT2 三条都是 0.9953）。所以默认仍用 `rrf`（理论依据 + 保留 BM25 增量）。

**延迟**（CPU 8 线程，不含 LLM 生成）

| 候选数 | 召回 | 重排 | 总计 |
|---|---|---|---|
| 10 | 8.45 s | 6.78 s | 15.23 s |
| 20 | 8.86 s | 13.17 s | 22.03 s |
| 40 | 8.48 s | 24.91 s | 33.39 s |

**BM25 索引**：27 349 块 / 142 925 词 / 3 155 148 posting / 29.4 MB / 构建 5.7 s

---

## 输出格式（流水线）

```
用户查询 → enhance_query() → 向量 2~3 路 + BM25 1 路 → 融合 → 重排序 → 证据列表
```

每条结果带 `rank / score / criteria(rel,rec,auth) / path_ranks / text / metadata`。
metadata 含 `doc_id / journal / pub_year / pmid / is_table`，保证医学场景可溯源。

---

## 复现

```bash
pip install -r requirements.txt          # 含 transformers==5.19.0
python scripts/bm25_index.py --build     # 5.7 s
python scripts/retrieval_pipeline.py "二甲双胍对心血管疾病有何影响？"
python scripts/bench_retrieval.py --all --log --json
```

**重排序模型不在仓库内**（1.06 GB，已加入 `.gitignore`），需下载到 `models/bge-reranker-base/`：

```
https://modelscope.cn/api/v1/models/BAAI/bge-reranker-base/repo?Revision=master&FilePath=<文件名>
# config.json / tokenizer.json / tokenizer_config.json /
# special_tokens_map.json / sentencepiece.bpe.model / model.safetensors
```

（HF 主站大文件会被 TLS 中断，故走 ModelScope 镜像）

---

## 已知局限（报告 §10）

1. **无标注集**：所有"是否更相关"用 cross-encoder 自评，存在自证偏差，算不了 Recall@K / MRR。这是最需要补的一环。
2. **authority 准则用代理指标**：没有 JCR 影响因子，用"期刊在本语料的发文量"代替。消融中出现失效案例（`Foods` 排在 `Cardiovasc Drugs Ther` 之前）。
3. **int8 量化默认关闭**：快 2.1 倍，但 top5 重合仅 3~5/5、分数最大偏差 0.47~0.54，会改写排序。
