"""多准则重排序器（检索系统开发第二部分）。

为什么重排序能提升：
    召回阶段用的是双塔向量（query 与 doc 各自独立编码），计算快但交互浅；
    重排序用 cross-encoder（query 与 doc 拼在一起过模型），能建模词级交互，
    代价是必须对每个候选单独推理——所以它只能作用于召回后的几十条，不能全库跑。

三个准则的取值设计（都压到 [0,1] 后才加权，否则权重没有意义）：
    relevance  cross-encoder 概率。注意原始分跨查询不可比（同一批里可能全在
               0.02~0.3），所以在候选集内做 min-max 归一化，让 0.6 的权重
               真正表达"相关性最重要"，而不是被量纲吃掉。
    recency    按发表年份线性衰减：w = (year - Y0) / (Y1 - Y0)，缺年份取中性 0.5。
    authority  期刊权威性代理。没有 JCR / 影响因子数据，用「该期刊在本语料中的
               发文量」做代理（发文多 = 该领域主力期刊），再对顶刊白名单加成。
               这是启发式，已在报告中明确标注局限。

用法:
    python scripts/reranker.py "MI 患者的死亡率 2020年以后的研究"
    python scripts/reranker.py "metformin cardiovascular" --top-n 5 --weights 1,0,0
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import torch                                        # noqa: E402
from transformers import (AutoModelForSequenceClassification,  # noqa: E402
                          AutoTokenizer)

MODEL_DIR = os.path.join(ROOT, "models", "bge-reranker-base")
HF_NAME = "BAAI/bge-reranker-base"
CHUNKS_PARQUET = os.path.join(ROOT, "data", "chunks", "chunks_length.parquet")

# 语料年份范围(阶段③统计), 固定参考系, 避免不同批次的候选集漂移
YEAR_MIN, YEAR_MAX = 1992, 2026

DEFAULT_CRITERIA = {"relevance": 0.6, "recency": 0.25, "authority": 0.15}

# 顶刊白名单: 没有影响因子数据时, 至少把公认顶刊显式提权(仅作加成, 不替代代理指标)
TOP_JOURNALS = {
    "the new england journal of medicine", "lancet", "jama",
    "jama cardiology", "jama internal medicine", "bmj",
    "nature", "nature medicine", "science", "cell",
    "circulation", "european heart journal", "journal of the american college of cardiology",
    "annals of internal medicine", "plos medicine", "diabetes care",
    "kidney international", "american journal of respiratory and critical care medicine",
    "journal of clinical oncology", "blood", "gut", "brain",
}


class BGEReranker:
    """bge-reranker-base (XLM-RoBERTa cross-encoder) + 多准则加权。"""

    def __init__(self, model_path: str = MODEL_DIR, batch_size: int = 8,
                 max_length: int = 512, verbose: bool = True,
                 quantize: bool = False):
        path = model_path if os.path.isdir(model_path) else HF_NAME
        t0 = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.model.eval()
        if quantize:
            # 实测(3 条查询 × 20 候选): 20 条 12.11s → 5.78s, 快 2.1 倍;
            # 但 top5 重合只有 3~5/5, 分数最大偏差 0.47~0.54 —— 排序会被改写。
            # 因此默认关闭: 重排序是最后一道质量关, 不该为省 6 秒牺牲 top5。
            self.model = torch.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8)
        torch.set_num_threads(os.cpu_count() or 8)
        self.batch_size = batch_size
        self.max_length = max_length
        self._journal_w = self._build_journal_weights()
        if verbose:
            print(f"重排序模型加载 {time.perf_counter() - t0:.1f}s  "
                  f"({sum(p.numel() for p in self.model.parameters()) / 1e6:.0f}M 参数)",
                  flush=True)

    # ------------------------------------------------------------- 期刊权重
    @staticmethod
    def _build_journal_weights() -> dict[str, float]:
        """用本语料的期刊发文量做权威性代理: 0.4 起步, log 归一, 顶刊再 +0.15。"""
        try:
            df = pd.read_parquet(CHUNKS_PARQUET)
            cnt = df.groupby("doc_id")["journal"].first().value_counts()
        except Exception:                            # noqa: BLE001
            return {}
        if cnt.empty:
            return {}
        mx = math.log1p(int(cnt.max()))
        w: dict[str, float] = {}
        for j, c in cnt.items():
            base = 0.4 + 0.6 * (math.log1p(int(c)) / mx)
            if str(j).strip().lower() in TOP_JOURNALS:
                base += 0.15
            w[str(j)] = round(min(base, 1.0), 4)
        return w

    def authority(self, journal) -> float:
        if journal is None or not str(journal).strip():
            return 0.3
        j = str(journal).strip()
        if j in self._journal_w:
            return self._journal_w[j]
        low = j.lower()
        if low in TOP_JOURNALS:
            return 1.0
        return 0.45                                  # 语料里没出现过的期刊: 中性偏下

    # ------------------------------------------------------------- 相关性
    def relevance_scores(self, query: str, texts: list[str]) -> np.ndarray:
        """cross-encoder 打分, 返回 sigmoid 概率。"""
        if not texts:
            return np.zeros(0, dtype=np.float32)
        outs = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            enc = self.tokenizer([query] * len(batch), batch, padding=True,
                                 truncation=True, max_length=self.max_length,
                                 return_tensors="pt")
            with torch.no_grad():
                logits = self.model(**enc).logits.squeeze(-1)
            outs.append(torch.sigmoid(logits.float()).numpy())
        return np.concatenate(outs).astype(np.float32)

    # ------------------------------------------------------------- 主入口
    def rerank(self, query: str, candidates: list[dict], top_n: int = 5,
               criteria_weights: dict | None = None,
               normalize_relevance: bool = True) -> list[dict]:
        """对召回候选重排序。

        Args:
            query: 原始查询文本
            candidates: MultiPathRetriever.retrieve() 的输出
            top_n: 返回条数
            criteria_weights: {'relevance','recency','authority'}
            normalize_relevance: 是否在候选集内归一化相关性分（默认开, 理由见模块文档）
        """
        if not candidates:
            return []
        cw = dict(DEFAULT_CRITERIA)
        if criteria_weights:
            cw.update({k: v for k, v in criteria_weights.items() if k in cw})
        total = sum(cw.values()) or 1.0
        cw = {k: v / total for k, v in cw.items()}   # 归一化, 防止权重和不为 1

        texts = [c.get("text", "") for c in candidates]
        t0 = time.perf_counter()
        rel = self.relevance_scores(query, texts)
        rerank_s = time.perf_counter() - t0

        if normalize_relevance and len(rel) > 1:
            lo, hi = float(rel.min()), float(rel.max())
            rel_n = (rel - lo) / (hi - lo) if hi - lo > 1e-9 else np.ones_like(rel)
        else:
            rel_n = rel

        for c, r_raw, r_n in zip(candidates, rel, rel_n):
            year = c.get("metadata", {}).get("pub_year")
            try:
                y = int(str(year)[:4])
                rec = (y - YEAR_MIN) / (YEAR_MAX - YEAR_MIN)
            except (TypeError, ValueError):
                rec = 0.5                            # 缺年份: 中性, 不惩罚也不奖励
            rec = min(max(rec, 0.0), 1.0)
            auth = self.authority(c.get("metadata", {}).get("journal"))
            c["criteria"] = {
                "relevance": round(float(r_n), 4),
                "recency": round(float(rec), 4),
                "authority": round(float(auth), 4),
            }
            c["relevance_raw"] = round(float(r_raw), 4)
            c["rerank_score"] = round(
                cw["relevance"] * float(r_n)
                + cw["recency"] * rec
                + cw["authority"] * auth, 6)
            c["fused_rank_before"] = None

        out = sorted(candidates, key=lambda c: -c["rerank_score"])
        for i, c in enumerate(out):
            c["rerank_rank"] = i + 1
        out[0]["_rerank_seconds"] = round(rerank_s, 2)
        return out[:top_n]


def main() -> None:
    ap = argparse.ArgumentParser(description="多准则重排序")
    ap.add_argument("query", nargs="?", default="MI 患者的死亡率 2020年以后的研究")
    ap.add_argument("--top-n", type=int, default=5)
    ap.add_argument("--candidates", type=int, default=20, help="召回候选数")
    ap.add_argument("--fusion", default="rrf", choices=["rrf", "weighted", "simple"])
    ap.add_argument("--weights", default=None,
                    help="三准则权重 relevance,recency,authority，如 1,0,0")
    ap.add_argument("--no-normalize", action="store_true", help="不做相关性归一化")
    args = ap.parse_args()

    from multipath_retriever import MultiPathRetriever
    from query_understanding import enhance_query

    qi = enhance_query(args.query)
    retr = MultiPathRetriever()
    cands = retr.retrieve(qi, top_k_vector=args.candidates,
                          top_k_keyword=args.candidates,
                          fusion_strategy=args.fusion)
    print(f"查询: {qi.original}   召回 {len(cands)} 条")

    cw = None
    if args.weights:
        r, rc, a = (float(x) for x in args.weights.split(","))
        cw = {"relevance": r, "recency": rc, "authority": a}

    rr = BGEReranker()
    top = rr.rerank(args.query, cands, top_n=args.top_n, criteria_weights=cw,
                    normalize_relevance=not args.no_normalize)
    print(f"\n重排序 top{len(top)} (用时 {top[0].get('_rerank_seconds')}s):")
    for c in top:
        md = c["metadata"]
        cr = c["criteria"]
        print(f"  {c['rerank_score']:.4f}  rel={cr['relevance']:.3f} "
              f"rec={cr['recency']:.2f} auth={cr['authority']:.2f}  "
              f"{str(md['pub_year'])}  {str(md['source_title'])[:40]}")


if __name__ == "__main__":
    main()
