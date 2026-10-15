"""完整检索流水线（检索系统开发第二部分的第 3 项）。

把前两部分的产出串成一条链:

    用户查询
      → 第一部分 enhance_query()     查询理解与增强（缩写/中英/过滤）
      → 第二部分 MultiPathRetriever  向量多路 + BM25 多路 + 融合
      → 第二部分 BGEReranker         cross-encoder 重排序 + 多准则加权
      → 候选文档列表（带出处, 供生成层使用）

输出不是答案，是"证据清单"。生成层（r4）拿它去写答案，所以每条结果必须
带上 doc_id / journal / pub_year / pmid，否则医学场景下无法溯源。

用法:
    python scripts/retrieval_pipeline.py "二甲双胍对心血管疾病有何影响？"
    python scripts/retrieval_pipeline.py "metformin cardiovascular" --json
    python scripts/retrieval_pipeline.py "MI 死亡率" --no-rerank --fusion weighted
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from multipath_retriever import MultiPathRetriever          # noqa: E402
from query_understanding import enhance_query               # noqa: E402


class RetrievalPipeline:
    def __init__(self, fusion: str = "rrf", top_k_vector: int = 20,
                 top_k_keyword: int = 20, rerank_candidates: int = 20,
                 criteria_weights: dict | None = None, use_rerank: bool = True):
        self.fusion = fusion
        self.top_k_vector = top_k_vector
        self.top_k_keyword = top_k_keyword
        self.rerank_candidates = rerank_candidates
        self.criteria_weights = criteria_weights
        self.use_rerank = use_rerank
        self.retriever = MultiPathRetriever()
        self._reranker = None                               # 延迟加载: 模型 1.1 GB

    @property
    def reranker(self):
        if self._reranker is None:
            from reranker import BGEReranker
            self._reranker = BGEReranker(verbose=True)
        return self._reranker

    def search(self, query: str, top_k: int = 5,
               where_filter: dict | None = None) -> dict:
        t_all = time.perf_counter()

        t = time.perf_counter()
        qi = enhance_query(query)
        t_enhance = time.perf_counter() - t

        t = time.perf_counter()
        cands = self.retriever.retrieve(
            qi, top_k_vector=self.top_k_vector,
            top_k_keyword=self.top_k_keyword,
            fusion_strategy=self.fusion,
            where_filter=where_filter if where_filter is not None else None)
        t_retrieve = time.perf_counter() - t

        used = cands[:self.rerank_candidates]
        t = time.perf_counter()
        if self.use_rerank and used:
            results = self.reranker.rerank(
                qi.cleaned or query, used, top_n=top_k,
                criteria_weights=self.criteria_weights)
            t_rerank = time.perf_counter() - t
            stage = "rerank"
        else:
            results = used[:top_k]
            for i, c in enumerate(results):
                c["rerank_rank"] = i + 1
                c["rerank_score"] = c["fused_score"]
            t_rerank = 0.0
            stage = "fusion_only"

        return {
            "query": query,
            "stage": stage,
            "enhanced": {
                "lang": qi.lang,
                "entities": qi.entities,
                "canonical_terms": qi.canonical_terms,
                "vector_query": qi.vector_query,
                "keyword_query": qi.keyword_query,
                "filters": qi.filters,
                "filter_notes": qi.filter_notes,
                "n_variants": len(qi.variants),
            },
            "fusion": self.fusion,
            "n_candidates": len(cands),
            "results": [
                {
                    "rank": r.get("rerank_rank"),
                    "chunk_id": r["chunk_id"],
                    "score": r.get("rerank_score", r["fused_score"]),
                    "fused_score": r["fused_score"],
                    "criteria": r.get("criteria"),
                    "relevance_raw": r.get("relevance_raw"),
                    "path_ranks": r.get("path_ranks"),
                    "text": r["text"][:400],
                    "metadata": r["metadata"],
                }
                for r in results
            ],
            "timing": {
                "enhance_s": round(t_enhance, 3),
                "retrieve_s": round(t_retrieve, 2),
                "rerank_s": round(t_rerank, 2),
                "total_s": round(time.perf_counter() - t_all, 2),
            },
        }


def main() -> None:
    ap = argparse.ArgumentParser(description="完整检索流水线")
    ap.add_argument("query", nargs="?", default="二甲双胍对心血管疾病有何影响？")
    ap.add_argument("--fusion", default="rrf", choices=["rrf", "weighted", "simple"])
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--candidates", type=int, default=20)
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--weights", default=None, help="relevance,recency,authority")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    cw = None
    if args.weights:
        r, rc, a = (float(x) for x in args.weights.split(","))
        cw = {"relevance": r, "recency": rc, "authority": a}

    pipe = RetrievalPipeline(fusion=args.fusion, rerank_candidates=args.candidates,
                             criteria_weights=cw, use_rerank=not args.no_rerank)
    out = pipe.search(args.query, top_k=args.top_k)

    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    e = out["enhanced"]
    print(f"查询      : {out['query']}")
    print(f"语言/术语 : {e['lang']}  {e['canonical_terms']}")
    print(f"过滤      : {e['filters'] or '无'}  {e['filter_notes']}")
    print(f"召回/融合 : {out['n_candidates']} 条候选, 策略 {out['fusion']}, "
          f"阶段 {out['stage']}")
    t = out["timing"]
    print(f"耗时      : 增强 {t['enhance_s']}s + 召回 {t['retrieve_s']}s "
          f"+ 重排 {t['rerank_s']}s = {t['total_s']}s\n")
    for r in out["results"]:
        md = r["metadata"]
        cr = r["criteria"]
        extra = ""
        if cr:
            extra = (f"  rel={cr['relevance']:.2f} rec={cr['recency']:.2f} "
                     f"auth={cr['authority']:.2f}")
        print(f"[{r['rank']}] {r['score']:.4f}{extra}  {md['pub_year']}  "
              f"{str(md['journal'])[:26]}")
        print(f"     {r['chunk_id']}  {str(md['source_title'])[:52]}")


if __name__ == "__main__":
    main()
