"""多路检索器 MultiPathRetriever（检索系统开发第二部分）。

两路互补的动机：
    向量路按"意思"找，同义能召回，但罕见词/精确编号（eGFR、SGLT2）会被语义平均掉；
    BM25 路按"字面"找，精确术语不会漏，但"心梗"找不到 myocardial infarction。
    所以两路都要，再用融合策略把各自的排名证据合起来。

三种融合策略为什么要都实现：
    simple   合并去重，忽略排名 —— 作为基线，用来量化"排名信息值多少"
    rrf      只用排名，不需要分数可比 —— 两路分数天然不可比（余弦 ∈[-1,1]，
             BM25 无上界），RRF 绕开了归一化这个坑，是学术检索的默认选择
    weighted 分数加权 —— 需要先把各路分数归一化到 [0,1]，因此受归一化方式影响

用法:
    python scripts/multipath_retriever.py "MI 患者的死亡率 2020年以后的研究"
    python scripts/multipath_retriever.py "metformin cardiovascular" --fusion weighted
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import bm25_index as B                              # noqa: E402
import embed_chunks as E                            # noqa: E402  (内含 NO_PROXY 设置)

CHUNKS_PARQUET = os.path.join(ROOT, "data", "chunks", "chunks_length.parquet")

RRF_K = 60          # Cormack et al. 2009 原论文取值, 对 top-k 不敏感
DEFAULT_WEIGHTS = {"vector": 0.7, "keyword": 0.3}   # 语义为主, 字面为辅


class MultiPathRetriever:
    """向量路（Chroma，可多变体） + BM25 路，输出融合后的候选。"""

    def __init__(self, vector_index: E.MedicalIndex | None = None,
                 bm25: B.BM25Index | None = None,
                 chunks_df: pd.DataFrame | None = None):
        self.vector = vector_index or E.MedicalIndex()
        self.bm25 = bm25 or B.BM25Index.load()
        if chunks_df is None:
            chunks_df = pd.read_parquet(CHUNKS_PARQUET)
        self.df = chunks_df.set_index("chunk_id")

    # ------------------------------------------------------------------ 主入口
    def retrieve(self, query_info, top_k_vector: int = 20, top_k_keyword: int = 20,
                 fusion_strategy: str = "rrf", where_filter: dict | None = None,
                 rrf_k: int = RRF_K,
                 weights: dict | None = None) -> list[dict]:
        """多路检索 + 融合。

        Args:
            query_info: 第一部分 enhance_query() 的输出（EnhancedQuery）
            top_k_vector: 每个向量变体路返回数量
            top_k_keyword: BM25 路返回数量
            fusion_strategy: 'rrf' | 'weighted' | 'simple'
            where_filter: 元数据过滤（一般用 query_info.filters）
            rrf_k: RRF 的 k
            weights: weighted 策略下的 {'vector': w, 'keyword': w}

        Returns:
            融合后的候选列表，按融合分数降序；每项含 chunk_id / 文本 / 元数据 /
            各路分数与排名，供重排序器与生成层使用。
        """
        weights = weights or DEFAULT_WEIGHTS
        where = where_filter if where_filter is not None else (query_info.filters or None)

        paths: list[tuple[str, str, list[tuple[str, float]]]] = []

        # --- 向量路: 对第一部分产出的每个变体各检索一次 ---
        variants = list(query_info.variants) or [query_info.vector_query]
        for i, vq in enumerate(variants):
            if not vq or not vq.strip():
                continue
            r = self.vector.query(vq, n_results=top_k_vector, where_filter=where)
            paths.append((f"vec{i}", "vector",
                          list(zip(r["ids"], r["similarities"]))))

        # --- BM25 路: 用第一部分生成的英文关键词串 ---
        kw = getattr(query_info, "keyword_query", "") or query_info.cleaned
        # 先多取再过滤(over-fetch): BM25 没有 where 能力, 只能后置过滤。
        # 若直接取 top_k 再过滤, 实测会损失 45%~60% 的候选
        # (年份过滤下 20 条只剩 8~11 条), 等于变相削弱了这一路。
        n_fetch = top_k_keyword * (5 if where else 1)
        idxs, scores = self.bm25.query(kw, top_k=n_fetch)
        bm25_pairs = [(self.bm25.chunk_ids[i], s) for i, s in zip(idxs, scores)]
        if bm25_pairs:
            # 元数据过滤: BM25 没有 where, 用 DataFrame 补齐后剔除
            bm25_pairs = self._apply_filter(bm25_pairs, where)[:top_k_keyword]
            paths.append(("bm25", "keyword", bm25_pairs))

        fused = self._fuse(paths, fusion_strategy, rrf_k, weights)
        return self._materialize(fused)

    # ------------------------------------------------------------------ 融合
    @staticmethod
    def _fuse(paths, strategy, rrf_k, weights) -> list[dict]:
        """paths: [(路名, 类型, [(chunk_id, score) 降序]), ...]"""
        if not paths:
            return []

        if strategy == "rrf":
            acc: dict[str, dict] = {}
            for name, kind, pairs in paths:
                for rank, (cid, sc) in enumerate(pairs):
                    d = acc.setdefault(cid, {"chunk_id": cid, "rrf": 0.0,
                                             "scores": {}, "ranks": {}})
                    d["rrf"] += 1.0 / (rrf_k + rank + 1)
                    d["scores"][name] = float(sc)
                    d["ranks"][name] = rank + 1
            for d in acc.values():
                d["fused"] = d["rrf"]
            out = sorted(acc.values(), key=lambda d: -d["fused"])

        elif strategy == "weighted":
            acc = {}
            for name, kind, pairs in paths:
                norm = MultiPathRetriever._minmax([s for _, s in pairs])
                for rank, (cid, sc) in enumerate(pairs):
                    d = acc.setdefault(cid, {"chunk_id": cid, "part": {},
                                             "scores": {}, "ranks": {}})
                    v = norm[rank]
                    # 同一类型取最高: 多个变体路不应被简单相加而互相放大
                    if kind not in d["part"] or v > d["part"][kind]:
                        d["part"][kind] = v
                    d["scores"][name] = float(sc)
                    d["ranks"][name] = rank + 1
            for d in acc.values():
                d["fused"] = sum(weights.get(k, 0.0) * v for k, v in d["part"].items())
            out = sorted(acc.values(), key=lambda d: -d["fused"])

        elif strategy == "simple":
            # 基线: 按路顺序拼接去重, 不看分数也不看排名 —— 缺陷正是它存在的意义
            seen: dict[str, dict] = {}
            for name, kind, pairs in paths:
                for rank, (cid, sc) in enumerate(pairs):
                    if cid in seen:
                        continue
                    seen[cid] = {"chunk_id": cid, "fused": 1.0 / (len(seen) + 1),
                                 "scores": {name: float(sc)}, "ranks": {name: rank + 1}}
            out = list(seen.values())

        else:
            raise ValueError(f"未知融合策略: {strategy}")

        for d in out:
            d["n_paths"] = len(d.get("ranks", {}))
        return out

    @staticmethod
    def _minmax(vals: list[float]) -> list[float]:
        """把一路内的分数压到 [0,1]。单值路返回 [1.0]，避免除零。"""
        if not vals:
            return []
        lo, hi = min(vals), max(vals)
        if hi - lo < 1e-9:
            return [1.0] * len(vals)
        return [(v - lo) / (hi - lo) for v in vals]

    # ------------------------------------------------------------------ 辅助
    def _apply_filter(self, pairs, where: dict | None) -> list[tuple[str, float]]:
        if not where:
            return pairs
        keep = []
        for cid, sc in pairs:
            if cid not in self.df.index:
                continue
            row = self.df.loc[cid]
            ok = True
            for key, cond in where.items():
                val = row.get(key)
                if isinstance(cond, dict) and "$in" in cond:
                    ok = ok and str(val) in {str(x) for x in cond["$in"]}
                else:
                    ok = ok and str(val) == str(cond)
            if ok:
                keep.append((cid, sc))
        return keep

    def _materialize(self, fused: list[dict]) -> list[dict]:
        out = []
        for d in fused:
            cid = d["chunk_id"]
            if cid not in self.df.index:
                continue
            row = self.df.loc[cid]
            out.append({
                "chunk_id": cid,
                "fused_score": round(float(d["fused"]), 6),
                "path_scores": {k: round(v, 4) for k, v in d.get("scores", {}).items()},
                "path_ranks": d.get("ranks", {}),
                "n_paths": d.get("n_paths", 1),
                "text": row["text"],
                "metadata": {
                    "doc_id": row.get("doc_id"), "source_title": row.get("source_title"),
                    "pub_year": row.get("pub_year"), "journal": row.get("journal"),
                    "lang": row.get("lang"), "pmid": row.get("pmid"),
                    "is_table": bool(row.get("is_table")),
                },
            })
        return out


def main() -> None:
    ap = argparse.ArgumentParser(description="多路检索")
    ap.add_argument("query", nargs="?", default="MI 患者的死亡率 2020年以后的研究")
    ap.add_argument("--fusion", default="rrf", choices=["rrf", "weighted", "simple"])
    ap.add_argument("--top-k-vector", type=int, default=10)
    ap.add_argument("--top-k-keyword", type=int, default=10)
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args()

    from query_understanding import enhance_query
    qi = enhance_query(args.query)
    print(f"查询      : {qi.original}")
    print(f"向量变体  : {len(qi.variants)} 路   关键词: {qi.keyword_query[:60]}")
    print(f"过滤      : {qi.filters or '无'}")

    retr = MultiPathRetriever()
    res = retr.retrieve(qi, top_k_vector=args.top_k_vector,
                        top_k_keyword=args.top_k_keyword,
                        fusion_strategy=args.fusion)
    print(f"\n融合策略 {args.fusion}  候选 {len(res)} 条  取 top{args.top}:")
    for i, r in enumerate(res[:args.top]):
        md = r["metadata"]
        print(f"  [{i}] {r['fused_score']:.4f}  {r['chunk_id']}  "
              f"{str(md['pub_year'])}  {str(md['source_title'])[:44]}")
        print(f"       路={r['path_ranks']}")


if __name__ == "__main__":
    main()
