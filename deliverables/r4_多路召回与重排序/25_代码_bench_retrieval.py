"""检索系统第二部分的对照实验与消融。

难点: 没有 (问题, 相关块) 标注集, 算不了 Recall@K / MRR。
怎么衡量"融合策略 A 比 B 好"、"重排序有没有用"?

做法: 用 cross-encoder(bge-reranker-base) 的分数当裁判。
    它比双塔向量更准（query 与 doc 一起过模型, 能建模词级交互）, 因此
    「候选池里 rerank 后 top-k 的 relevance 均值」可以作为相关性的代理指标。
    这不等于人工标注, 但比"top1 余弦相似度"可信得多, 且完全可复现。
    局限已在报告 §6 写明: 裁判与被测的重排序器是同一个模型, 存在自证偏差,
    所以只用来比较**候选池质量**（不同融合策略送进去的候选不同）, 而不用来
    宣称"重排序把相关性提升了 X%"。

四个实验:
    --fusion     三种融合策略各召回 20 条, 用 cross-encoder 给候选池打分
    --rerank     重排序前后 top1/top5 的 relevance 变化
    --criteria   多准则权重消融(纯相关性 vs 相关性+时效+权威)
    --latency    各阶段耗时与候选数的关系

用法:
    python scripts/bench_retrieval.py --all --json
    python scripts/bench_retrieval.py --fusion
    python scripts/bench_retrieval.py --latency
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from multipath_retriever import MultiPathRetriever           # noqa: E402
from query_understanding import enhance_query                # noqa: E402
from reranker import BGEReranker                             # noqa: E402

QUERIES = [
    "二甲双胍对心血管疾病有何影响？",
    "MI 患者的死亡率 2020年以后的研究",
    "Does metformin reduce cardiovascular mortality in type 2 diabetes?",
    "SGLT2 inhibitor heart failure hospitalization",
    "近五年关于慢性肾病eGFR的随机对照试验",
    "抗生素治疗社区获得性肺炎的疗程",
]

FUSIONS = ["rrf", "weighted", "simple"]
LOG_PATH = os.path.join(ROOT, "data", "chunks", "bench_retrieval.log")


class Tee:
    """同时写控制台与日志文件。"""
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, "w", encoding="utf-8")
        self.console = sys.stdout

    def write(self, s):
        self.console.write(s)
        self.f.write(s)

    def flush(self):
        self.console.flush()
        self.f.flush()


def _hdr(title: str) -> None:
    print()
    print("=" * 76)
    print(title)
    print("=" * 76)


# ---------------------------------------------------------------- 实验 A
def exp_fusion(rr: BGEReranker, retr: MultiPathRetriever, k: int = 20) -> list[dict]:
    _hdr("A) 融合策略对比 —— 候选池质量（用 cross-encoder 打分, 越高越好）")
    print(f"{'查询':<28} " + "  ".join(f"{f:>18}" for f in FUSIONS))
    print("-" * 76)
    rows = []
    agg = {f: [] for f in FUSIONS}
    for q in QUERIES:
        qi = enhance_query(q)
        line = f"{q[:26]:<28} "
        row = {"query": q}
        for f in FUSIONS:
            cands = retr.retrieve(qi, top_k_vector=k, top_k_keyword=k,
                                  fusion_strategy=f)
            if not cands:
                row[f] = None
                line += f"{'—':>18}  "
                continue
            rel = rr.relevance_scores(qi.cleaned or q,
                                      [c["text"] for c in cands[:k]])
            top1, top5 = float(rel[0]), float(rel[:5].mean())
            row[f] = {"n": len(cands), "top1": round(top1, 4),
                      "top5": round(top5, 4)}
            agg[f].append(top5)
            line += f"{top1:>8.4f}/{top5:<8.4f}  "
        rows.append(row)
        print(line, flush=True)
    print("-" * 76)
    print("候选池 top5 relevance 均值:  " +
          "  ".join(f"{f}={st.fmean(agg[f]):.4f}" for f in FUSIONS if agg[f]))
    return rows


# ---------------------------------------------------------------- 实验 B
def exp_rerank(rr: BGEReranker, retr: MultiPathRetriever, k: int = 20) -> list[dict]:
    _hdr("B) 重排序收益 —— 同一候选池, 重排前后 top1/top5 的 relevance")
    print(f"{'查询':<28} {'重排前top1':>10} {'重排后top1':>11} {'Δtop1':>8} "
          f"{'Δtop5':>8}")
    print("-" * 76)
    rows = []
    d1, d5 = [], []
    for q in QUERIES:
        qi = enhance_query(q)
        cands = retr.retrieve(qi, top_k_vector=k, top_k_keyword=k,
                              fusion_strategy="rrf")
        if len(cands) < 5:
            continue
        used = cands[:k]
        rel = rr.relevance_scores(qi.cleaned or q, [c["text"] for c in used])
        # 重排前: 融合顺序下的 top1 / top5
        before1, before5 = float(rel[0]), float(rel[:5].mean())
        # 重排后: 按 cross-encoder 分数排序（纯相关性准则下的理想顺序）
        order = sorted(range(len(rel)), key=lambda i: -rel[i])
        after1 = float(rel[order[0]])
        # 多准则重排后的平均 relevance（衡量加入时效/权威后付出了多少相关性代价）
        ranked = rr.rerank(qi.cleaned or q, used, top_n=5)
        multi5 = st.fmean(c["relevance_raw"] for c in ranked)
        d1.append(after1 - before1)
        d5.append(multi5 - before5)
        rows.append({"query": q, "before_top1": round(before1, 4),
                     "after_top1": round(after1, 4),
                     "delta_top1": round(after1 - before1, 4),
                     "before_top5": round(before5, 4),
                     "multi_top5": round(multi5, 4)})
        print(f"{q[:26]:<28} {before1:>10.4f} {after1:>11.4f} "
              f"{after1 - before1:>+8.4f} {multi5 - before5:>+8.4f}", flush=True)
    print("-" * 76)
    print(f"平均 Δtop1 = {st.fmean(d1):+.4f}   "
          f"多准则 top5 相对融合 top5 = {st.fmean(d5):+.4f}")
    return rows


# ---------------------------------------------------------------- 实验 C
def exp_criteria(rr: BGEReranker, retr: MultiPathRetriever, k: int = 20) -> list[dict]:
    _hdr("C) 多准则权重消融 —— 加入时效/权威后, 结果与纯相关性有什么不同")
    print(f"{'查询':<26} {'准则':<16} {'top1年份':>8} {'期刊':>22} {'rel':>7}")
    print("-" * 76)
    settings = [
        ("纯相关性", {"relevance": 1.0, "recency": 0.0, "authority": 0.0}),
        ("默认 6/2.5/1.5", None),
        ("重时效 4/5/1", {"relevance": 0.4, "recency": 0.5, "authority": 0.1}),
    ]
    rows = []
    for q in QUERIES[:4]:
        qi = enhance_query(q)
        cands = retr.retrieve(qi, top_k_vector=k, top_k_keyword=k,
                              fusion_strategy="rrf")[:k]
        if not cands:
            continue
        for name, w in settings:
            top = rr.rerank(qi.cleaned or q, list(cands), top_n=1,
                            criteria_weights=w)[0]
            md = top["metadata"]
            print(f"{q[:24]:<26} {name:<16} {str(md['pub_year']):>8} "
                  f"{str(md['journal'])[:20]:>22} {top['relevance_raw']:>7.3f}",
                  flush=True)
            rows.append({"query": q, "setting": name,
                         "top1": top["chunk_id"], "year": md["pub_year"],
                         "journal": md["journal"],
                         "relevance": top["relevance_raw"]})
    return rows


# ---------------------------------------------------------------- 实验 D
def exp_latency(rr: BGEReranker, retr: MultiPathRetriever) -> list[dict]:
    _hdr("D) 延迟 —— 候选数对重排序耗时的影响（CPU, 8 线程）")
    print(f"{'候选数':>6} {'召回s':>8} {'重排s':>8} {'总计s':>8} {'ms/条':>8}")
    print("-" * 76)
    rows = []
    q = "Does metformin reduce cardiovascular mortality in type 2 diabetes?"
    qi = enhance_query(q)
    for k in (10, 20, 40):
        t = time.perf_counter()
        cands = retr.retrieve(qi, top_k_vector=k, top_k_keyword=k,
                              fusion_strategy="rrf")
        t_ret = time.perf_counter() - t
        used = cands[:k]
        t = time.perf_counter()
        rr.rerank(qi.cleaned or q, used, top_n=5)
        t_rr = time.perf_counter() - t
        print(f"{k:>6} {t_ret:>8.2f} {t_rr:>8.2f} {t_ret + t_rr:>8.2f} "
              f"{t_rr / max(len(used), 1) * 1000:>8.0f}", flush=True)
        rows.append({"candidates": k, "retrieve_s": round(t_ret, 2),
                     "rerank_s": round(t_rr, 2),
                     "total_s": round(t_ret + t_rr, 2)})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="检索系统第二部分实验")
    ap.add_argument("--all", action="store_true", help="跑全部实验")
    ap.add_argument("--fusion", action="store_true")
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--criteria", action="store_true")
    ap.add_argument("--latency", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--log", action="store_true", help="落盘日志")
    args = ap.parse_args()

    if args.log:
        sys.stdout = Tee(LOG_PATH)
    if not (args.all or args.fusion or args.rerank or args.criteria or args.latency):
        args.all = True

    retr = MultiPathRetriever()
    rr = BGEReranker()

    out: dict = {}
    if args.all or args.fusion:
        out["fusion"] = exp_fusion(rr, retr)
    if args.all or args.rerank:
        out["rerank_gain"] = exp_rerank(rr, retr)
    if args.all or args.criteria:
        out["criteria"] = exp_criteria(rr, retr)
    if args.all or args.latency:
        out["latency"] = exp_latency(rr, retr)

    if args.json:
        path = os.path.join(ROOT, "data", "chunks", "bench_retrieval.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\nJSON 已保存 -> {path}")


if __name__ == "__main__":
    main()
