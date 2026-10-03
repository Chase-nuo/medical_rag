# -*- coding: utf-8 -*-
"""查询增强的对照实验（检索系统开发 · 第一部分的验证）
=====================================================================
查询理解做得对不对，不能只看输出长得好不好看，要看检索结果是否真的变好。
本脚本对每条查询跑两路：

  baseline : 原始查询，无过滤            （不做任何增强）
  enhanced : vector_query + 元数据过滤   （查询理解与增强的产物）

并用三个**可自动判定**的指标比较：

  1 top1 相似度      增强后是否把"最相关"的那条推得更靠前
  2 过滤合规率       若提取了 pub_year 过滤，返回块是否 100% 满足条件
                     （这一项 baseline 必然违规，因为它根本不知道时间约束）
  3 术语命中率       top5 文本里是否出现规范化术语或其英文同义词
                     （验证缩写展开 / 中英桥接是否真的把词带进了结果）

用法:
    python scripts/bench_query_enhancement.py
    python scripts/bench_query_enhancement.py --n 10 --json
"""
import argparse
import json
import os
import statistics as st
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import embed_chunks as E                    # noqa: E402
from query_understanding import enhance_query  # noqa: E402

# 覆盖三类难点：中文查询、英文缩写、时间约束
QUERIES = [
    "二甲双胍对心血管疾病有何影响？",
    "MI 患者的死亡率 2020年以后的研究",
    "Does metformin reduce cardiovascular mortality in type 2 diabetes?",
    "SGLT2 inhibitor heart failure hospitalization",
    "近五年关于慢性肾病eGFR的随机对照试验",
    "抗生素治疗社区获得性肺炎的疗程",
]


def _year_ok(md, cond) -> bool:
    """校验单条结果的 pub_year 是否满足过滤条件（支持 $in 枚举与等值）。"""
    y = str(md.get("pub_year", ""))
    if not y.isdigit():
        return False
    if isinstance(cond, dict):
        if "$in" in cond:
            return y in {str(c) for c in cond["$in"]}
        if "$gte" in cond and int(y) < int(cond["$gte"]):
            return False
        if "$lte" in cond and int(y) > int(cond["$lte"]):
            return False
        return True
    return y == str(cond)


def _term_hit(texts, terms) -> bool:
    low = " ".join(texts).lower()
    return any(t.lower() in low for t in terms) if terms else False


def run_one(idx, q, n):
    eq = enhance_query(q)
    # baseline：原样查询，不加过滤
    b = idx.query(q, n_results=n)
    # enhanced：对所有变体逐路检索（都带过滤），取 top1 最高的一路。
    # 这就是下游"多路召回 + 融合"后的实际效果，公平对baseline的单路。
    kw = {"where_filter": eq.filters or None}
    e, best_variant = None, eq.vector_query
    for v in (eq.variants or [eq.vector_query]):
        r = idx.query(v, n_results=n, **kw)
        if e is None or r["similarities"][0] > e["similarities"][0]:
            e, best_variant = r, v

    syn_terms = []
    for canon, syns in eq.synonyms.items():
        syn_terms.append(canon)
        syn_terms += [s for s in syns if not any("\u4e00" <= c <= "\u9fff" for c in s)]

    # 过滤合规率
    if "pub_year" in eq.filters:
        cond = eq.filters["pub_year"]
        e_ok = sum(_year_ok(m, cond) for m in e["metadatas"]) / len(e["metadatas"])
        b_ok = sum(_year_ok(m, cond) for m in b["metadatas"]) / len(b["metadatas"])
    else:
        e_ok = b_ok = None

    return {
        "query": q,
        "lang": eq.lang,
        "vector_query": eq.vector_query,
        "best_variant": best_variant,
        "n_variants": len(eq.variants),
        "filters": eq.filters,
        "filter_notes": eq.filter_notes,
        "baseline_top1": b["similarities"][0],
        "enhanced_top1": e["similarities"][0],
        "baseline_topn_mean": round(st.fmean(b["similarities"]), 4),
        "enhanced_topn_mean": round(st.fmean(e["similarities"]), 4),
        "filter_compliance_baseline": None if e_ok is None else round(b_ok, 3),
        "filter_compliance_enhanced": None if e_ok is None else round(e_ok, 3),
        "term_hit_baseline": _term_hit(b["documents"], syn_terms),
        "term_hit_enhanced": _term_hit(e["documents"], syn_terms),
        "top1_title_baseline": b["metadatas"][0].get("source_title", "")[:60],
        "top1_title_enhanced": e["metadatas"][0].get("source_title", "")[:60],
    }


ZH_QUERIES = [
    "二甲双胍对心血管疾病有何影响？",
    "抗生素治疗社区获得性肺炎的疗程",
    "近五年关于慢性肾病eGFR的随机对照试验",
    "MI 患者的死亡率 2020年以后的研究",
]


def ablation_zh(idx, n=5):
    """中文查询的桥接策略消融。

    语料 27196 en / 153 zh，中文查询怎么送进向量模型是个真问题。三种候选：
      A 原句           : 靠 bge-m3 自身的多语言能力做跨语言匹配
      B 原句 + 英文术语 : 把识别出的英文 canonical 追加在句尾做锚点
      C 纯英文术语串   : 完全丢弃中文句，只用术语检索
    用 top1 相似度决定默认策略，不靠直觉。
    """
    print("=" * 78)
    print("中文桥接策略消融   A=原句  B=原句+英文术语  C=纯英文术语串")
    print("=" * 78)
    print(f"{'查询':<30} {'A':>8} {'B':>8} {'C':>8}   最优")
    print("-" * 78)
    wins = {"A": 0, "B": 0, "C": 0}
    for q in ZH_QUERIES:
        eq = enhance_query(q)
        en_terms = [c for c in eq.canonical_terms
                    if not any("\u4e00" <= ch <= "\u9fff" for ch in c)]
        where = eq.filters or None
        a = idx.query(q, n_results=n, where_filter=where)["similarities"][0]
        b = idx.query(f"{eq.cleaned} {' '.join(en_terms)}".strip(),
                      n_results=n, where_filter=where)["similarities"][0]
        c = (idx.query(" ".join(en_terms), n_results=n, where_filter=where)["similarities"][0]
             if en_terms else float("-inf"))
        best = max((a, "A"), (b, "B"), (c, "C"))[1]
        wins[best] += 1
        print(f"{q[:28]:<30} {a:>8.4f} {b:>8.4f} {c:>8.4f}   {best}")
    print("-" * 78)
    print(f"各策略最优次数: {wins}")


def main():
    ap = argparse.ArgumentParser(description="查询增强对照实验")
    ap.add_argument("--n", type=int, default=5, help="每路返回条数")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    ap.add_argument("--ablation", action="store_true", help="只跑中文桥接策略消融")
    args = ap.parse_args()

    idx = E.MedicalIndex()
    if args.ablation:
        ablation_zh(idx, args.n)
        return
    rows = []
    for q in QUERIES:
        t = time.perf_counter()
        try:
            r = run_one(idx, q, args.n)
            r["seconds"] = round(time.perf_counter() - t, 1)
            rows.append(r)
        except Exception as exc:                              # noqa: BLE001
            rows.append({"query": q, "error": f"{type(exc).__name__}: {exc}"})
        # 进度行走 stderr: 保证 --json 时 stdout 是纯净 JSON, 可被下游直接解析
        print(f"[done] {q[:40]}  {time.perf_counter() - t:.1f}s",
              flush=True, file=sys.stderr)

    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return

    print("\n" + "=" * 78)
    print("对照实验汇总   baseline = 原查询无过滤   enhanced = 增强查询 + 过滤")
    print("=" * 78)
    print(f"{'查询':<34} {'top1(基)':>8} {'top1(增)':>8} {'Δ':>7} "
          f"{'过滤合规':>16} {'术语命中':>10}")
    print("-" * 78)
    for r in rows:
        if "error" in r:
            print(f"{r['query'][:32]:<34} ERROR {r['error'][:40]}")
            continue
        d = r["enhanced_top1"] - r["baseline_top1"]
        if r["filter_compliance_enhanced"] is None:
            comp = "—"
        else:
            comp = (f"{r['filter_compliance_baseline']:.0%} → "
                    f"{r['filter_compliance_enhanced']:.0%}")
        th = (f"{int(r['term_hit_baseline'])} → {int(r['term_hit_enhanced'])}")
        print(f"{r['query'][:32]:<34} {r['baseline_top1']:>8.4f} "
              f"{r['enhanced_top1']:>8.4f} {d:>+7.4f} {comp:>16} {th:>10}")

    ok = [r for r in rows if "error" not in r]
    if ok:
        mb = st.fmean(r["baseline_top1"] for r in ok)
        me = st.fmean(r["enhanced_top1"] for r in ok)
        print("-" * 78)
        print(f"{'平均 top1':<34} {mb:>8.4f} {me:>8.4f} {me - mb:>+7.4f}")
        filt = [r for r in ok if r["filter_compliance_enhanced"] is not None]
        if filt:
            print(f"过滤合规率(增强后): "
                  f"{st.fmean(r['filter_compliance_enhanced'] for r in filt):.0%}"
                  f"  (共 {len(filt)} 条带时间约束的查询)")
        print(f"术语命中: baseline {sum(r['term_hit_baseline'] for r in ok)}/{len(ok)}"
              f"  →  enhanced {sum(r['term_hit_enhanced'] for r in ok)}/{len(ok)}")


if __name__ == "__main__":
    main()
