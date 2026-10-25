"""上下文组装器的对照实验（生成模块第一部分）。

组装器里有四个"看不见"的机制：去重、多样性降权、预算截断、句末截断。
它们各写了十几行代码，但都可能在真实数据上形同虚设（比如 top-k 少的时候
去重根本不会触发）。所以每个机制都要单独构造场景验证，而不是跑一遍就信。

五个实验:
    --dedup          Jaccard 去重的阈值行为
    --diversity      同一文献多块时的降权是否生效
    --budget         不同 token 预算下的取舍与丢弃分布
    --truncate       截断是否落在句子边界
    --token-accuracy 估算值与 LLM 实际 token 数的偏差（需调用模型，较慢）

用法:
    python scripts/bench_context.py --all
    python scripts/bench_context.py --dedup --diversity
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from context_assembler import ContextAssembler, DocumentChunk   # noqa: E402

QUERY = "二甲双胍对心血管疾病有何影响？"
OUT_JSON = os.path.join(ROOT, "data", "chunks", "bench_context.json")
LOG_PATH = os.path.join(ROOT, "data", "chunks", "bench_context.log")


class Tee:
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


def _hdr(t: str) -> None:
    print()
    print("=" * 74)
    print(t)
    print("=" * 74)


def _mk(text: str, doc: str, idx: str, score: float) -> DocumentChunk:
    return DocumentChunk(text=text, metadata={"doc_id": doc, "pub_year": "2023",
                                              "journal": "J Test"},
                         relevance_score=score, source=doc,
                         chunk_id=f"{doc}_{idx}")


# ---------------------------------------------------------------- 1 去重
def exp_dedup() -> dict:
    _hdr("1) Jaccard 去重 —— 阈值 0.7 下, 近似-OCR 重复能否被识别")
    base = ("Metformin reduces hepatic glucose production and improves insulin "
            "sensitivity in patients with type 2 diabetes mellitus.")
    cases = {
        "完全相同": base,
        "仅差标点空格": base.replace(" ", "  ").replace(".", " ."),
        "轻微改写字序": ("Patients with type 2 diabetes mellitus benefit from "
                     "metformin via reduced hepatic glucose production and "
                     "improved insulin sensitivity."),
        "不同内容": ("Community-acquired pneumonia requires risk stratification "
                   "before choosing empirical antibiotic therapy duration."),
    }
    asm = ContextAssembler()
    print(f"{'变体':<12} {'Jaccard':>12}  {'判定(阈值0.7)':>14}")
    print("-" * 74)
    rows = []
    for name, t in cases.items():
        j = asm.jaccard(base, t)
        verdict = "重复(丢弃)" if j >= 0.7 else "保留"
        print(f"{name:<12} {j:>14.4f}  {verdict:>14}")
        rows.append({"case": name, "jaccard": round(j, 4), "dropped": j >= 0.7})

    # 端到端: 塞 4 条高度相似的进来
    chunks = [_mk(base, f"PMC{i:08d}", f"{i:04d}", 0.9 - i * 0.01) for i in range(1, 5)]
    kept = asm.deduplicate(chunks)
    print(f"\n端到端: 4 条相似块 -> 去重后 {len(kept)} 条")
    rows.append({"end_to_end": f"{len(chunks)}->{len(kept)}"})
    return {"dedup": rows}


# ---------------------------------------------------------------- 2 多样性
def exp_diversity() -> dict:
    _hdr("2) 多样性 —— 同一文献多块是否被降权")
    asm = ContextAssembler(source_penalty=0.6, max_per_source=3)
    doc_a = "Metformin and cardiovascular outcomes in type 2 diabetes: a cohort study."
    doc_b = "SGLT2 inhibitors reduce heart failure hospitalization in RCTs."
    chunks = [
        _mk(doc_a, "PMC11111111", "0001", 0.95),
        _mk(doc_a, "PMC11111111", "0002", 0.93),   # 同篇第 2 块
        _mk(doc_a, "PMC11111111", "0003", 0.92),   # 同篇第 3 块
        _mk(doc_a, "PMC11111111", "0004", 0.91),   # 第 4 块, 应被 max_per_source 截断
        _mk(doc_b, "PMC22222222", "0001", 0.80),
    ]
    # 纯按分数排
    by_score = sorted(chunks, key=lambda c: -c.relevance_score)
    print("纯按相关性排序:")
    for c in by_score:
        print(f"  {c.relevance_score:.2f}  {c.chunk_id}")

    sel = asm.select(chunks)
    print("\n经多样性降权后:")
    for c in sel:
        print(f"  {c.chunk_id}  (原分 {c.relevance_score:.2f})")
    at_b = [i for i, c in enumerate(sel) if c.doc_id == "PMC22222222"][0] + 1
    print(f"\n另一篇文献的块从第 5 位提到第 {at_b} 位 (未被完全挤出)")
    print(f"单篇最多 {max(sum(1 for c in sel if c.doc_id == d) for d in {c.doc_id for c in sel})} 块 "
          f"(上限 {asm.max_per_source})")
    return {"diversity": {"n_in": len(chunks), "n_out": len(sel),
                          "rank_of_other_doc": at_b,
                          "order": [c.chunk_id for c in sel]}}


# ---------------------------------------------------------------- 3 预算
def exp_budget(docs: list) -> dict:
    _hdr("3) token 预算 —— 不同预算下的取舍")
    print(f"{'预算':>6} {'入选':>5} {'估算token':>9} {'去重丢':>7} {'多样性丢':>8} {'预算丢':>7}")
    print("-" * 74)
    rows = []
    for budget in (400, 800, 1500, 2500, 4000):
        asm = ContextAssembler(max_tokens=budget)
        res = asm.assemble(docs)
        m = res["metadata"]
        print(f"{budget:>6} {m['chunks_selected']:>5} {m['estimated_tokens']:>9} "
              f"{m['dropped_by_dedup']:>7} {m['dropped_by_diversity']:>8} "
              f"{m['dropped_by_budget']:>7}")
        rows.append({"budget": budget, **{k: m[k] for k in
                     ("chunks_selected", "estimated_tokens", "dropped_by_dedup",
                      "dropped_by_diversity", "dropped_by_budget")}})
    return {"budget": rows}


# ---------------------------------------------------------------- 4 截断
def exp_truncate(docs: list) -> dict:
    _hdr("4) 句末截断 —— 是否落在完整句子边界")
    long_text = (" ".join(
        f"This sentence number {i} describes part of the mechanism under study."
        for i in range(1, 40)))
    asm = ContextAssembler(max_tokens=2500)
    print(f"原文估算 {asm.estimate_tokens(long_text)} tokens")
    print(f"{'预算':>6} {'截断后token':>10} {'末尾字符':>10} {'落在句末?':>10}")
    print("-" * 74)
    rows = []
    for budget in (100, 200, 400):
        out = asm._truncate_at_boundary(long_text, budget)
        tok = asm.estimate_tokens(out)
        tail = out[-1:] if out else ""
        ok = tail in ".!?。！？"
        print(f"{budget:>6} {tok:>10} {repr(tail):>10} {str(ok):>10}")
        rows.append({"budget": budget, "tokens": tok, "tail": tail,
                     "ends_at_sentence": ok})
    return {"truncate": rows}


# ---------------------------------------------------------------- 5 估算精度
def exp_token_accuracy(docs: list) -> dict:
    _hdr("5) token 估算精度 —— 估算值 vs LLM 实际计入的 token 数")
    asm = ContextAssembler(max_tokens=2500)
    res = asm.assemble(docs)
    ctx = res["context_text"]
    est = res["metadata"]["estimated_tokens"]

    import ollama
    print(f"上下文估算 {est} tokens, 字符数 {len(ctx)}")
    t = time.perf_counter()
    r = ollama.generate(model="deepseek-r1:7b", prompt=ctx,
                        options={"num_predict": 1, "num_ctx": 8192})
    dt = time.perf_counter() - t
    actual = getattr(r, "prompt_eval_count", 0) or 0
    err = (est - actual) / actual * 100 if actual else 0
    print(f"实际计入 {actual} tokens   ({dt:.1f}s)")
    print(f"估算偏差 {err:+.1f}%  {'（估算偏大=更安全, 不易超窗）' if err > 0 else '（偏小有超窗风险）'}")
    return {"token_accuracy": {"estimated": est, "actual": actual,
                               "error_pct": round(err, 1), "chars": len(ctx)}}


def main() -> None:
    ap = argparse.ArgumentParser(description="上下文组装器实验")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--dedup", action="store_true")
    ap.add_argument("--diversity", action="store_true")
    ap.add_argument("--budget", action="store_true")
    ap.add_argument("--truncate", action="store_true")
    ap.add_argument("--token-accuracy", action="store_true", help="需调用 LLM, 较慢")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--log", action="store_true")
    args = ap.parse_args()

    if args.log:
        sys.stdout = Tee(LOG_PATH)
    if args.all:
        for k in ("dedup", "diversity", "budget", "truncate", "token_accuracy"):
            setattr(args, k, True)
    if not any([args.dedup, args.diversity, args.budget, args.truncate,
                args.token_accuracy]):
        args.all = True
        args.dedup = args.diversity = args.budget = args.truncate = True
        args.token_accuracy = False

    # 只有这两个实验需要真实检索结果; truncate 用构造的长文本, 不必等 30 秒检索
    need_docs = args.budget or args.token_accuracy
    docs = []
    if need_docs:
        from retrieval_pipeline import RetrievalPipeline
        print(f"检索中: {QUERY}", flush=True)
        docs = RetrievalPipeline(rerank_candidates=40).search(QUERY, top_k=40)["results"]
        print(f"取回 {len(docs)} 条\n", flush=True)

    out: dict = {}
    if args.dedup:
        out.update(exp_dedup())
    if args.diversity:
        out.update(exp_diversity())
    if args.budget:
        out.update(exp_budget(docs))
    if args.truncate:
        out.update(exp_truncate(docs))
    if args.token_accuracy:
        out.update(exp_token_accuracy(docs))

    if args.json:
        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\nJSON 已保存 -> {OUT_JSON}")


if __name__ == "__main__":
    main()
