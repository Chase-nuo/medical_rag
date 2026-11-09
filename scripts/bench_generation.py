"""完整生成流程的测试与日志（生成模块与提示词工程第二部分 · 任务3）。

跑若干条测试 query，记录每条的关键指标与引用来源:
    总耗时 / 各阶段耗时 / 各阶段 token / 答案长度 / 阶段成败 / 引用来源 / 降级警告

延迟提醒（实测）: fast 模式(只跑草稿)每条约 3 分钟；完整四阶段每条约 9 分钟。
所以默认 fast + 2 条，避免一跑半小时。

用法:
    python scripts/bench_generation.py                    # fast, 2 条
    python scripts/bench_generation.py --n 1              # 只跑1条
    python scripts/bench_generation.py --full --n 1       # 完整四阶段
    python scripts/bench_generation.py --log --json       # 落盘日志与结构化结果
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from generation_pipeline import MedicalGenerationPipeline   # noqa: E402

TEST_QUERIES = [
    "二甲双胍对心血管疾病有何影响？",
    "SGLT2 inhibitor heart failure hospitalization",
    "MI 患者的死亡率 2020年以后的研究",
]

LOG_PATH = os.path.join(ROOT, "data", "chunks", "bench_generation.log")
OUT_JSON = os.path.join(ROOT, "data", "chunks", "bench_generation.json")


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


def main() -> None:
    ap = argparse.ArgumentParser(description="完整生成流程测试")
    ap.add_argument("--n", type=int, default=2, help="跑几条测试query")
    ap.add_argument("--full", action="store_true", help="完整四阶段(每条约9分钟)")
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--no-review", action="store_true")
    ap.add_argument("--top-k", type=int, default=12)
    ap.add_argument("--log", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.log:
        sys.stdout = Tee(LOG_PATH)

    use_eval = not (args.no_eval or not args.full)
    use_review = not (args.no_review or not args.full)
    mode = "完整四阶段" if args.full else "fast(仅草稿)"
    queries = TEST_QUERIES[:args.n]

    print("=" * 78)
    print(f"完整生成流程测试  模式={mode}  top_k={args.top_k}  共 {len(queries)} 条")
    print(f"开始时间 {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 78)

    pipe = MedicalGenerationPipeline(use_evaluation=use_eval, use_review=use_review,
                                     top_k=args.top_k)
    results = []
    for i, q in enumerate(queries, 1):
        print(f"\n{'#' * 78}\n# [{i}/{len(queries)}] {q}\n{'#' * 78}", flush=True)
        t = time.perf_counter()
        r = pipe.generate(q)
        wall = round(time.perf_counter() - t, 1)
        m = r["generation_metrics"]
        cm = r["context_metadata"]

        print(f"\n总耗时 {m['total_time_seconds']}s (墙钟 {wall}s)   "
              f"答案 {r['answer_length']} 字符")
        print(f"{'阶段':<24} {'耗时(s)':>8} {'tokens':>7} {'成功':>6}")
        for k, v in m["stage_times"].items():
            print(f"{k:<24} {v:>8} {m['token_counts'].get(k, 0):>7} "
                  f"{str(m['stage_success'].get(k, False)):>6}")
        print(f"上下文: 检索 {cm.get('total_chunks_retrieved')} → 入选 "
              f"{cm.get('chunks_selected')} → 评估后 "
              f"{cm.get('chunks_after_eval_filter', '-')}   "
              f"估算 {cm.get('estimated_tokens')} tokens")
        for w in r.get("warnings", []):
            print(f"  ! {w}")

        print("\n--- 答案(前 900 字符) ---")
        print(r["answer"][:900])
        print("\n--- 引用来源 ---")
        for s in r["sources"]:
            print(f"  [{s['ref']}] {s['doc_id']}  {s['pub_year']}  {s['journal']}  "
                  f"score={s['score']}")
        if r["intermediate_results"].get("evidence_evaluation"):
            print("\n--- 证据评估(前 400 字符) ---")
            print(r["intermediate_results"]["evidence_evaluation"][:400])
        results.append(r)
        print("", flush=True)

    # 汇总
    print("\n" + "=" * 78)
    print("汇总")
    print("=" * 78)
    print(f"{'查询':<34} {'总耗时':>8} {'答案长度':>8} {'来源数':>6} {'生成tokens':>10}")
    for r in results:
        q = r["query"][:32]
        m = r["generation_metrics"]
        print(f"{q:<34} {m['total_time_seconds']:>8} {r['answer_length']:>8} "
              f"{len(r['sources']):>6} {sum(m['token_counts'].values()):>10}")
    if results:
        avg = sum(r["generation_metrics"]["total_time_seconds"] for r in results) / len(results)
        print(f"\n平均每条 {avg:.1f}s   模式={mode}")
        fails = [r["query"] for r in results
                 if not r["generation_metrics"]["stage_success"].get("draft_answer")]
        print(f"草稿阶段失败: {len(fails)} 条")
        warned = sum(1 for r in results if r.get("warnings"))
        print(f"触发降级警告: {warned} 条")

    if args.json:
        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump({"mode": mode, "top_k": args.top_k,
                       "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "results": results}, f, ensure_ascii=False, indent=2)
        print(f"\nJSON 已保存 -> {OUT_JSON}")
    if args.log:
        print(f"日志已保存 -> {LOG_PATH}")


if __name__ == "__main__":
    main()
