"""医学提示词的验证（生成模块与提示词工程第一部分）。

验证两件事:
  1. 模板本身: 变量齐不齐、渲染安不安全（证据文本里常有花括号，
     str.format 会直接抛 KeyError，所以这儿也一并验证）
  2. 端到端: 真实检索 → 组装上下文 → 调用模型，看模板能不能驱动模型
     产出符合要求的东西（带 [n] 引用、无证据时承认不知道）

延迟为什么写在代码里: deepseek-r1:7b 在 CPU 上约 6.3 tokens/s，thinking
无法关闭（实测 think=False 无效），单次生成 40~90 秒。所以默认只跑单阶段，
完整四阶段要给足时间预算（约 5 分钟）。

用法:
    python scripts/bench_prompts.py --check                   只检查模板(秒级)
    python scripts/bench_prompts.py --stage answer_generator  跑单个阶段
    python scripts/bench_prompts.py --full                    四阶段串联(约5分钟)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from context_assembler import ContextAssembler                    # noqa: E402
from medical_prompts import (DISCLAIMER, PIPELINE_ORDER,          # noqa: E402
                             PROMPT_STAGES, get_stage)

MODEL = "deepseek-r1:7b"
NUM_CTX = 8192          # 默认是 2048, 会静默截断上下文, 必须显式放大
OUT_JSON = os.path.join(ROOT, "data", "chunks", "bench_prompts.json")
QUERY = "二甲双胍对心血管疾病有何影响？"


def llm_call(system: str, user: str, temperature: float,
             max_tokens: int) -> dict:
    """调用模型，返回答案正文 / thinking / 延迟等。"""
    import ollama
    t = time.perf_counter()
    r = ollama.chat(
        model=MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        options={"temperature": temperature, "num_predict": max_tokens,
                 "num_ctx": NUM_CTX})
    dt = time.perf_counter() - t
    msg = r.message
    content = msg.content or ""
    thinking = getattr(msg, "thinking", None) or ""
    n = getattr(r, "eval_count", 0) or 0
    return {"content": content, "thinking": thinking, "eval_count": n,
            "seconds": round(dt, 1),
            "tokens_per_s": round(n / dt, 2) if dt else 0.0}


# ---------------------------------------------------------------- 1 模板检查
def check_templates() -> dict:
    print("=" * 74)
    print("模板检查")
    print("=" * 74)
    print(f"{'阶段':<20} {'变量':<42} {'渲染':>6}")
    print("-" * 74)
    rows = []
    # 故意混入花括号，验证渲染不会炸
    tricky = "[1] 来源: PMC1\n给药剂量为 p{0.05} 时观察到效应 {n=12}；见表 {tbl}."
    for key in PIPELINE_ORDER:
        st = get_stage(key)
        kw = {v: tricky if v == "context" else f"<{v}>"
              for v in st.required_vars}
        try:
            rendered = st.render(**kw)
            ok = "OK" if rendered else "空"
        except Exception as e:                       # noqa: BLE001
            ok = f"失败:{type(e).__name__}"
        print(f"{key:<20} {','.join(st.required_vars):<42} {ok:>6}")
        rows.append({"stage": key, "vars": st.required_vars, "render": ok})
    print(f"\n含花括号的证据文本渲染通过（{'{'} 不会被 format 当成占位符）")

    # 变量依赖链检查: 后一阶段要用的产出, 前一阶段必须产出
    print("\n依赖链:")
    chain = {"answer_generator": "evidence_assessment",
             "critical_reviewer": "draft_answer",
             "final_assembler": "draft_answer, review"}
    for k, v in chain.items():
        need = set(v.split(", "))
        have = set()
        for prev in PIPELINE_ORDER[:PIPELINE_ORDER.index(k)]:
            have |= set(PROMPT_STAGES[prev].required_vars)
        missing = need - {"draft_answer", "review", "evidence_assessment"}
        print(f"  {k:<20} 需要 {sorted(need)}  缺失: {sorted(missing) or '无'}")
        rows.append({"stage": k + "_deps", "missing": sorted(missing)})
    return {"templates": rows}


# ---------------------------------------------------------------- 2 单阶段
def run_stage(stage_key: str, top_k: int = 12) -> dict:
    st = get_stage(stage_key)
    print("=" * 74)
    print(f"单阶段生成: {stage_key} ({st.name})")
    print("=" * 74)

    print("检索并组装上下文...", flush=True)
    from retrieval_pipeline import RetrievalPipeline
    t = time.perf_counter()
    docs = RetrievalPipeline(rerank_candidates=top_k).search(QUERY, top_k=top_k)["results"]
    ctx = ContextAssembler(max_tokens=2500).assemble(docs)
    print(f"  候选 {len(docs)} 条, 上下文 {ctx['metadata']['estimated_tokens']} tokens "
          f"({time.perf_counter() - t:.1f}s)\n", flush=True)

    kw = {v: ctx["context_text"] if v == "context" else "" for v in st.required_vars}
    kw["question"] = QUERY
    if "evidence_assessment" in kw:
        kw["evidence_assessment"] = "（未单独调用评估器，此处按全部片段可用处理）"
    if "disclaimer" in kw:
        kw["disclaimer"] = DISCLAIMER

    print("生成中(CPU 上约 40~90 秒)...", flush=True)
    res = llm_call(st.system_prompt, st.render(**kw), st.temperature, st.max_tokens)
    print(f"  耗时 {res['seconds']}s  {res['eval_count']} tokens  "
          f"{res['tokens_per_s']} tokens/s")
    print(f"  thinking {len(res['thinking'])} 字符 / 答案 {len(res['content'])} 字符")
    print("\n--- 答案 ---")
    print(res["content"][:1200])

    # 引用检查: 答案里是否带 [n] 引用
    import re
    cites = re.findall(r"\[\d+\]", res["content"])
    print(f"\n引用编号出现 {len(cites)} 次: {sorted(set(cites))[:10]}")
    return {"stage": stage_key, "answer_len": len(res["content"]),
            "thinking_len": len(res["thinking"]), **res, "citations": sorted(set(cites))}


# ---------------------------------------------------------------- 3 四阶段
def run_full(top_k: int = 12) -> dict:
    print("=" * 74)
    print("四阶段串联（每阶段一次模型调用，预计 4~6 分钟）")
    print("=" * 74)
    from retrieval_pipeline import RetrievalPipeline
    docs = RetrievalPipeline(rerank_candidates=top_k).search(QUERY, top_k=top_k)["results"]
    ctx = ContextAssembler(max_tokens=2500).assemble(docs)
    context_text = ctx["context_text"]
    print(f"候选 {len(docs)} 条, 上下文 {ctx['metadata']['estimated_tokens']} tokens\n")

    state = {"question": QUERY, "context": context_text, "disclaimer": DISCLAIMER,
             "evidence_assessment": "", "draft_answer": "", "review": ""}
    out = {"stages": []}
    total = 0.0
    for key in PIPELINE_ORDER:
        st = get_stage(key)
        kw = {v: state.get(v, "") for v in st.required_vars}
        print(f"[{key}] 生成中...", flush=True)
        res = llm_call(st.system_prompt, st.render(**kw), st.temperature,
                       st.max_tokens)
        total += res["seconds"]
        msg = res["content"]
        if key == "evidence_evaluator":
            state["evidence_assessment"] = msg
        elif key == "answer_generator":
            state["draft_answer"] = msg
        elif key == "critical_reviewer":
            state["review"] = msg
        print(f"   {res['seconds']}s  {res['eval_count']} tokens  "
              f"输出 {len(msg)} 字符")
        out["stages"].append({"stage": key, "seconds": res["seconds"],
                              "eval_count": res["eval_count"],
                              "out_len": len(msg)})
        if key == "answer_generator":
            print("\n--- 初稿 ---")
            print(msg[:800])
    print(f"\n--- 最终答案 ---\n{state['review'][:1200] if False else msg[:1200]}")
    print(f"\n四阶段总耗时 {total:.1f}s")
    out["total_seconds"] = round(total, 1)
    out["final_answer"] = msg
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="医学提示词验证")
    ap.add_argument("--check", action="store_true", help="只检查模板(秒级)")
    ap.add_argument("--stage", default=None, help="跑单个阶段")
    ap.add_argument("--full", action="store_true", help="四阶段串联(慢)")
    ap.add_argument("--top-k", type=int, default=12)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    out: dict = {}
    if args.check or not (args.stage or args.full):
        out.update(check_templates())
    if args.stage:
        out[args.stage] = run_stage(args.stage, args.top_k)
    if args.full:
        out.update(run_full(args.top_k))

    if args.json:
        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\nJSON 已保存 -> {OUT_JSON}")


if __name__ == "__main__":
    main()
