"""医学问答生成流水线（生成模块与提示词工程第二部分）。

把前面所有组件串起来:
    检索 → 上下文组装 → (证据评估+筛选) → 答案草稿 → (批判性审查) → 终稿 → 后处理

三条降级原则（都是实测逼出来的，不是泛泛的"健壮性"）:

1. **评估解析失败就不筛选，保留全部上下文**。
   实测证据评估器有过"无可用证据"的误报，若照它筛，上下文会被清空、
   后面直接无米下锅。所以"没解析出可用编号"和"确实没有可用证据"都按
   保留全部处理，并记 warning —— 宁可多用几条证据，也不能把流程搞崩。

2. **审查失败就用草稿**。任务书明确要求，且审查阶段输入最长、最容易超时。

3. **终稿失败也退回草稿 + 后处理**。后处理（引用、来源清单、免责声明）
   是代码做的，不依赖模型，所以总能给出一个结构完整的结果。

用法:
    python scripts/generation_pipeline.py "二甲双胍对心血管疾病有何影响？" --fast
    python scripts/generation_pipeline.py "二甲双胍对心血管疾病有何影响？"
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from context_assembler import ContextAssembler, DocumentChunk     # noqa: E402
from llm_generator import LLMGenerator                            # noqa: E402
from medical_prompts import DISCLAIMER, PROMPT_STAGES, get_stage  # noqa: E402


class MedicalGenerationPipeline:
    """整合上下文组装、医学模板与本地 LLM 的生成流水线。"""

    def __init__(self, llm: Optional[LLMGenerator] = None,
                 assembler: Optional[ContextAssembler] = None,
                 use_evaluation: bool = True, use_review: bool = True,
                 top_k: int = 12, max_context_tokens: int = 2500,
                 use_rerank: bool = True):
        """
        Args:
            use_evaluation: 是否跑"证据评估 + 按评估筛选上下文"（可选，省时）
            use_review: 是否跑"批判性审查"（可选，省时）
            top_k: 检索候选数
            max_context_tokens: 上下文 token 预算
        """
        self.llm = llm or LLMGenerator()
        self.assembler = assembler or ContextAssembler(max_tokens=max_context_tokens)
        self.use_evaluation = use_evaluation
        self.use_review = use_review
        self.top_k = top_k
        self.use_rerank = use_rerank
        self._retriever = None            # 延迟加载：检索要加载 278M 重排模型

    # ------------------------------------------------------------ 检索
    def _retrieve(self, query: str) -> List[Dict[str, Any]]:
        if self._retriever is None:
            from retrieval_pipeline import RetrievalPipeline
            self._retriever = RetrievalPipeline(rerank_candidates=self.top_k,
                                                use_rerank=self.use_rerank)
        return self._retriever.search(query, top_k=self.top_k)["results"]

    # ------------------------------------------------------------ 评估解析
    @staticmethod
    def parse_usable_ids(eval_text: str) -> Optional[List[int]]:
        """从评估输出里解析"可用证据编号"。

        返回 None 表示**没解析出来**（调用方应保留全部上下文），
        返回 [] 表示模型明确说没有可用证据（调用方同样保留全部并告警）。
        两者区分开，是为了日志里能看出到底是哪种情况。
        """
        if not eval_text:
            return None
        # 1) JSON 形态
        obj = LLMGenerator.extract_json(eval_text)
        if isinstance(obj, dict):
            for key in ("usable_ids", "available_ids", "usable", "usable_evidence_ids"):
                if key in obj and isinstance(obj[key], list):
                    ids = [int(x) for x in obj[key] if str(x).strip().isdigit()]
                    return ids
        # 2) "可用证据编号：3,5,12" 或 "可用证据编号：无可用证据"
        m = re.search(r"可用证据编号\s*[：:]\s*([^\n]+)", eval_text)
        if m:
            tail = m.group(1).strip()
            if "无" in tail and not re.search(r"\d", tail):
                return []
            ids = [int(x) for x in re.findall(r"\d+", tail)]
            return ids
        # 3) 逐行 "编号 - 判定 - 理由"（v3 简化模板的输出）
        rows = re.findall(r"^\s*(\d+)\s*[-–—]\s*([^\-\n]+?)\s*[-–—]", eval_text, re.M)
        if rows:
            ids = [int(n) for n, verdict in rows
                   if ("支持" in verdict and "不支持" not in verdict)]
            return ids
        return None

    def _filter_context(self, context_result: Dict[str, Any],
                        usable_ids: Optional[List[int]],
                        eval_text: str) -> Dict[str, Any]:
        """按评估结果筛选上下文（基于编号 / 文档id，并检测标题行）。

        上下文里每块的标题行形如 "[3] 来源: PMCxxxx（2023 / Journal）"，
        selected_chunks 的顺序与编号一一对应，因此按编号筛即可；
        若编号对不上，退回按 doc_id 匹配标题行做兜底。
        """
        chunks: List[DocumentChunk] = context_result["selected_chunks"]
        before = len(chunks)
        note = ""
        if not usable_ids:
            note = ("评估未给出可用编号（解析失败或模型称无可用证据），"
                    "保留全部上下文")
            context_result["metadata"]["filter_note"] = note
            context_result["metadata"]["chunks_after_eval_filter"] = before
            return context_result

        n = len(chunks)
        keep = [(i, c) for i, c in enumerate(chunks, 1) if i in set(usable_ids)]
        if not keep:                        # 编号与块对不上，别筛空
            note = f"评估编号 {usable_ids} 与上下文 {n} 块对不上，保留全部"
            context_result["metadata"]["filter_note"] = note
            context_result["metadata"]["chunks_after_eval_filter"] = before
            return context_result

        parts = []
        for orig_i, c in keep:
            # 保留原编号, 不要重新编号: 评估文本里的编号是筛选前的, 若重排成 1..k,
            # 模型会在答案里引用旧编号(实测出现过 [12], 而筛选后只剩 8 块),
            # 造成引用悬空。编号允许不连续([3][4][8]...), 与评估结果保持一致。
            c.metadata["_ref"] = orig_i
            meta_bits = [b for b in (c.pub_year, c.journal) if b]
            header = f"[{orig_i}] 来源: {c.source}"
            if meta_bits:
                header += f"（{' / '.join(meta_bits)}）"
            parts.append(f"{header}\n{c.text.strip()}\n")
        new_text = "\n".join(parts).strip()

        kept_chunks = [c for _, c in keep]
        note = f"按评估筛选: {before} -> {len(kept_chunks)} 块（可用编号 {usable_ids}）"
        context_result["selected_chunks"] = kept_chunks
        context_result["context_text"] = new_text
        context_result["metadata"]["chunks_after_eval_filter"] = len(kept_chunks)
        context_result["metadata"]["estimated_tokens"] = self.assembler.estimate_tokens(new_text)
        context_result["metadata"]["filter_note"] = note
        context_result["metadata"]["chunk_sources"] = self.assembler._analyze_sources(kept_chunks)
        return context_result

    # ------------------------------------------------------------ 后处理
    def _postprocess(self, answer: str, sources: List[Dict[str, Any]]) -> str:
        """引用整理、格式美化、来源清单与免责声明（纯代码，不依赖模型）。"""
        text = (answer or "").strip()
        text = re.sub(r"\n{3,}", "\n\n", text)               # 多余空行
        text = re.sub(r"[ \t]+\n", "\n", text)                # 行尾空格
        # 引用标记统一成半角 [n]，模型偶尔输出全角【n】
        text = re.sub(r"【\s*(\d+)\s*】", r"[\1]", text)

        if sources and "参考来源" not in text and "来源：" not in text[-600:]:
            lines = ["", "参考来源："]
            for s in sources:
                bits = [b for b in (s.get("pub_year"), s.get("journal")) if b]
                suffix = f"（{' / '.join(bits)}）" if bits else ""
                lines.append(f"[{s['ref']}] {s['doc_id']}{suffix}")
            text += "\n" + "\n".join(lines)
        if "不构成临床诊断" not in text and "免责声明" not in text:
            text += "\n\n免责声明：\n" + DISCLAIMER
        return text.strip()

    @staticmethod
    def _format_sources(chunks: List[DocumentChunk]) -> List[Dict[str, Any]]:
        out = []
        for i, c in enumerate(chunks, 1):
            out.append({
                # 筛选后 _ref 保留原编号, 与答案里的 [n] 引用一致
                "ref": c.metadata.get("_ref", i),
                "chunk_id": c.chunk_id, "doc_id": c.doc_id,
                "pmid": c.metadata.get("pmid", ""),
                "journal": c.journal, "pub_year": c.pub_year,
                "score": round(float(c.relevance_score), 4),
            })
        return out

    # ------------------------------------------------------------ 主流程
    def generate(self, query: str,
                 docs: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        t0 = time.perf_counter()
        stage_times: Dict[str, float] = {}
        token_counts: Dict[str, int] = {}
        stage_success: Dict[str, bool] = {}
        intermediate = {"evidence_evaluation": "", "draft_answer": "",
                        "review_feedback": ""}
        warnings: List[str] = []

        def mark(name: str, t_start: float, res: Optional[Dict[str, Any]] = None) -> None:
            stage_times[name] = round(time.perf_counter() - t_start, 1)
            if res is not None:
                token_counts[name] = res.get("eval_count", 0)
                stage_success[name] = bool(res.get("success"))

        # 1. 检索 + 上下文组装
        t = time.perf_counter()
        if docs is None:
            docs = self._retrieve(query)
        context_result = self.assembler.assemble(docs)
        stage_times["retrieve_and_assemble"] = round(time.perf_counter() - t, 1)
        stage_success["retrieve_and_assemble"] = bool(context_result["selected_chunks"])
        if not context_result["selected_chunks"]:
            return self._empty_result(query, context_result, stage_times,
                                      token_counts, stage_success, intermediate,
                                      t0, "上下文为空：检索或组装阶段没有得到任何文档块")

        # 2. 证据评估（可选）+ 筛选上下文
        if self.use_evaluation:
            t = time.perf_counter()
            st = get_stage("evidence_evaluator")
            res = self.llm.generate_json(
                st.render(context=context_result["context_text"], question=query),
                system_prompt=st.system_prompt, temperature=st.temperature,
                max_tokens=st.max_tokens)
            mark("evidence_evaluation", t, res)
            intermediate["evidence_evaluation"] = res["content"]
            ids = self.parse_usable_ids(res["content"])
            context_result = self._filter_context(context_result, ids, res["content"])
            note = context_result["metadata"].get("filter_note", "")
            if note:
                warnings.append(note)
        else:
            stage_success["evidence_evaluation"] = False

        # 3. 答案草稿
        t = time.perf_counter()
        st = get_stage("answer_generator")
        res = self.llm.generate(
            st.render(context=context_result["context_text"], question=query,
                      evidence_assessment=intermediate["evidence_evaluation"]
                      or "（未单独评估，按全部片段可用处理）"),
            system_prompt=st.system_prompt, temperature=st.temperature,
            max_tokens=st.max_tokens)
        mark("draft_answer", t, res)
        draft = res["content"].strip()
        intermediate["draft_answer"] = draft

        # 4. 批判性审查（可选）
        review_text = ""
        if self.use_review and draft:
            t = time.perf_counter()
            st = get_stage("critical_reviewer")
            res = self.llm.generate(
                st.render(context=context_result["context_text"], question=query,
                          draft_answer=draft),
                system_prompt=st.system_prompt, temperature=st.temperature,
                max_tokens=st.max_tokens)
            mark("critical_review", t, res)
            review_text = res["content"].strip()
            intermediate["review_feedback"] = review_text
        else:
            stage_success["critical_review"] = False

        # 5. 终稿：有审查意见就走定稿阶段，否则直接用草稿
        final_answer = draft
        if self.use_review and review_text and draft:
            t = time.perf_counter()
            st = get_stage("final_assembler")
            res = self.llm.generate(
                st.render(context=context_result["context_text"], question=query,
                          draft_answer=draft, review=review_text,
                          disclaimer=DISCLAIMER),
                system_prompt=st.system_prompt, temperature=st.temperature,
                max_tokens=st.max_tokens)
            mark("final_assembly", t, res)
            if res["content"].strip():
                final_answer = res["content"].strip()
            else:
                warnings.append("定稿阶段未产出内容，退回草稿")
                stage_success["final_assembly"] = False
        else:
            stage_success["final_assembly"] = False

        # 6. 后处理 + 组装结果
        sources = self._format_sources(context_result["selected_chunks"])
        answer = self._postprocess(final_answer, sources)
        total = round(time.perf_counter() - t0, 1)

        return {
            "query": query,
            "answer": answer,
            "answer_length": len(answer),
            "context_metadata": context_result["metadata"],
            "generation_metrics": {
                "total_time_seconds": total,
                "stage_times": stage_times,
                "token_counts": token_counts,
                "stage_success": stage_success,
            },
            "intermediate_results": intermediate,
            "sources": sources,
            "warnings": warnings,
            "config": {"use_evaluation": self.use_evaluation,
                       "use_review": self.use_review, "top_k": self.top_k},
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    def _empty_result(self, query, context_result, stage_times, token_counts,
                      stage_success, intermediate, t0, reason) -> Dict[str, Any]:
        return {"query": query, "answer": reason, "answer_length": 0,
                "context_metadata": context_result.get("metadata", {}),
                "generation_metrics": {
                    "total_time_seconds": round(time.perf_counter() - t0, 1),
                    "stage_times": stage_times, "token_counts": token_counts,
                    "stage_success": stage_success},
                "intermediate_results": intermediate, "sources": [],
                "warnings": [reason], "config": {},
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}


# ------------------------------------------------------------ 展示
def print_result(r: Dict[str, Any], show_intermediate: bool = False) -> None:
    m = r["generation_metrics"]
    print("=" * 74)
    print(f"查询: {r['query']}")
    print(f"总耗时 {m['total_time_seconds']}s   答案 {r.get('answer_length', 0)} 字符")
    print("-" * 74)
    print(f"{'阶段':<24} {'耗时(s)':>8} {'tokens':>7} {'成功':>6}")
    for k in m["stage_times"]:
        print(f"{k:<24} {m['stage_times'][k]:>8} "
              f"{m['token_counts'].get(k, 0):>7} "
              f"{str(m['stage_success'].get(k, False)):>6}")
    print("-" * 74)
    cm = r["context_metadata"]
    print(f"上下文: 检索 {cm.get('total_chunks_retrieved')} → 入选 "
          f"{cm.get('chunks_selected')} → 评估后 {cm.get('chunks_after_eval_filter', '-')}   "
          f"估算 {cm.get('estimated_tokens')} tokens")
    if r.get("warnings"):
        for w in r["warnings"]:
            print(f"  ! {w}")
    print("\n--- 答案 ---")
    print(r["answer"][:1500])
    print("\n--- 引用来源 ---")
    for s in r["sources"]:
        print(f"  [{s['ref']}] {s['doc_id']}  {s['pub_year']}  {s['journal']}  "
              f"score={s['score']}")
    if show_intermediate:
        for k, v in r["intermediate_results"].items():
            print(f"\n--- 中间结果 {k} ({len(v)} 字符) ---")
            print(v[:600])


def main() -> None:
    ap = argparse.ArgumentParser(description="医学问答生成流水线")
    ap.add_argument("query", nargs="?", default="二甲双胍对心血管疾病有何影响？")
    ap.add_argument("--fast", action="store_true", help="只跑草稿(跳过评估与审查)")
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--no-review", action="store_true")
    ap.add_argument("--top-k", type=int, default=12)
    ap.add_argument("--intermediate", action="store_true", help="打印中间结果")
    ap.add_argument("--json", default=None, help="结果写入 JSON 文件")
    args = ap.parse_args()

    use_eval = not (args.fast or args.no_eval)
    use_review = not (args.fast or args.no_review)
    pipe = MedicalGenerationPipeline(use_evaluation=use_eval, use_review=use_review,
                                     top_k=args.top_k)
    print(f"配置: 评估={use_eval} 审查={use_review} top_k={args.top_k}", flush=True)
    r = pipe.generate(args.query)
    print_result(r, show_intermediate=args.intermediate)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(r, f, ensure_ascii=False, indent=2)
        print(f"\nJSON 已保存 -> {args.json}")


if __name__ == "__main__":
    main()
