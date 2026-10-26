"""上下文组装器（生成模块与提示词工程第一部分）。

为什么需要这一层 —— 检索给的东西不能直接塞进 prompt:
    1. 同一篇文献的相邻块内容高度重叠，全塞进去既浪费配额又稀释重点
    2. 检索 top-k 里可能 5 条有 3 条来自同一篇，答案视角会很窄
    3. 模型有上下文上限，超了会被**静默截断**（末段证据直接消失）

token 怎么算 —— 这是本模块唯一需要反复权衡的地方:
    Ollama 没有 /api/tokenize（实测 404），也没有本地 deepseek tokenizer
    （HF 主站大文件被 TLS 中断）。所以不能直接用分词器数。

    替代办法：反过来用 Ollama 的 prompt_eval_count 当标尺——发一段已知
    文本给 deepseek-r1:7b，看它实际算了几个 token，反推字符/token 比。
    实测校准值（scripts 下的探测脚本生成）:
        中文    1.57 字符/token
        英文    5.03 字符/token
    这比"统一按 4 字符/token"猜准得多，也比借用一个**别的模型**的分词器更
    贴近目标模型（专职人员请注意：跨模型的分词器没有意义，中文尤其明显）。

    代价：这是估算，不是精确值。所以默认留了余量（见 MAX_TOKENS 注释）。

用法:
    python scripts/context_assembler.py              # 用流水线结果演示组装
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Sequence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

# ------------------------------------------------------------------ 数据类
@dataclass
class DocumentChunk:
    """一块可用于组装上下文的文档。"""
    text: str
    metadata: Dict[str, Any]
    relevance_score: float
    source: str
    chunk_id: str

    @property
    def doc_id(self) -> str:
        """所属文献 ID：chunk_id 形如 PMC12345678_0003，取下划线前部分。"""
        cid = self.chunk_id or ""
        return cid.split("_")[0] if "_" in cid else cid

    @property
    def pub_year(self) -> str:
        return str(self.metadata.get("pub_year", "") or "")

    @property
    def journal(self) -> str:
        return str(self.metadata.get("journal", "") or "")

    @classmethod
    def from_result(cls, r: Dict[str, Any],
                    score_keys: Sequence[str] = ("rerank_score", "fused_score",
                                                 "relevance_score", "score")) -> "DocumentChunk":
        """把第二阶段流水线输出的一条结果转成 DocumentChunk。

        分数取值优先级：重排序 > 融合 > 原始相似度。重排序分数是最后一道
        质量关产出的，优先用它；没有重排序时退到融合分。
        """
        md = dict(r.get("metadata") or {})
        score = 0.0
        for k in score_keys:
            if k in r and r[k] is not None:
                score = float(r[k])
                break
        return cls(
            text=r.get("text", "") or "",
            metadata=md,
            relevance_score=score,
            source=str(md.get("doc_id") or r.get("chunk_id", "")),
            chunk_id=str(r.get("chunk_id", "") or ""),
        )


# ------------------------------------------------------------ token 估算
_ZH = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
_PUNCT_ZH = re.compile(r"[\u3002\uff1b\uff01\uff1f]")     # 。；！？
_SENT_END = re.compile(r"[.!?。！？；\n]")


class ContextAssembler:
    """把检索结果组装成给 LLM 的上下文字符串。"""

    # 实测校准的字符/token 比（用 deepseek-r1:7b 的 prompt_eval_count 反推）
    CHARS_PER_TOKEN_ZH = 1.57
    CHARS_PER_TOKEN_OTHER = 5.03

    # 安全系数: 上面两个比值是用"纯中/英文散文"校准的, 但真实上下文里混着
    # PMC11219383 这类文献号、年份、斜杠和编号表头。数字和符号会被分词器拆得
    # 比散文碎得多。实测: 一段 12042 字符的真实上下文, 按上面公式估 2418 token,
    # 模型实际计入 3147 token —— 低估 23.2%。低估会导致上下文超窗(而超窗是
    # 静默截断, 末段证据凭空消失)。所以乘安全系数, 宁可估大。
    SAFETY_FACTOR = 1.3

    def __init__(self, max_tokens: int = 2500, dedup_threshold: float = 0.7,
                 source_penalty: float = 0.6, max_per_source: int = 3,
                 min_chunk_tokens: int = 80, tokenizer: Any = None,
                 safety_factor: float = SAFETY_FACTOR):
        """
        Args:
            max_tokens: 上下文 token 预算。默认 2500 是双上限权衡后的结果——
                LLM 的 num_ctx 要显式设（默认 2048 会静默截断），而 prompt
                越长处理越慢（实测 ~19 tokens/s，1000 token 要 50 秒）。
                2500 留给 system prompt + 问题 + 答案仍有余量。
            dedup_threshold: Jaccard 相似度超过此值视为重复
            source_penalty: 同一来源第 n 块的分数惩罚系数（软惩罚而非硬截断）
            max_per_source: 单个来源最多保留几块
            min_chunk_tokens: 剩余预算低于此值就不再塞更多块
            tokenizer: 可选的真实分词器（有则优先用，无则用校准公式）
        """
        self.max_tokens = max_tokens
        self.dedup_threshold = dedup_threshold
        self.source_penalty = source_penalty
        self.max_per_source = max_per_source
        self.min_chunk_tokens = min_chunk_tokens
        self.safety_factor = safety_factor
        self._tokenizer = tokenizer

    # ------------------------------------------------------------ tokenizer
    def load_tokenizer(self, model_dir: str | None = None) -> bool:
        """尝试加载本地分词器（可选）。失败返回 False，自动回退到估算公式。"""
        model_dir = model_dir or os.path.join(ROOT, "models", "bge-reranker-base")
        if not os.path.isdir(model_dir):
            return False
        try:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(model_dir)
            return True
        except Exception:                            # noqa: BLE001
            return False

    def estimate_tokens(self, text: str) -> int:
        """估算 token 数。有真实分词器就用，否则用实测校准的字符比例。"""
        if not text:
            return 0
        if self._tokenizer is not None:
            try:
                return len(self._tokenizer.encode(text, add_special_tokens=False))
            except Exception:                        # noqa: BLE001
                pass
        zh = len(_ZH.findall(text))
        other = len(text) - zh
        return int((zh / self.CHARS_PER_TOKEN_ZH
                    + other / self.CHARS_PER_TOKEN_OTHER)
                   * self.safety_factor) + 1

    # ------------------------------------------------------------ 转换
    @staticmethod
    def convert(results: Iterable[Dict[str, Any]]) -> List[DocumentChunk]:
        """把流水线输出的多条结果转成 DocumentChunk 列表。"""
        out = []
        for r in results:
            if isinstance(r, DocumentChunk):
                out.append(r)
            else:
                out.append(DocumentChunk.from_result(r))
        return out

    # ------------------------------------------------------------ 去重
    @staticmethod
    def _shingles(text: str, n: int = 4) -> set:
        """字符 n-gram 集合。用字符而非词，是因为中文没有空格分词，
        且英文里 'metformin' / 'metformin.' 也应视为相同。"""
        t = re.sub(r"\s+", "", text.lower())
        if len(t) <= n:
            return {t} if t else set()
        return {t[i:i + n] for i in range(len(t) - n + 1)}

    def jaccard(self, a: str, b: str) -> float:
        sa, sb = self._shingles(a), self._shingles(b)
        if not sa or not sb:
            return 0.0
        inter = len(sa & sb)
        union = len(sa | sb)
        return inter / union if union else 0.0

    def deduplicate(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """贪心去重：按分数从高到低，与已保留块相似度超阈值的丢弃。

        为什么贪心而不是两两聚类：候选只有几十条，且分数高的优先保留本身
        就是想要的行为（保留最相关的那个版本）。
        """
        kept: List[DocumentChunk] = []
        kept_shingles = []
        for c in sorted(chunks, key=lambda x: -x.relevance_score):
            s = self._shingles(c.text)
            dup = False
            for ks in kept_shingles:
                if not s or not ks:
                    continue
                inter = len(s & ks)
                union = len(s | ks)
                if union and inter / union >= self.dedup_threshold:
                    dup = True
                    break
            if not dup:
                kept.append(c)
                kept_shingles.append(s)
        return kept

    # ------------------------------------------------------------ 选择
    def select(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """相关性 + 多样性：同一来源的第 n 块按 source_penalty^n 降权。

        用软惩罚而非硬限制，是为了避免"该文献确实贡献了高质量证据"时
        被一刀切掉；同时又能防止单一来源垄断上下文。
        """
        scored: List[tuple[float, DocumentChunk]] = []
        per_source: Dict[str, int] = {}
        for c in sorted(chunks, key=lambda x: -x.relevance_score):
            n_same = per_source.get(c.doc_id, 0)
            if n_same >= self.max_per_source:
                continue
            penalty = self.source_penalty ** n_same
            scored.append((c.relevance_score * penalty, c))
            per_source[c.doc_id] = n_same + 1
        scored.sort(key=lambda t: -t[0])
        return [c for _, c in scored]

    # ------------------------------------------------------------ 上下文构建
    def _truncate_at_boundary(self, text: str, budget_tokens: int) -> str:
        """截断到预算附近，并保证落在完整句子边界。

        任务书要求"在完整段落处截断，如在后 10% 文本中找到句号"：
        先按预算取字符数，再在末段 10% 区域里从后往前找句末标点；找不到就
        再往前找（整段内），仍找不到才硬截。
        """
        if self.estimate_tokens(text) <= budget_tokens:
            return text
        # 先按平均比例粗估能留多少字符
        zh_ratio = len(_ZH.findall(text)) / max(len(text), 1)
        # 必须除以安全系数: estimate_tokens 是"保守估算"(乘了 safety_factor),
        # 这里若按不含系数的真实比例去截, 截出来的文本用保守估算一量就会超出
        # 预算 —— 实测过: 预算 100 会截出 123 token, 等于截了个寂寞, 照样超窗。
        chars_per_token = ((self.CHARS_PER_TOKEN_ZH * zh_ratio
                            + self.CHARS_PER_TOKEN_OTHER * (1 - zh_ratio))
                           / self.safety_factor)
        keep = int(budget_tokens * chars_per_token)
        if keep <= 0:
            return ""
        candidate = text[:keep]

        tail_start = int(keep * 0.9)
        hit = -1
        for m in _SENT_END.finditer(candidate, tail_start):
            hit = m.end()
        if hit == -1:                                # 末段没句号，整段找
            for m in _SENT_END.finditer(candidate):
                hit = m.end()
        return candidate[:hit] if hit > 0 else candidate.rstrip()

    def build_context(self, chunks: List[DocumentChunk]) -> tuple[str, List[DocumentChunk], int]:
        """逐块累加直到用完预算。返回 (上下文文本, 实际入选块, token 数)。"""
        parts: List[str] = []
        used: List[DocumentChunk] = []
        total = 0
        for i, c in enumerate(chunks, 1):
            header = f"[{i}] 来源: {c.source}"
            meta_bits = [b for b in (c.pub_year, c.journal) if b]
            if meta_bits:
                header += f"（{' / '.join(meta_bits)}）"
            block = f"{header}\n{c.text.strip()}\n"
            need = self.estimate_tokens(block)
            remain = self.max_tokens - total
            if need <= remain:
                parts.append(block)
                used.append(c)
                total += need
                continue
            # 这块塞不下：如果剩余还够最小块，就尝试截断后塞入，否则停
            if remain >= self.min_chunk_tokens:
                budget = remain - self.estimate_tokens(header) - 1
                body = self._truncate_at_boundary(c.text, budget)
                if body.strip():
                    block = f"{header}\n{body.strip()}\n"
                    need = self.estimate_tokens(block)
                    if need <= remain:
                        parts.append(block)
                        used.append(c)
                        total += need
            break
        return "\n".join(parts).strip(), used, total

    # ------------------------------------------------------------ 统计
    @staticmethod
    def _analyze_sources(chunks: List[DocumentChunk]) -> Dict[str, Any]:
        per: Dict[str, int] = {}
        years: List[str] = []
        for c in chunks:
            per[c.doc_id] = per.get(c.doc_id, 0) + 1
            if c.pub_year:
                years.append(c.pub_year)
        return {
            "unique_docs": len(per),
            "per_doc": dict(sorted(per.items(), key=lambda kv: -kv[1])),
            "year_range": [min(years), max(years)] if years else [],
        }

    # ------------------------------------------------------------ 主入口
    def assemble(self, retrieved_docs: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        raw = list(retrieved_docs)
        chunks = self.convert(raw)
        unique = self.deduplicate(chunks)
        selected = self.select(unique)
        final_context, used, tok = self.build_context(selected)

        metadata = {
            "total_chunks_retrieved": len(raw),
            "unique_chunks_after_dedup": len(unique),
            "chunks_selected": len(used),
            "estimated_tokens": self.estimate_tokens(final_context),
            "chunk_sources": self._analyze_sources(used),
            "token_budget": self.max_tokens,
            "dropped_by_dedup": len(chunks) - len(unique),
            "dropped_by_diversity": len(unique) - len(selected),
            "dropped_by_budget": len(selected) - len(used),
        }
        return {
            "context_text": final_context,
            "metadata": metadata,
            "selected_chunks": used,
        }


def main() -> None:
    ap = argparse.ArgumentParser(description="上下文组装器")
    ap.add_argument("query", nargs="?", default="二甲双胍对心血管疾病有何影响？")
    ap.add_argument("--top-k", type=int, default=12)
    ap.add_argument("--max-tokens", type=int, default=2500)
    ap.add_argument("--no-rerank", action="store_true")
    args = ap.parse_args()

    from retrieval_pipeline import RetrievalPipeline
    pipe = RetrievalPipeline(rerank_candidates=args.top_k, use_rerank=not args.no_rerank)
    out = pipe.search(args.query, top_k=args.top_k)
    docs = out["results"]

    asm = ContextAssembler(max_tokens=args.max_tokens)
    res = asm.assemble(docs)
    md = res["metadata"]
    print(f"查询: {args.query}")
    print(f"检索 {md['total_chunks_retrieved']} 条 -> 去重后 {md['unique_chunks_after_dedup']} "
          f"(去掉 {md['dropped_by_dedup']}) -> 多样性筛选 {md['chunks_selected']} "
          f"(去掉 {md['dropped_by_diversity']}) -> 预算内 {len(res['selected_chunks'])} "
          f"(去掉 {md['dropped_by_budget']})")
    print(f"估算 token: {md['estimated_tokens']} / 预算 {md['token_budget']}")
    print(f"来源分布: {md['chunk_sources']['per_doc']}  年份范围: {md['chunk_sources']['year_range']}")
    print("\n--- 上下文前 900 字符 ---")
    print(res["context_text"][:900])
    print("\n--- metadata ---")
    print(json.dumps(md, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
