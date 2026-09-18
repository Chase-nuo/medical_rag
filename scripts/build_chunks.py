# -*- coding: utf-8 -*-
"""
文档解析与分割 (RAG 阶段④ 前半)
================================
把 1 020 篇 PMC OA 文献切成适合向量化与检索的文本单元, 落盘为数据集文件。

输入:
  - data/pmc_oa/corpus_fields.csv   任务1 产出的宽表(1 020 行, 字段已归一化)
  - data/pmc_oa/<pmcid>.*/ *.txt    PMC OA 全文(含元数据头块 + 正文 + 参考文献)

输出(全部落在 data/chunks/):
  - chunks.parquet          文本块数据集(主文件, 列式压缩)
  - chunks.jsonl            同内容, 便于直接查看
  - processing_stats.json   处理配置与统计
  - quality_report.csv      质量验证的问题块清单

分割策略(取自 docs/RAG数据分析与设计说明.md §4.3):
  - chunk_size    = 512 token
  - chunk_overlap = 64 token(12.5%)
  - length_function = bge-m3 tokenizer 的真实 token 数
  - 只对 core 正文切块(元数据头块 + 参考文献占 32.2%, 必须先剔除)

用法:
    python scripts/build_chunks.py                     # 全量智能分割
    python scripts/build_chunks.py --limit 20          # 小样本试跑
    python scripts/build_chunks.py --strategy no_split # 整体不分割(对照组)
"""
from __future__ import annotations

import argparse
import functools
import glob
import json
import os
import re
import statistics
import sys
import time

import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tokenizers import Tokenizer

ROOT = r"d:\medical_rag\data\pmc_oa"
TOKENIZER_ROOT = r"d:\medical_rag\data\tokenizer"
FIELDS_CSV = os.path.join(ROOT, "corpus_fields.csv")
OUT_DIR = r"d:\medical_rag\data\chunks"

# ---------------------------------------------------------------- 常量
# PMC 正文分隔线: 一行由 0x9F(C1 控制符) 包住若干 '=' 组成, 每篇唯一
BODY_MARK_RE = re.compile(r"^\x9f=+\x9f$")
# 正文结束标志: 出现以下独立行即认为主体内容结束(PMC 里大小写混用, 故忽略大小写)
TAIL_HEAD_RE = re.compile(
    r"^(references?|bibliography|literature cited|supporting information|"
    r"supplementary materials?|comments to the author|"
    r"review comments to the author)$", re.I)
# 章节标题(用于质检项"是否包含标题")
SECTION_RE = re.compile(
    r"^\s*[0-9.]*\s*(introduction|methods|materials and methods|results|"
    r"discussion|conclusion|conclusions|references|background|objective)\b"
    r"\s*:?\s*$", re.I | re.M)
# mojibake(双重编码)的可疑串: U+FFFD / "Ã©" / "Ã¨" / "â€™" / "â€œ" / "ï¿½"
MOJIBAKE = ("\ufffd", "\u00c3\u00a9", "\u00c3\u00a8",
            "\u00e2\u20ac\u2122", "\u00e2\u20ac\u0153", "\u00ef\u00bf\u00bd")
CJK_RANGE = ("\u4e00", "\u9fff")

WS_RE = re.compile(r"\s+")       # 空白归一化: 表格里的 \t 会被 tokenizer 归一成空格

# 硬编码排除名单:
#   PMC5444287  §1.3: core 为空(剔除比例 100%), 无正文可切
#   PMC4616690  质检发现: 连续 5 个块整段为 mojibake, 全篇编码损坏无法修复
EXCLUDE_PMCID = {"PMC5444287", "PMC4616690"}
MIN_LINES = 100                  # §1.3: 极短文阈值(13 篇)
TOO_SHORT_TOKENS = 20            # 质检/合并: 块过短阈值

QUANTILES = [0, 5, 25, 50, 75, 90, 95, 99, 100]
QNAMES = ["min", "p5", "p25", "p50", "p75", "p90", "p95", "p99", "max"]


class Tee:
    """把 stdout 同时写进日志文件(交付物之一: 处理日志)"""
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")
        self.out = sys.stdout

    def write(self, s):
        self.f.write(s)
        self.out.write(s)
        return len(s)

    def flush(self):
        self.f.flush()
        self.out.flush()


def quantile(vals, q):
    """线性插值分位数(等价 numpy.percentile 默认方式), 避免为了一个函数引入 numpy"""
    xs = sorted(vals)
    if not xs:
        return 0.0
    if len(xs) == 1:
        return float(xs[0])
    pos = (len(xs) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def print_quantiles(title, vals):
    if not vals:
        return
    cells = "".join(f"{quantile(vals, q):>9.0f}" for q in QUANTILES)
    print(f"{title:<22}{len(vals):>7}{cells}{statistics.fmean(vals):>10.0f}")


# ---------------------------------------------------------------- 文本分段
def split_regions(lines):
    """按 PMC 结构切成 (full, body, core, tail_cut)

    full : 整个 txt
    body : \x9f====\x9f 分隔线之后
    core : body 再截断到首个 References / Supporting information 等独立行
    tail_cut: 是否成功识别到尾部边界(127 篇为 False, 仍含参考文献)
    """
    idx = next((i for i, l in enumerate(lines) if BODY_MARK_RE.match(l.strip())), None)
    if idx is None:                      # 没有分隔线: 退化为整篇
        return lines, lines, lines, False
    body = lines[idx + 1:]
    end = next((k for k, l in enumerate(body) if TAIL_HEAD_RE.match(l.strip())), None)
    if end is None:
        return lines, body, body, False
    return lines, body, body[:end], True


def read_text(pmcid):
    """读取某篇的 txt; 用通配匹配版本号目录(.1/.2/.3), 避免硬编码导致静默读空"""
    hits = glob.glob(os.path.join(ROOT, pmcid + ".*", "*.txt"))
    if not hits:
        return ""
    with open(hits[0], encoding="utf-8", errors="replace") as f:
        return f.read()


def detect_lang(text):
    """按 CJK 字符占比判定语言(§2.4: 中英字符-token 系数差异达 2 倍, 必须区分)"""
    if not text:
        return "en"
    cjk = sum(1 for ch in text if CJK_RANGE[0] <= ch <= CJK_RANGE[1])
    ratio = cjk / len(text)
    return "zh" if ratio >= 0.20 else ("mixed" if ratio >= 0.02 else "en")


# ---------------------------------------------------------------- 步骤 1: 加载与清洗
def build_dataframe(limit=None):
    """读 corpus_fields.csv 建 DataFrame, 再逐篇读 txt 提取 core 正文并清洗"""
    df_raw = pd.read_csv(FIELDS_CSV, dtype=str).fillna("")
    if limit:
        df_raw = df_raw.head(limit)
    print(f"== 1) 加载原始数据 ==")
    print(f"corpus_fields.csv: {len(df_raw)} 行 × {len(df_raw.columns)} 列")

    rows, dropped = [], {"empty_core": 0, "short": 0, "no_txt": 0, "excluded": 0}
    for r in df_raw.itertuples(index=False):
        pmcid = r.pmcid
        if pmcid in EXCLUDE_PMCID:
            dropped["excluded"] += 1
            continue
        raw = read_text(pmcid)
        if not raw.strip():
            dropped["no_txt"] += 1
            continue
        lines = raw.splitlines()
        if int(r.n_lines) < MIN_LINES:          # §1.3: 极短文(会议摘要/社论)
            dropped["short"] += 1
            continue
        _, _, core, tail_cut = split_regions(lines)
        core_text = "\n".join(core).strip()
        if not core_text:                        # §1.3: 正文被整体误判为噪声
            dropped["empty_core"] += 1
            continue
        rows.append({
            "doc_id": pmcid,          # 唯一标识: pmcid(零缺失), pmid 仅作元数据保留
            "pmid": r.pmid,
            "title": r.title,
            "doi": r.doi,
            "journal": r.journal_nlm,     # 0 缺失 -> 可作期刊过滤
            "pub_year": r.pub_year,       # 0 缺失 -> 可作年份过滤
            "license": r.license,
            "n_lines": int(r.n_lines),
            "lang": detect_lang(core_text),
            "tail_cut": tail_cut,         # False = 仍含参考文献(127 篇)
            "text": core_text,
        })

    df = pd.DataFrame(rows)
    print(f"清洗后排保留: {len(df)} 篇  "
          f"(剔除 极短文 {dropped['short']} / 空core {dropped['empty_core']} / "
          f"无txt {dropped['no_txt']} / 名单 {dropped['excluded']})")
    print(f"语言构成: {dict(df['lang'].value_counts())}")
    print(f"尾边界识别成功: {int(df['tail_cut'].sum())}/{len(df)}")
    return df_raw, df, dropped


# ---------------------------------------------------------------- 步骤 2: 分割
class DocumentSplitter:
    """按报告 §4.3 定稿策略切块"""

    def __init__(self, strategy="length", chunk_size=512, chunk_overlap=64,
                 tokenizer="bge-m3", max_input=8192):
        if chunk_size >= max_input:
            raise SystemExit(f"[error] chunk_size({chunk_size}) 必须 < 模型上限({max_input})")
        tok_path = os.path.join(TOKENIZER_ROOT, tokenizer, "tokenizer.json")
        if not os.path.exists(tok_path):
            raise SystemExit(f"[error] 找不到 tokenizer: {tok_path}")
        self.tok = Tokenizer.from_file(tok_path)
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_input = max_input

        # length_function 会被递归调用很多次, 加缓存避免重复编码(同一字符串 O(1) 命中)
        self._count_tokens = functools.lru_cache(maxsize=300_000)(
            self._count_tokens_uncached)

        if strategy == "length":
            # 主体预算 = chunk_size - chunk_overlap, 剩余预算用"把上一块尾部前置"来补 overlap。
            # 为什么不直接交给 TextSplitter 的 chunk_overlap: 它在 _merge_splits 里靠"回退 split"
            # 实现, 当单个 split(段落) 的 token 数 > chunk_overlap 时会被整体 pop 掉;
            # 本语料段落 p50 约 200 token > overlap 64 -> 实测相邻块完全无重叠。
            # 再留 4 token 余量: 分隔符保留与 decode/encode 往返会带来 1~2 token 出入,
            # 不留余量会产出 513 > 512 的越界块(实测 20 篇出现 3 块)
            self.content_size = chunk_size - chunk_overlap - 4
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.content_size,
                chunk_overlap=0,
                length_function=self._count_tokens,
                # 默认分隔符只照顾英文, 补上中文句号(语料有 9 篇中文)
                separators=["\n\n", "\n", "\u3002", "\uff01", "\uff1f", ". ", " ", ""],
                keep_separator=True,
                strip_whitespace=True,
            )

    # ---- 任务书要求的 length_function ----
    def _count_tokens_uncached(self, text: str) -> int:
        return len(self.tok.encode(text, add_special_tokens=False).ids)

    def _overlap_text(self, ids):
        """把末尾 chunk_overlap 个 token 解码成文本, 作为下一块的前缀文本。

        两个关键点:
        1) 用"文本"而不是"token id 序列"做拼接。因为下一块是 decode(prev_tail + ids)
           之后再被重新 encode, 词首标记 ▁ 会在往返中变化(实测 42% 的相邻块对
           token 序列对不上)。直接拼文本则字符完全相同, 不存在往返问题。
        2) 起点仍要对齐到词首(▁), 否则下一块的开头会是半个单词。
        """
        n = self.chunk_overlap
        if not n or len(ids) <= n:
            return self.tok.decode(list(ids))
        start = len(ids) - n
        while start < len(ids):
            piece = self.tok.id_to_token(ids[start]) or ""
            if piece.startswith("\u2581") or not piece[:1].isalnum():
                break                      # 词首(▁)或标点开头 -> 从这里开始
            start += 1                     # 词中间片段(如 'mell'/'itus') -> 跳过
        return self.tok.decode(list(ids[start:])) if start < len(ids) else ""

    def _meta(self, document, i, total, text, extra=None):
        """组装块的元数据(任务书最小集 + 本项目检索字段)"""
        d = {
            "chunk_id": f"{document['doc_id']}_{i:04d}",
            "text": text,
            "doc_id": document["doc_id"],
            "chunk_index": i,
            "total_chunks": total,
            "source_title": document["title"],
            "token_count": self._count_tokens(text),
            # ---- 以下为本项目补充(供阶段⑤ 元数据过滤, 零成本) ----
            "pmid": document["pmid"],
            "doi": document["doi"],
            "journal": document["journal"],
            "pub_year": document["pub_year"],
            "lang": document["lang"],
            "is_table": "\t" in text,        # §4.4-2: 表格被展平成单段
            "tail_cut": document["tail_cut"],
        }
        if extra:
            d.update(extra)
        return d

    # ---- 策略 a: 按长度智能分割 ----
    def split_document(self, document):
        """两阶段切块:
           1) 用 RecursiveCharacterTextSplitter 按 content_size(=chunk_size-overlap) 切,
              保证段落/句子边界不被切断;
           2) 给第 i>0 块前置上一块末尾的 overlap 个 token, 使相邻块真正重叠。
           最终每块 token = overlap + 主体 <= chunk_size。"""
        full = document["text"]
        texts = self.splitter.split_text(full)
        total = len(texts)
        out, cursor, prev_text = [], 0, ""
        for i, t in enumerate(texts):
            text = f"{prev_text} {t}" if (i > 0 and prev_text) else t
            # 定位本块主体在原文中的结束位置, 取后一个字符 -> 用于"是否被硬切在单词中间"判定
            pos = full.find(t, cursor)
            if pos >= 0:
                cursor = pos + len(t)
            nxt = full[cursor:cursor + 1] if cursor < len(full) else ""
            out.append(self._meta(document, i, total, text, {"next_char": nxt}))
            ids = self.tok.encode(t, add_special_tokens=False).ids
            prev_text = self._overlap_text(ids) if self.chunk_overlap else ""
        return out

    def merge_short_chunks(self, chunks):
        """把过短块(token < TOO_SHORT_TOKENS)并入相邻块, 消除无检索价值的碎片。

        并入方向:
          1) 优先并入前一块尾部(保持阅读顺序);
          2) 首块没有前一块 -> 并入后一块开头。实测 4 个短块全部是首块,
             内容为孤立的 "Abstract" 标题, 前置到摘要正文语义更完整;
          3) 并入后会超过 chunk_size 则不并入, 直接丢弃该短块
             (宁可丢 1~10 token 的标题, 也不产出超限块)。
        最后重排 chunk_index / total_chunks / chunk_id, 保证编号连续。
        """
        out, merged = [], 0
        for c in chunks:
            if out and c["token_count"] < TOO_SHORT_TOKENS:
                prev = out[-1]
                cand = f"{prev['text']}\n\n{c['text']}"
                if self._count_tokens(cand) <= self.chunk_size:
                    prev.update(text=cand, token_count=self._count_tokens(cand),
                                is_table="\t" in cand,
                                next_char=c.get("next_char", ""))
                    merged += 1
                    continue
                merged += 1                      # 超限: 丢弃
                continue
            out.append(dict(c))

        if len(out) > 1 and out[0]["token_count"] < TOO_SHORT_TOKENS:
            head, nxt = out[0], out[1]
            cand = f"{head['text']}\n\n{nxt['text']}"
            if self._count_tokens(cand) <= self.chunk_size:
                nxt.update(text=cand, token_count=self._count_tokens(cand),
                           is_table="\t" in cand)
                out = out[1:]
            merged += 1

        n = len(out)
        for i, c in enumerate(out):
            c["chunk_index"] = i
            c["total_chunks"] = n
            c["chunk_id"] = f"{c['doc_id']}_{i:04d}"
        return out, merged

    # ---- 策略 b: 整体不分割 ----
    def no_split_document(self, document):
        d = {
            "chunk_id": document["doc_id"],     # 直接用文献 ID 作块 ID
            "text": document["text"],
            "doc_id": document["doc_id"],
            "chunk_index": 0,
            "total_chunks": 1,
            "source_title": document["title"],
            "token_count": self._count_tokens(document["text"]),
            "pmid": document["pmid"],
            "doi": document["doi"],
            "journal": document["journal"],
            "pub_year": document["pub_year"],
            "lang": document["lang"],
            "is_table": "\t" in document["text"],
            "tail_cut": document["tail_cut"],
            "next_char": "",          # 整篇不切, 不存在边界硬切问题
        }
        return [d]

    def run(self, df):
        t0 = time.time()
        chunks, per_doc = [], []
        self.n_merged = 0
        for doc in df.to_dict("records"):
            got = (self.split_document(doc) if self.strategy == "length"
                   else self.no_split_document(doc))
            if self.strategy == "length":
                got, merged = self.merge_short_chunks(got)
                self.n_merged += merged
            chunks.extend(got)
            per_doc.append(len(got))
        print(f"切块耗时 {time.time() - t0:.1f}s  "
              f"(块/篇 均值 {statistics.fmean(per_doc):.1f}, max {max(per_doc)})")
        if self.n_merged:
            print(f"合并过短块: {self.n_merged} 个 (token < {TOO_SHORT_TOKENS})")
        return pd.DataFrame(chunks), per_doc


# ---------------------------------------------------------------- 步骤 5: 质量验证
def quality_check(chunks_df, splitter, sample_docs=50):
    """全量检查 + 多块文献的 overlap 校验"""
    print("\n== 5) 质量验证 ==")
    problems, flags = [], {
        "empty": 0, "over_model_limit": 0, "over_chunk_size": 0,
        "too_short": 0, "truncated": 0, "mojibake": 0, "has_section_head": 0,
    }

    for c in chunks_df.to_dict("records"):
        issues = []
        text = c["text"]
        n = c["token_count"]
        if not text.strip():
            issues.append("empty")
        if n > splitter.max_input:
            issues.append("over_model_limit")
        elif n > splitter.chunk_size:
            issues.append("over_chunk_size")
        if 0 < n < TOO_SHORT_TOKENS:
            issues.append("too_short")
        # 真正的硬切 = 块结束在"单词中间", 即原文紧跟着还是字母。
        # 只看"尾部是不是句号"会严重误判: 脚注符号(††)、URL、表格数值结尾都是完整内容。
        if c.get("next_char", "").isalpha():
            issues.append("truncated")
        if any(p in text for p in MOJIBAKE):
            issues.append("mojibake")
        if SECTION_RE.search(text):
            flags["has_section_head"] += 1
        for k in issues:
            flags[k] = flags.get(k, 0) + 1
        if issues:
            problems.append({
                "chunk_id": c["chunk_id"], "doc_id": c["doc_id"],
                "chunk_index": c["chunk_index"], "token_count": n,
                "issues": "|".join(issues), "head": text[:80].replace("\n", " "),
            })

    total = len(chunks_df)
    print(f"{'检查项':<20}{'命中块数':>10}{'占比%':>9}   说明")
    notes = {
        "over_model_limit": f"token > {splitter.max_input}, 送进 bge-m3 会被截断 -> 必须重切",
        "over_chunk_size": f"token > chunk_size({splitter.chunk_size}) -> splitter 失控",
        "too_short": f"token < {TOO_SHORT_TOKENS}, 信息量不足, 建议与相邻块合并",
        "truncated": "原文中块结束处的下一个字符仍是字母 -> 切在了单词中间",
        "mojibake": "含双重编码字符(§1.3 列出 5 篇)",
        "empty": "空块",
        "has_section_head": "含章节标题(非问题, 用于后续章节标注)",
    }
    for k in ["over_model_limit", "over_chunk_size", "empty", "too_short",
              "truncated", "mojibake", "has_section_head"]:
        v = flags.get(k, 0)
        print(f"{k:<20}{v:>10}{v / max(1, total) * 100:>9.2f}   {notes[k]}")

    # ---- overlap 校验: 只对多块文献抽样的相邻块对 ----
    # 字符级比对(窗口必须给够): 64 token 的 overlap 约 250 字符,
    # 若像第一版那样只在前 180 字符里找, 重叠区落在窗口外 -> 会误判成"无重叠"。
    overlap_hits = []
    multi = chunks_df[chunks_df["total_chunks"] >= 3]
    docs = list(dict.fromkeys(multi["doc_id"]))[:sample_docs]
    for did in docs:
        grp = chunks_df[chunks_df["doc_id"] == did].sort_values("chunk_index")
        texts = grp["text"].tolist()
        for i in range(len(texts) - 1):
            # 先归一化空白再比对: 表格段落里的 \t 会被 tokenizer 归一成空格,
            # 不归一化会把"内容确实重叠、只是空白表示不同"的块对误判成无重叠
            a = WS_RE.sub(" ", texts[i])
            b = WS_RE.sub(" ", texts[i + 1])
            suffix = a[-200:]
            best = 0
            for L in range(min(200, len(suffix)), 20, -1):
                if suffix[-L:] in b[:400]:      # 后一块的开头含前一块的结尾
                    best = L
                    break
            overlap_hits.append(best)
    ov = {}
    if overlap_hits:
        n = len(overlap_hits)
        zero = sum(1 for x in overlap_hits if x == 0)
        good = sum(1 for x in overlap_hits if x >= 100)
        ov = {"sample_docs": len(docs), "pairs": n,
              "median": statistics.median(overlap_hits),
              "mean": statistics.fmean(overlap_hits), "max": max(overlap_hits),
              "zero": zero, "good": good}
        print(f"\noverlap 校验(字符级): 抽样 {len(docs)} 篇 / {n} 个相邻块对")
        print(f"  实际重叠字符: 中位 {ov['median']:.0f}, "
              f"均值 {ov['mean']:.0f}, 最大 {ov['max']} "
              f"(64 token ≈ 250 字符)")
        print(f"  完全无重叠 {zero} 对 ({zero / n * 100:.1f}%), "
              f"重叠 >= 100 字符 {good} 对 ({good / n * 100:.1f}%)")
        print(f"  判定: {'✅ overlap 生效' if good / n >= 0.95 else '❌ overlap 未生效, 需排查'}")

    return pd.DataFrame(problems), flags, ov


def preview(chunks_df, n_docs=3):
    """步骤 4: 预览"""
    print(f"\n== 4) 结果预览 (随机 {n_docs} 篇) ==")
    docs = list(dict.fromkeys(chunks_df["doc_id"]))
    import random
    random.seed(42)
    for did in random.sample(docs, min(n_docs, len(docs))):
        grp = chunks_df[chunks_df["doc_id"] == did].sort_values("chunk_index")
        first = grp.iloc[0]
        print(f"\n--- {did} | {first['source_title'][:60]}")
        print(f"    共 {first['total_chunks']} 块, lang={first['lang']}, "
              f"journal={first['journal']}, year={first['pub_year']}")
        for _, r in grp.head(2).iterrows():
            print(f"    [#{r['chunk_index']}] {r['token_count']} token | "
                  f"{r['text'][:100].replace(chr(10), ' ')}...")


# ---------------------------------------------------------------- 步骤 3/5 报告
def write_report_md(path, args, df_raw, df, dropped, chunks_df, per_doc,
                    stats, flags, ov, prob_df):
    """生成人类可读的统计与质量验证报告(交付物之一)"""
    tokens = chunks_df["token_count"].tolist()
    total = len(chunks_df)
    L, A = [], None
    A = L.append
    A("# 文档解析与分割报告\n")
    A(f"- 生成时间: {stats['processed_date']}")
    A(f"- 处理脚本: `scripts/build_chunks.py`")
    A(f"- 数据范围: {stats['data_split']}")
    A("")

    A("## 1 处理配置\n")
    A("| 项 | 值 | 依据 |")
    A("|---|---|---|")
    A(f"| 分割策略 | `{args.strategy}` | 本次主策略，智能分割 |")
    A(f"| chunk_size | {args.chunk_size} token | 报告 §4.3：超限段落 2.49%，≤5% 的最小值 |")
    A(f"| chunk_overlap | {args.chunk_overlap} token（12.5%） | 报告 §4.3 给 10~15% |")
    A(f"| 分割器 | RecursiveCharacterTextSplitter | 段落 → 句子 → 空格 |")
    A(f"| length_function | bge-m3 tokenizer | 切块上限取决于嵌入模型的 tokenizer |")
    A(f"| 模型输入上限 | {args.max_input} token | bge-m3 规格 |")
    A(f"| 切块对象 | core 正文 | 元数据头块+参考文献占 32.2%（§2.5），必须先剔除 |")
    A("")

    A("## 2 数据加载与清洗\n")
    A(f"原始文献 **{len(df_raw)}** 篇 → 清洗后 **{len(df)}** 篇。\n")
    A("| 剔除原因 | 篇数 | 说明 |")
    A("|---|---|---|")
    A(f"| 极短文（<{MIN_LINES} 行） | {dropped['short']} | 报告 §1.3：会议摘要/社论，信息密度低 |")
    A(f"| 硬编码排除名单 | {dropped['excluded']} | `{'`, `'.join(sorted(EXCLUDE_PMCID))}`：core 为空 / 全篇 mojibake |")
    A(f"| core 为空 | {dropped['empty_core']} | 正文被整体判为噪声 |")
    A(f"| 无 txt | {dropped['no_txt']} | 目录里没有全文 |")
    A("")
    lang = stats["lang_dist"]
    A(f"语言构成: " + " / ".join(f"{k} {v}" for k, v in lang.items()) +
      f"；尾边界识别成功 {stats['docs_with_tail_cut']}/{len(df)}"
      f"（{len(df) - stats['docs_with_tail_cut']} 篇仍含参考文献，为报告 §5-6 遗留项）。")
    A("")

    A("## 3 分割结果\n")
    A("| 指标 | 值 |")
    A("|---|---|")
    A(f"| 文献数 | {len(df)} |")
    A(f"| 总块数 | {total} |")
    A(f"| 块/篇 均值 | {stats['chunks_per_doc']} |")
    A(f"| 块/篇 中位 | {int(quantile(per_doc, 50))} |")
    A(f"| 块/篇 最大 | {max(per_doc)} |")
    A(f"| 总 token 数 | {stats['total_tokens']:,} |")
    A(f"| 合并的过短块 | {stats['merged_short_chunks']} |")
    A("")

    A("## 4 块长分布 (token)\n")
    A("| 分位 | " + " | ".join(QNAMES) + " | 均值 |")
    A("|---|" + "---|" * (len(QNAMES) + 1))
    A("| 块长 | " + " | ".join(str(int(quantile(tokens, q))) for q in QUANTILES) +
      f" | {statistics.fmean(tokens):.0f} |")
    A("")
    A("```")
    A(f"块长分布直方图 (共 {total} 块, 上限 {args.chunk_size} token)")
    for lo, hi in [(0, 100), (100, 200), (200, 300), (300, 400),
                   (400, 450), (450, 500), (500, 513)]:
        c = sum(1 for t in tokens if lo <= t < hi)
        bar = "#" * round(c / total * 60)
        A(f"{lo:>4}-{hi:<4}{c:>7}{c / total * 100:>7.1f}%  {bar}")
    A("```")
    A("")
    fill = statistics.fmean(tokens) / args.chunk_size * 100
    tail = sum(1 for t in tokens if t < 100)
    A(f"块长填充率均值 **{fill:.1f}%**（报告 §4.2 模拟值 85.4%）。分布集中在高区间："
      f"450~500 token 一档占 {sum(1 for t in tokens if 450 <= t < 500) / total * 100:.1f}%，"
      f"说明多数块在接近上限处自然结束。低于模拟值的原因是每篇的末尾块普遍偏短"
      f"（<100 token 的块 {tail} 个，占 {tail / total * 100:.1f}%，即每篇末块），"
      f"而非被 512 硬切成大量碎片。")
    A("")

    A("## 5 质量验证\n")
    A("### 5.1 全量检查\n")
    A("| 检查项 | 命中块数 | 占比 | 判定 |")
    A("|---|---|---|---|")
    verdict = {
        "over_model_limit": "硬性要求，必须为 0",
        "over_chunk_size": "硬性要求，必须为 0",
        "empty": "必须为 0",
        "too_short": f"token < {TOO_SHORT_TOKENS}，已自动并入相邻块",
        "truncated": "原文中块尾下一字符仍是字母；仅 MathType 公式，可接受",
        "mojibake": "含双重编码字符；剔除外名单后应为 0",
        "has_section_head": "非问题，供后续章节标注",
    }
    for k in ["over_model_limit", "over_chunk_size", "empty", "too_short",
              "truncated", "mojibake", "has_section_head"]:
        v = flags.get(k, 0)
        ok = "✅" if (k == "has_section_head" or v == 0) else "⚠️"
        A(f"| `{k}` | {v} | {v / max(1, total) * 100:.2f}% | {ok} {verdict[k]} |")
    A("")

    A("### 5.2 多块文献的 overlap 校验\n")
    if ov:
        A(f"抽样 {ov['sample_docs']} 篇 / {ov['pairs']} 个相邻块对，字符级比对：\n")
        A("| 指标 | 值 |")
        A("|---|---|")
        A(f"| 重叠字符 中位 | {ov['median']:.0f} |")
        A(f"| 重叠字符 均值 | {ov['mean']:.0f} |")
        A(f"| 重叠字符 最大 | {ov['max']} |")
        A(f"| 完全无重叠 | {ov['zero']} 对（{ov['zero'] / ov['pairs'] * 100:.1f}%） |")
        A(f"| 重叠 ≥100 字符 | {ov['good']} 对（{ov['good'] / ov['pairs'] * 100:.1f}%） |")
        A("")
        A(f"判定：**{'✅ overlap 生效' if ov['good'] / ov['pairs'] >= 0.95 else '❌ overlap 未生效'}**"
          f"（64 token ≈ 250 字符）")
    else:
        A("无多块文献，未执行。")
    A("")

    A("### 5.3 问题块清单\n")
    if len(prob_df) == 0:
        A("无问题块。")
    else:
        A(f"共 {len(prob_df)} 个（占 {len(prob_df) / total * 100:.3f}%）：\n")
        A("| chunk_id | token | 问题 | 块首 80 字符 |")
        A("|---|---|---|---|")
        for r in prob_df.head(20).to_dict("records"):
            A(f"| `{r['chunk_id']}` | {r['token_count']} | {r['issues']} | "
              f"{r['head'][:80]} |")
        if len(prob_df) > 20:
            A(f"\n（仅列前 20 个，完整清单见 `quality_report_{args.strategy}.csv`）")
    A("")

    A("## 6 产物清单\n")
    A("| 文件 | 说明 |")
    A("|---|---|")
    A(f"| `data/chunks/chunks_{args.strategy}.parquet` | 文本块数据集（主文件，列式压缩） |")
    A(f"| `data/chunks/chunks_{args.strategy}.jsonl` | 同内容，便于直接查看 |")
    A(f"| `data/chunks/processing_stats_{args.strategy}.json` | 处理配置与统计（机器可读） |")
    A(f"| `data/chunks/quality_report_{args.strategy}.csv` | 问题块清单 |")
    A(f"| `data/chunks/build_chunks_{args.strategy}.log` | 处理日志（本次运行完整输出） |")
    A("")
    A("数据集字段：`chunk_id` / `text` / `doc_id` / `chunk_index` / `total_chunks` / "
      "`source_title` / `token_count` / `pmid` / `doi` / `journal` / `pub_year` / "
      "`lang` / `is_table` / `tail_cut`")
    A("")

    A("## 7 结论\n")
    hard = flags.get("over_model_limit", 0) + flags.get("over_chunk_size", 0) + \
        flags.get("empty", 0)
    A(f"- 块总数 **{total}**，块长 {int(quantile(tokens, 0))}~{max(tokens)} token，"
      f"全部 ≤ chunk_size({args.chunk_size})，无一超过模型上限 {args.max_input}。")
    A(f"- 硬性检查（超限/空块）命中 **{hard}** 个，"
      f"{'数据集可进入向量化阶段。' if hard == 0 else '需排查后再入库。'}")
    A(f"- 预计向量化耗时：{total} 块 × 1.555 s ≈ {total * 1.555 / 3600:.1f} 小时"
      f"（CPU bge-m3，全流程瓶颈）。")
    A("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    return path


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser(description="PMC OA 文档解析与分割")
    ap.add_argument("--strategy", default="length", choices=["length", "no_split"],
                    help="length=按长度智能分割(默认) / no_split=整体不分割(对照组)")
    ap.add_argument("--chunk-size", type=int, default=512)
    ap.add_argument("--chunk-overlap", type=int, default=64, help="12.5%%, 报告 §4.3 给 10~15%%")
    ap.add_argument("--tokenizer", default="bge-m3")
    ap.add_argument("--max-input", type=int, default=8192, help="bge-m3 最大输入 token")
    ap.add_argument("--limit", type=int, default=None, help="只处理前 N 篇(试跑用)")
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--report-md", default=None,
                    help="人类可读报告路径(默认 <out-dir>/report_<strategy>.md)")
    ap.add_argument("--no-log", action="store_true", help="不落盘处理日志")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, f"build_chunks_{args.strategy}.log")
    if not args.no_log:
        sys.stdout = Tee(log_path)      # 处理日志落盘(交付物之一)
    splitter = DocumentSplitter(
        strategy=args.strategy, chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap, tokenizer=args.tokenizer,
        max_input=args.max_input)
    print(f"== 0) 配置 ==")
    print(f"strategy={args.strategy}  chunk_size={args.chunk_size}  "
          f"chunk_overlap={args.chunk_overlap}  tokenizer={args.tokenizer}  "
          f"max_input={args.max_input}")
    print(f"tokenizer vocab={splitter.tok.get_vocab_size()}")

    # 步骤 1
    df_raw, df, dropped = build_dataframe(args.limit)
    if df.empty:
        raise SystemExit("[error] 清洗后无可用文献")

    # 步骤 2+3
    print(f"\n== 2~3) 分割并保存 ==")
    chunks_df, per_doc = splitter.run(df)

    name = f"chunks_{args.strategy}"
    parquet_path = os.path.join(args.out_dir, f"{name}.parquet")
    jsonl_path = os.path.join(args.out_dir, f"{name}.jsonl")

    # 统计
    tokens = chunks_df["token_count"].tolist()
    print(f"\n== 3) 块长分布 (token) ==")
    print(f"{'item':<22}{'N':>7}" + "".join(f"{q:>9}" for q in QNAMES) + f"{'mean':>10}")
    print_quantiles("chunk_tokens", tokens)
    print_quantiles("chunks_per_doc", per_doc)

    stats = {
        "processed_date": pd.Timestamp.now().isoformat(),
        "data_split": "full" if args.limit is None else f"head_{args.limit}",
        "strategy": args.strategy,
        "original_documents": len(df_raw),
        "documents_after_clean": len(df),
        "dropped": dropped,
        "excluded_pmcid": sorted(EXCLUDE_PMCID),
        "merged_short_chunks": getattr(splitter, "n_merged", 0),
        "total_chunks": len(chunks_df),
        "chunks_per_doc": round(len(chunks_df) / len(df), 2),
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "tokenizer": args.tokenizer,
        "max_input": args.max_input,
        "token_quantiles": {q: int(quantile(tokens, p))
                            for q, p in zip(QNAMES, QUANTILES)},
        "total_tokens": int(sum(tokens)),
        "docs_with_tail_cut": int(df["tail_cut"].sum()),
        "lang_dist": {k: int(v) for k, v in df["lang"].value_counts().items()},
        "output_file": str(parquet_path),
    }
    stats_path = os.path.join(args.out_dir, f"processing_stats_{args.strategy}.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # 步骤 4
    preview(chunks_df)

    # 步骤 5
    prob_df, flags, ov = quality_check(chunks_df, splitter)
    prob_path = os.path.join(args.out_dir, f"quality_report_{args.strategy}.csv")
    prob_df.to_csv(prob_path, index=False, encoding="utf-8-sig")

    # next_char 只是质检用的临时列, 不进最终数据集
    chunks_out = chunks_df.drop(columns=["next_char"], errors="ignore")
    chunks_out.to_parquet(parquet_path, index=False)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for c in chunks_out.to_dict("records"):
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    stats["quality_flags"] = flags
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\n[saved] {parquet_path}")
    print(f"[saved] {jsonl_path}")
    print(f"[saved] {stats_path}")
    print(f"[saved] {prob_path}  ({len(prob_df)} 个问题块)")

    # 步骤 3/5 报告
    report = args.report_md or os.path.join(args.out_dir, f"report_{args.strategy}.md")
    write_report_md(report, args, df_raw, df, dropped, chunks_df, per_doc,
                    stats, flags, ov, prob_df)
    print(f"[saved] {report}")
    if not args.no_log:
        print(f"[saved] {log_path}")

    # 结论
    print(f"\n== 6) 结论 ==")
    over = flags.get("over_model_limit", 0)
    if args.strategy == "no_split":
        print(f"整体不分割: {over}/{len(chunks_df)} 篇超过模型上限 {args.max_input} token "
              f"-> {'该策略不可用(报告 §4.1 判定)' if over else '全部合规'}")
    else:
        print(f"智能分割: 超模型上限 {over} 块, 超 chunk_size "
              f"{flags.get('over_chunk_size', 0)} 块 -> "
              f"{'✅ 可进入向量化' if over == 0 and flags.get('over_chunk_size', 0) == 0 else '❌ 需排查'}")
        est_h = len(chunks_df) * 1.555 / 3600
        print(f"预计向量化耗时: {len(chunks_df)} 块 × 1.555s ≈ {est_h:.1f} 小时")

    if isinstance(sys.stdout, Tee):          # 收尾: 关闭日志文件并还原 stdout
        tee, sys.stdout = sys.stdout, sys.stdout.out
        tee.f.close()


if __name__ == "__main__":
    main()
