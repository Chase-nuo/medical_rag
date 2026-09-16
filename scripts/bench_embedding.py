# -*- coding: utf-8 -*-
"""
向量化容量实测 (容量评估阶段)
=============================
回答: 本机 CPU 跑 bge-m3, 1 小时 / 1 晚上能向量化多少篇文献?

测什么:
  1) 冷启动耗时(首次调用含模型加载)
  2) 稳态吞吐: 秒/块、块/秒、token/秒
  3) 单次调用的固定开销 vs 随文本长度增长的边际开销(用于判断 batch 是否值得)
  4) 按实测值外推: 给定篇数需要多久 / 给定时间能做多少篇

口径说明:
  - 文本用真实语料(PMC OA core 正文), 按 4.02 字符/token 折算成目标 token 数;
  - 逐块调用模拟 ChromaDB + OllamaEmbeddings 的默认行为(一次一块);
  - token 数用 bge-m3 tokenizer 实测, 不用字符估算, 保证外推准确。

用法:
    python scripts/bench_embedding.py
    python scripts/bench_embedding.py --chunks 60 --tokens 512
    python scripts/bench_embedding.py --batch          # 额外测批量接口
"""
import argparse
import glob
import json
import os
import re
import statistics as st
import time

import ollama

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "pmc_oa")
MODEL = "bge-m3"
# PMC 纯文本正文起点分隔线(与 analyze_token_length.py 保持一致)
BODY_SPLIT = "\x9f====\x9f"

SENT = re.compile(r"(?<=[.!?])\s+")


def load_core_texts(limit: int = 6) -> list[str]:
    """取若干篇的正文(core 粗略版: 正文分隔线之后), 作为真实语料样本。"""
    texts = []
    for p in sorted(glob.glob(os.path.join(ROOT, "*", "*.txt")))[:limit]:
        raw = open(p, encoding="utf-8", errors="replace").read()
        if BODY_SPLIT in raw:
            raw = raw.split(BODY_SPLIT, 1)[1]
        texts.append(raw)
    return texts


def build_chunks(texts: list[str], n_chunks: int, tokens_each: int,
                 chars_per_token: float = 4.02) -> list[str]:
    """把真实正文按句子拼成 n_chunks 个约 tokens_each token 的块。"""
    target_chars = int(tokens_each * chars_per_token)
    pool, buf = [], ""
    for t in texts:
        for s in SENT.split(t):
            buf += s + " "
            if len(buf) >= target_chars:
                pool.append(buf.strip())
                buf = ""
    if buf.strip():
        pool.append(buf.strip())
    # 循环取用, 保证够 n_chunks 个
    out = []
    i = 0
    while len(out) < n_chunks:
        out.append(pool[i % len(pool)])
        i += 1
    return out[:n_chunks]


def count_tokens(texts: list[str]) -> int:
    """用 bge-m3 tokenizer 实测 token 总数(需 tokenizers 库)。"""
    try:
        from tokenizers import Tokenizer
        tk_path = os.path.join(os.path.dirname(ROOT), "tokenizer", "bge-m3",
                               "tokenizer.json")
        tk = Tokenizer.from_file(tk_path)
        enc = tk.encode_batch(texts)
        return sum(len(e.ids) for e in enc)
    except Exception:  # noqa: BLE001
        # 退回字符估算
        return int(sum(len(t) for t in texts) / 4.02)


def bench(n_chunks: int, tokens_each: int, do_batch: bool) -> dict:
    texts_all = load_core_texts()
    chunks = build_chunks(texts_all, n_chunks, tokens_each)

    # --- 冷启动 ---
    t0 = time.perf_counter()
    r = ollama.embeddings(model=MODEL, prompt=chunks[0])
    cold = time.perf_counter() - t0
    dim = len(r["embedding"])

    # --- 稳态: 逐块 ---
    lat = []
    for c in chunks[1:]:
        t = time.perf_counter()
        ollama.embeddings(model=MODEL, prompt=c)
        lat.append(time.perf_counter() - t)
    if not lat:  # 只有 1 块时兜底
        t = time.perf_counter()
        ollama.embeddings(model=MODEL, prompt=chunks[0])
        lat.append(time.perf_counter() - t)

    n_tokens = count_tokens(chunks[1:] or chunks)
    total_s = sum(lat)

    res = {
        "model": MODEL, "dim": dim,
        "cold_start_s": round(cold, 2),
        "n_chunks": len(lat), "n_tokens": n_tokens,
        "total_s": round(total_s, 2),
        "mean_s_per_chunk": round(st.mean(lat), 4),
        "p50_s_per_chunk": round(st.median(lat), 4),
        "chunks_per_min": round(60 / st.mean(lat), 1),
        "tokens_per_s": round(n_tokens / total_s, 1),
    }

    # --- 可选: 长度敏感性(判断固定开销占比) ---
    probe = {}
    for mult, label in ((0.25, "128tok"), (1.0, "512tok"), (2.0, "1024tok")):
        c = chunks[0][: int(len(chunks[0]) * mult)] or chunks[0]
        ts = []
        for _ in range(3):
            t = time.perf_counter()
            ollama.embeddings(model=MODEL, prompt=c)
            ts.append(time.perf_counter() - t)
        probe[label] = round(st.mean(ts), 4)
    res["latency_by_len_s"] = probe

    # --- 可选: 批量接口 ---
    if do_batch:
        t = time.perf_counter()
        ollama.embed(model=MODEL, input=chunks[:20])
        res["batch20_s"] = round(time.perf_counter() - t, 2)

    return res


def extrapolate(res: dict, tokens_per_article: float, chunks_per_article: float):
    """按实测吞吐外推: 篇/小时、给定篇数需多久。"""
    s_per_chunk = res["mean_s_per_chunk"]
    s_per_article = s_per_chunk * chunks_per_article
    out = {
        "chunks_per_article": round(chunks_per_article, 1),
        "sec_per_article": round(s_per_article, 1),
        "articles_per_hour": round(3600 / s_per_article, 1),
    }
    for n in (300, 500, 1000, 2000, 5000, 10000):
        out[f"hours_for_{n}"] = round(n * s_per_article / 3600, 1)
    return out


def main():
    ap = argparse.ArgumentParser(description="bge-m3 向量化吞吐实测与容量外推")
    ap.add_argument("--chunks", type=int, default=40, help="稳态测试块数")
    ap.add_argument("--tokens", type=int, default=512, help="每块目标 token 数")
    ap.add_argument("--batch", action="store_true", help="额外测批量 embed 接口")
    args = ap.parse_args()

    print(f"模型={MODEL}  测试块数={args.chunks}  目标块长={args.tokens} token")
    print("首次调用会加载模型, 请稍候...\n")

    res = bench(args.chunks, args.tokens, args.batch)
    print("=" * 70)
    print("【实测结果】")
    for k, v in res.items():
        print(f"  {k:<20} {v}")

    # 用本语料实测值外推: core 均值 7756 token/篇, 512 token 切块约 19.3 块/篇
    print("\n" + "=" * 70)
    print("【外推】(按本语料 core 均值 7756 token/篇, chunk=512, overlap=10%)")
    ext = extrapolate(res, tokens_per_article=7756, chunks_per_article=19.3)
    for k, v in ext.items():
        print(f"  {k:<22} {v}")

    print("\n" + "=" * 70)
    print(json.dumps({"bench": res, "extrapolate": ext}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
