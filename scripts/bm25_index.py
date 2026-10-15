"""BM25 关键词检索索引（检索系统开发第二部分）。

为什么需要 BM25 这一路：
    向量检索按"意思"找，死穴是罕见词与精确编号（如 `eGFR < 30`、`SGLT2`）；
    BM25 按"字面"找，正好补上这个洞。两路互补，这是多路召回的前提。

实现取舍：
    - 不装 rank_bm25：它把语料放内存且不支持持久化，27 349 块每次启动重建不划算。
      自己实现倒排索引，一次构建、存盘、秒级加载。
    - postings 用 CSR 扁平数组（int32）而非 Python list of tuple：
      约 7M 条 posting，list 方案 ~500 MB，CSR 方案 ~60 MB。
    - 中文无 jieba，用「单字 + bigram」：语料 99.4% 是英文（27 196 / 153），
      中文只需能匹配上即可，不值得为一个分词器引入新依赖。

用法:
    python scripts/bm25_index.py --build          # 构建并存盘
    python scripts/bm25_index.py --query "metformin cardiovascular mortality"
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

CHUNKS_PARQUET = os.path.join(ROOT, "data", "chunks", "chunks_length.parquet")
BM25_DIR = os.path.join(ROOT, "data", "bm25")
INDEX_PATH = os.path.join(BM25_DIR, "bm25_index.pkl")
META_PATH = os.path.join(BM25_DIR, "bm25_meta.json")

# 经典参数: k1 控制词频饱和, b 控制长度归一强度
K1 = 1.5
B = 0.75

_WORD = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")
_CJK = re.compile(r"[\u4e00-\u9fff]+")

# 停用词: 从查询理解模块复用, 保证"建索引"与"查索引"用同一套过滤规则,
# 否则查询里的 the/of 会匹配不上（建时被过滤, 查时没过滤）。
try:
    from query_understanding import STOPWORDS
except Exception:                                  # noqa: BLE001
    STOPWORDS = set("""a an the and or but of to in on for with by from as at be is
    are was were this that these those it its we our you your they their not no""".split())


def tokenize(text: str) -> list[str]:
    """英文按词、中文按单字+bigram。返回小写 token 列表。"""
    text = (text or "").lower()
    toks: list[str] = []
    for m in _WORD.finditer(text):
        t = m.group(0).strip("'-")
        if len(t) >= 2 and t not in STOPWORDS:
            toks.append(t)
    for m in _CJK.finditer(text):
        seg = m.group(0)
        toks.extend(seg)                            # 单字
        if len(seg) >= 2:
            toks.extend(seg[i:i + 2] for i in range(len(seg) - 1))   # bigram
    return toks


class BM25Index:
    """倒排索引 + BM25 打分。postings 以 CSR 扁平数组存储。"""

    def __init__(self, k1: float = K1, b: float = B):
        self.k1 = k1
        self.b = b
        self.chunk_ids: list[str] = []
        self.vocab: dict[str, int] = {}
        self.df: np.ndarray | None = None
        self.idf: np.ndarray | None = None
        self.doc_len: np.ndarray | None = None
        self.avgdl = 0.0
        self.offsets: np.ndarray | None = None      # len(vocab)+1
        self.post_docs: np.ndarray | None = None    # int32
        self.post_tfs: np.ndarray | None = None     # int32

    # ------------------------------------------------------------------ 构建
    def build(self, texts: list[str], chunk_ids: list[str], log=print) -> "BM25Index":
        t0 = time.perf_counter()
        n = len(texts)
        self.chunk_ids = list(chunk_ids)
        self.doc_len = np.zeros(n, dtype=np.int32)

        # term -> [doc, tf, doc, tf, ...]  构建期用 Python list 累积
        raw: dict[str, list[int]] = {}
        for i, txt in enumerate(texts):
            cnt: dict[str, int] = {}
            for t in tokenize(txt):
                cnt[t] = cnt.get(t, 0) + 1
            self.doc_len[i] = sum(cnt.values())
            for t, c in cnt.items():
                lst = raw.get(t)
                if lst is None:
                    raw[t] = [i, c]
                else:
                    lst.append(i)
                    lst.append(c)
            if (i + 1) % 5000 == 0:
                log(f"  分词 {i + 1}/{n}  {time.perf_counter() - t0:.1f}s", flush=True)

        # 转 CSR: 词表按字典序固定, 保证存盘/加载一致
        terms = sorted(raw)
        self.vocab = {t: k for k, t in enumerate(terms)}
        nterms = len(terms)
        self.df = np.zeros(nterms, dtype=np.int32)
        self.offsets = np.zeros(nterms + 1, dtype=np.int64)
        docs_parts, tfs_parts = [], []
        pos = 0
        for k, t in enumerate(terms):
            lst = raw[t]
            m = len(lst) // 2
            self.df[k] = m
            self.offsets[k] = pos
            docs_parts.append(np.asarray(lst[0::2], dtype=np.int32))
            tfs_parts.append(np.asarray(lst[1::2], dtype=np.int32))
            pos += m
        self.offsets[nterms] = pos
        self.post_docs = np.concatenate(docs_parts) if docs_parts else np.zeros(0, np.int32)
        self.post_tfs = np.concatenate(tfs_parts) if tfs_parts else np.zeros(0, np.int32)

        # BM25 的 idf (Robertson 版本, 保证非负)
        self.avgdl = float(self.doc_len.mean()) if n else 0.0
        self.idf = np.log(1.0 + (n - self.df + 0.5) / (self.df + 0.5))
        log(f"  词表 {nterms}  posting {pos}  avgdl {self.avgdl:.0f}  "
            f"{time.perf_counter() - t0:.1f}s", flush=True)
        return self

    # ------------------------------------------------------------------ 查询
    def query(self, text: str, top_k: int = 50) -> tuple[list[int], list[float]]:
        """返回 (doc 下标列表, BM25 分数列表)，按分数降序，长度 <= top_k。"""
        if self.idf is None:
            raise RuntimeError("索引未构建/加载")
        toks = tokenize(text)
        if not toks:
            return [], []
        n = len(self.chunk_ids)
        score = np.zeros(n, dtype=np.float32)
        hit = 0
        for t in toks:
            k = self.vocab.get(t)
            if k is None:
                continue
            hit += 1
            s, e = self.offsets[k], self.offsets[k + 1]
            if s == e:
                continue
            d = self.post_docs[s:e]
            tf = self.post_tfs[s:e].astype(np.float32)
            dl = self.doc_len[d].astype(np.float32)
            denom = tf + self.k1 * (1.0 - self.b + self.b * dl / self.avgdl)
            contrib = self.idf[k] * tf * (self.k1 + 1.0) / denom
            score += np.bincount(d, weights=contrib, minlength=n).astype(np.float32)
        if hit == 0 or score.max() <= 0:
            return [], []
        k = min(top_k, n)
        # 全量 argsort 在 27k 上也就几毫秒, 不必上 argpartition 增加复杂度
        idx = np.argsort(-score)[:k]
        return [int(i) for i in idx], [float(score[i]) for i in idx]

    # ------------------------------------------------------------------ 存盘
    def save(self, path: str = INDEX_PATH) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "chunk_ids": self.chunk_ids, "vocab": self.vocab, "df": self.df,
                "idf": self.idf, "doc_len": self.doc_len, "avgdl": self.avgdl,
                "offsets": self.offsets, "post_docs": self.post_docs,
                "post_tfs": self.post_tfs, "k1": self.k1, "b": self.b,
            }, f, protocol=5)
        size = os.path.getsize(path) / 1048576
        json.dump({
            "n_docs": len(self.chunk_ids),
            "n_terms": len(self.vocab),
            "n_postings": int(len(self.post_docs)),
            "avgdl": round(self.avgdl, 2),
            "k1": self.k1, "b": self.b,
            "index_size_mb": round(size, 1),
        }, open(META_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str = INDEX_PATH) -> "BM25Index":
        with open(path, "rb") as f:
            d = pickle.load(f)
        obj = cls(d.get("k1", K1), d.get("b", B))
        for key in ("chunk_ids", "vocab", "df", "idf", "doc_len", "avgdl",
                    "offsets", "post_docs", "post_tfs"):
            setattr(obj, key, d[key])
        return obj


def main() -> None:
    ap = argparse.ArgumentParser(description="BM25 索引")
    ap.add_argument("--build", action="store_true", help="构建并存盘")
    ap.add_argument("--query", default=None, help="测试查询")
    ap.add_argument("--top-k", type=int, default=10)
    args = ap.parse_args()

    if args.build:
        df = pd.read_parquet(CHUNKS_PARQUET)
        texts = df["text"].tolist()
        ids = df["chunk_id"].tolist()
        print(f"构建 BM25 索引: {len(texts)} 块")
        ix = BM25Index().build(texts, ids)
        ix.save()
        print("已保存 ->", INDEX_PATH)
        return

    ix = BM25Index.load()
    if args.query:
        ids, sc = ix.query(args.query, args.top_k)
        print(f"查询: {args.query}")
        print(f"命中 {len(ids)} 条 (token: {tokenize(args.query)[:12]})")
        for i, s in zip(ids, sc):
            print(f"  {s:7.3f}  {ix.chunk_ids[i]}")
        return
    print("索引已加载:", json.load(open(META_PATH, encoding="utf-8")))


if __name__ == "__main__":
    main()
