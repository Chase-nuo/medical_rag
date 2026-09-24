# -*- coding: utf-8 -*-
"""
向量化与索引构建 (阶段④)
=========================
输入: data/chunks/chunks_length.parquet (27 349 个文本块, 阶段③产出)
输出: data/chroma/ (ChromaDB 持久化索引) + data/chroma/index_stats.json

分两步, 各自可独立重跑:
  1) 嵌入: 调用本机 Ollama 的 bge-m3 生成向量, 落盘成 npy
     - 支持断点续跑: 每批写完立即 flush, 崩溃后从 progress 记录的下标继续
     - 这是全流程最耗时的一步(CPU 约 1.7 块/s, 全量 ≈ 4.5 h), 必须可续
  2) 索引: 读 npy 灌入 ChromaDB(余弦相似度), 几分钟就能重建

为什么要分开:
  嵌入一旦跑完就是资产, 换索引参数(collection 名/元数据/距离度量)时
  不应重新花 4.5 小时算向量, 只需 --skip-embed 重建索引。

用法:
    python scripts/embed_chunks.py                     # 全量(嵌入+建索引)
    python scripts/embed_chunks.py --limit 500         # 小样本试跑
    python scripts/embed_chunks.py --skip-embed        # 用已有 npy 重建索引
    python scripts/embed_chunks.py --skip-index        # 只算向量
    python scripts/embed_chunks.py --query "..."       # 直接查询(不重建)

    python scripts/embed_chunks.py --query "SGLT2 inhibitor heart failure" \
        --n 5 --where '{"lang": "en"}'                 # 带元数据过滤
"""
import argparse
import json
import os
import statistics as st
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHUNKS_PARQUET = os.path.join(ROOT, "data", "chunks", "chunks_length.parquet")
EMB_PATH = os.path.join(ROOT, "data", "chunks", "embeddings_{model}.npy")
PROGRESS_PATH = os.path.join(ROOT, "data", "chunks", "embeddings_{model}.progress.json")
CHROMA_DIR = os.path.join(ROOT, "data", "chroma")
STATS_PATH = os.path.join(ROOT, "data", "chroma", "index_stats.json")

# ---------------------------------------------------------------- 模型配置
# 选型说明(排除链 + 实测依据见 docs/向量化与索引构建报告.md 第 2 节):
#   bge-m3 = BAAI 多语言嵌入模型, 1024 维, 支持 8192 token 输入,
#   与阶段③切块用的 tokenizer 同源; 本机无 GPU, Ollama 的 GGUF 实现走 CPU。
#   语料 997 en + 9 zh, 纯英文模型(en-v1.5 系列)对中文块会退化, 故不用。
MODEL = "bge-m3"
DIM = 1024

# bge v1.5 系列官方建议给查询加指令前缀; bge-m3 官方说明查询无需前缀。
# 两种都实现, 用开关控制, 实测对比见 verify_index.py §3 与报告第 4 节。
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

BATCH = 64           # 实测吞吐饱和点: batch 32→1.65 块/s, 64→1.70, 256→1.85
METADATA_FIELDS = ["doc_id", "chunk_index", "total_chunks", "source_title",
                   "token_count", "pmid", "doi", "journal", "pub_year",
                   "lang", "is_table", "tail_cut"]


class Tee:
    """把 stdout 同时写进 UTF-8 日志文件。

    不用 shell 重定向: Windows PowerShell 的 > 走系统 ANSI 编码(GBK),
    日志里的中文会变乱码, 而日志本身是交付物之一, 所以脚本自己写文件。
    """

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


# ---------------------------------------------------------------- 嵌入
class OllamaEmbedder:
    """Ollama /api/embed 的薄封装, 带重试(长跑 4.5h, 偶发超时必须自愈)"""

    def __init__(self, model=MODEL, instruction=False):
        import ollama
        self.client = ollama.Client(host="http://localhost:11434", timeout=120)
        self.model = model
        self.instruction = instruction

    def _embed(self, texts, retries=3):
        last = None
        for i in range(retries):
            try:
                r = self.client.embed(model=self.model, input=texts)
                return np.asarray(r["embeddings"], dtype=np.float32)
            except Exception as e:                      # noqa: BLE001
                last = e
                print(f"    [retry {i + 1}/{retries}] {type(e).__name__}: {e}",
                      flush=True)
                time.sleep(3 * (i + 1))
        raise RuntimeError(f"embed 失败: {last}")

    def embed_documents(self, texts):
        return self._embed(texts)

    def embed_query(self, text):
        if self.instruction:
            text = QUERY_INSTRUCTION + text
        return self._embed([text])[0]


def run_embed(df, emb_path, progress_path, batch=BATCH, model=MODEL):
    """生成全部块的向量, 落盘 npy。已跑过的部分自动跳过。"""
    n = len(df)
    texts = df["text"].tolist()
    ids = df["chunk_id"].tolist()

    done_from = 0
    if os.path.exists(progress_path) and os.path.exists(emb_path):
        prog = json.load(open(progress_path, encoding="utf-8"))
        if prog.get("model") == model and prog.get("n") == n:
            done_from = int(prog.get("next_index", 0))
            if done_from >= n:
                print(f"嵌入已完成({n} 块), 跳过。删 {os.path.basename(progress_path)} 可重跑")
                return np.load(emb_path, mmap_mode="r")
            print(f"断点续跑: 已完成 {done_from}/{n}, 从 {done_from} 继续")
        else:
            print("progress 与当前数据/模型不匹配, 从头开始")

    # memmap 预分配, 边算边写, 内存占用与批量无关
    emb = np.lib.format.open_memmap(emb_path, mode="w+" if done_from == 0 else "r+",
                                    dtype=np.float32, shape=(n, DIM))
    if done_from == 0:
        emb[:] = 0

    enc = OllamaEmbedder(model)
    t0 = time.time()
    i = done_from
    while i < n:
        chunk = texts[i:i + batch]
        t = time.perf_counter()
        emb[i:i + len(chunk)] = enc.embed_documents(chunk)
        emb.flush()
        i += len(chunk)
        json.dump({"model": model, "dim": DIM, "n": n, "batch": batch,
                   "next_index": i, "updated": time.strftime("%Y-%m-%dT%H:%M:%S")},
                  open(progress_path, "w", encoding="utf-8"))
        if i % (batch * 5) == 0 or i >= n:
            el = time.time() - t0
            done = i - done_from
            rate = done / el
            eta = (n - i) / rate if rate > 0 else 0
            print(f"  {i}/{n}  {rate:.2f} 块/s  "
                  f"已用 {el / 60:.1f}min  ETA {eta / 60:.1f}min", flush=True)
            if i >= n:
                break
        # 单批耗时打印(前几批用于观察)
        if i <= batch * 3:
            print(f"    批耗时 {time.perf_counter() - t:.1f}s ({len(chunk)} 块)", flush=True)

    print(f"嵌入完成: {n} 块, 用时 {(time.time() - t0) / 60:.1f} min", flush=True)
    # 抽查: 全零行说明该批没写进去
    full = np.load(emb_path, mmap_mode="r")
    zero = int(np.sum(np.linalg.norm(full[:], axis=1) == 0))
    print(f"零向量行数: {zero} {'(异常!)' if zero else ''}", flush=True)
    return full


# ---------------------------------------------------------------- 索引
def clean_meta(row):
    """ChromaDB 的 metadata 不接受 None/NaN, 统一转成空串; bool 保持原样"""
    out = {}
    for k in METADATA_FIELDS:
        v = row.get(k, "")
        if v is None or (isinstance(v, float) and np.isnan(v)):
            out[k] = ""
        elif isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = float(v)
        elif isinstance(v, (np.bool_, bool)):
            out[k] = bool(v)
        else:
            out[k] = str(v)
    return out


def build_index(df, emb, collection_name="medical_chunks", reset=False):
    """把向量+元数据灌入 ChromaDB(余弦相似度)"""
    import chromadb

    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    if reset:
        try:
            client.delete_collection(collection_name)
            print(f"已删除旧 collection: {collection_name}")
        except Exception:                                # noqa: BLE001
            pass
    col = client.get_or_create_collection(
        collection_name, metadata={"hnsw:space": "cosine"})

    existing = col.count()
    if existing >= len(df):
        print(f"collection 已有 {existing} 条, 删除现有内容后重建")
        ids_all = col.get(limit=existing)["ids"]
        col.delete(ids=ids_all)

    add_batch = 512
    t0 = time.time()
    for i in range(0, len(df), add_batch):
        sl = df.iloc[i:i + add_batch]
        col.add(
            ids=sl["chunk_id"].astype(str).tolist(),
            embeddings=np.asarray(emb[i:i + add_batch], dtype=np.float32).tolist(),
            documents=sl["text"].tolist(),
            metadatas=[clean_meta(r) for r in sl.to_dict("records")],
        )
        if (i + add_batch) % 2048 == 0 or i + add_batch >= len(df):
            print(f"  已入库 {min(i + add_batch, len(df))}/{len(df)}", flush=True)
    print(f"入库完成: {col.count()} 条, 用时 {(time.time() - t0) / 60:.1f} min")
    return col


def save_stats(col, df, model, dim, collection_name):
    stats = {
        "collection_name": collection_name,
        "total_chunks": int(col.count()),
        "embedding_model": model,
        "embedding_dimension": int(dim),
        "index_built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "chunk_size_stats": ({
            "mean": round(float(st.fmean(df["token_count"])), 1),
            "max": int(df["token_count"].max()),
            "min": int(df["token_count"].min()),
        } if "token_count" in df.columns else {}),
        "metadata_fields": METADATA_FIELDS,
        "documents": int(df["doc_id"].nunique()),
        "hnsw_space": "cosine",
        "persist_dir": CHROMA_DIR,
    }
    os.makedirs(os.path.dirname(STATS_PATH), exist_ok=True)
    json.dump(stats, open(STATS_PATH, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    return stats


# ---------------------------------------------------------------- 查询
class MedicalIndex:
    """检索封装: 复用已建好的 collection"""

    def __init__(self, collection_name="medical_chunks", model=MODEL,
                 instruction=False):
        import chromadb
        self.client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.col = self.client.get_collection(collection_name)
        self.enc = OllamaEmbedder(model, instruction=instruction)

    def query(self, query_text, embedding_model=MODEL, n_results=5,
              where_filter=None, instruction=None):
        """语义检索。

        Args:
            query_text: 查询文本
            embedding_model: 嵌入模型(必须与建库时一致, 否则向量空间不同)
            n_results: 返回条数
            where_filter: Chroma 元数据过滤, 如 {"lang": "en"} / {"pub_year": "2023"}
        Returns:
            dict: ids / documents / metadatas / distances / similarities
        """
        use_ins = self.enc.instruction if instruction is None else instruction
        vec = OllamaEmbedder(embedding_model, instruction=use_ins).embed_query(query_text)
        kw = {"query_embeddings": [vec.tolist()], "n_results": n_results}
        if where_filter:
            kw["where"] = where_filter
        r = self.col.query(**kw)
        return {
            "ids": r["ids"][0],
            "documents": r["documents"][0],
            "metadatas": r["metadatas"][0],
            "distances": [round(d, 4) for d in r["distances"][0]],
            "similarities": [round(1 - d, 4) for d in r["distances"][0]],
        }


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="向量化与索引构建")
    ap.add_argument("--limit", type=int, default=None, help="只处理前 N 块(试跑)")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--collection", default="medical_chunks")
    ap.add_argument("--skip-embed", action="store_true")
    ap.add_argument("--skip-index", action="store_true")
    ap.add_argument("--reset", action="store_true", help="先删旧 collection")
    ap.add_argument("--query", default=None, help="直接查询, 不重建索引")
    ap.add_argument("--n", type=int, default=5, help="查询返回条数")
    ap.add_argument("--where", default=None, help='元数据过滤 JSON, 如 {"lang":"en"}')
    ap.add_argument("--instruction", action="store_true", help="查询时加 BGE 指令前缀")
    ap.add_argument("--no-log", action="store_true", help="不落盘日志")
    args = ap.parse_args()

    if not args.no_log and not args.query:
        sys.stdout = Tee(os.path.join(ROOT, "data", "chunks",
                                      f"embed_chunks_{args.model.replace(':', '_')}.log"))

    emb_path = EMB_PATH.format(model=args.model.replace(":", "_"))
    progress_path = PROGRESS_PATH.format(model=args.model.replace(":", "_"))

    # 纯查询模式
    if args.query:
        idx = MedicalIndex(args.collection, args.model, args.instruction)
        where = json.loads(args.where) if args.where else None
        r = idx.query(args.query, args.model, args.n, where)
        print(f"查询: {args.query[:80]}  filter={where}  model={args.model}"
              f"{'  +指令' if args.instruction else ''}")
        for i, (cid, md, sim, doc) in enumerate(
                zip(r["ids"], r["metadatas"], r["similarities"], r["documents"])):
            print(f"\n[{i + 1}] {cid}  sim={sim:.4f}  "
                  f"{md.get('source_title', '')[:60]} ({md.get('pub_year', '')}, "
                  f"{md.get('journal', '')[:40]})")
            print(f"    {doc[:200]}")
        return

    print("== 配置 ==")
    print(f"model={args.model}  batch={args.batch}  collection={args.collection}")
    df = pd.read_parquet(CHUNKS_PARQUET)
    if args.limit:
        df = df.head(args.limit).reset_index(drop=True)
    print(f"待处理: {len(df)} 块 / {df['doc_id'].nunique()} 篇")

    if not args.skip_embed:
        print("\n== 1) 嵌入 ==")
        emb = run_embed(df, emb_path, progress_path, args.batch, args.model)
    else:
        print("\n== 1) 嵌入(跳过, 读已有 npy) ==")
        emb = np.load(emb_path, mmap_mode="r")
        assert len(emb) >= len(df), f"npy 只有 {len(emb)} 行, 少于 {len(df)} 块"

    if not args.skip_index:
        print("\n== 2) 建索引 ==")
        col = build_index(df, emb[:len(df)], args.collection, args.reset)
        stats = save_stats(col, df, args.model, emb.shape[1], args.collection)
        print("\n== 3) 索引统计 ==")
        for k, v in stats.items():
            print(f"  {k:<20} {v}")
        print(f"\n[saved] {STATS_PATH}")


if __name__ == "__main__":
    main()
