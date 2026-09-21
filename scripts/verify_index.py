# -*- coding: utf-8 -*-
"""
向量索引质量验证 (阶段④)
=========================
对应任务书第 3 点, 五项检查:

  1) 基础统计验证: 向量数量 == 文本块数、维度、样本元数据完整性
  2) 相似性检索验证: 从索引摘文本当查询, 测自相似性
     - full: 整块文本作查询(上界参考, 几乎必中)
     - first_sentence: 只取首句作查询(更接近真实提问, 有区分度)
  3) 边界情况验证: 空查询 / 超长查询 / 单字符 / 不可能满足的过滤条件
  4) 元数据过滤验证: 过滤结果数与数据集真实值一致, 且返回条条满足过滤
  5) BGE 查询指令实验: 加/不加指令前缀的检索效果对比(决定线上默认开关)

用法:
    python scripts/verify_index.py                  # 全量验证
    python scripts/verify_index.py --n 10           # 抽样少一些(快)
    python scripts/verify_index.py --skip-instruction
    python scripts/verify_index.py --json data/chroma/verify_report.json
"""
import argparse
import json
import os
import random
import re
import statistics as st
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHUNKS_PARQUET = os.path.join(ROOT, "data", "chunks", "chunks_length.parquet")
CHROMA_DIR = os.path.join(ROOT, "data", "chroma")
STATS_PATH = os.path.join(ROOT, "data", "chroma", "index_stats.json")
DEFAULT_JSON = os.path.join(ROOT, "data", "chroma", "verify_report.json")

COLLECTION = "medical_chunks"
MODEL = "bge-m3"
SENT = re.compile(r"(?<=[.!?])\s+")


def hr(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


class Verifier:
    def __init__(self, collection=COLLECTION, model=MODEL):
        import chromadb
        from embed_chunks import OllamaEmbedder
        self.client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.col = self.client.get_collection(collection)
        self.enc = OllamaEmbedder(model)
        self.enc_ins = OllamaEmbedder(model, instruction=True)
        self.chunks = pd.read_parquet(CHUNKS_PARQUET)
        # 索引内的 id 与元数据(只取 metadata, 不取 embeddings, 内存可控)。
        # 自相似抽样必须限定在索引内的块: 否则抽到未入库的块, rank 恒为 0,
        # 会误判成"检索失效"(小库试跑时踩过)。
        got = self.col.get(limit=self.col.count(), include=["metadatas"])
        self.index_meta = pd.DataFrame(got["metadatas"])
        self.index_meta["chunk_id"] = got["ids"]
        self.in_index = self.chunks[self.chunks["chunk_id"].isin(set(got["ids"]))]
        self.report = {}

    # ---------------- 1) 基础统计 ----------------
    def v1_basic(self):
        hr("1) 基础统计验证")
        n_idx = self.col.count()
        n_chunks = len(self.chunks)
        ok = n_idx == n_chunks
        print(f"索引向量数: {n_idx}")
        print(f"文本块数  : {n_chunks}")
        print(f"判定: {'✅ 一致' if ok else '❌ 不一致, 索引缺失或重复'}")

        # 维度+零向量抽查
        got = self.col.get(limit=5, include=["embeddings", "metadatas", "documents"])
        dim = len(got["embeddings"][0])
        norms = [float(np.linalg.norm(v)) for v in got["embeddings"]]
        zero = sum(1 for x in norms if x == 0)
        print(f"\n向量维度: {dim}   零向量: {zero}/5   "
              f"模长范围 {min(norms):.2f}~{max(norms):.2f}")

        # 样本元数据完整性
        missing = []
        for md in got["metadatas"]:
            for k in ["doc_id", "chunk_index", "source_title", "token_count",
                      "journal", "pub_year", "lang"]:
                if md.get(k) in (None, ""):
                    missing.append((md.get("doc_id"), k))
        print(f"元数据必填字段缺失: {len(missing)} 处 {missing[:5]}")
        print("\n样本块:")
        for i, md in enumerate(got["metadatas"][:3]):
            print(f"  [{i}] {got['ids'][i]}  {md.get('source_title', '')[:55]} "
                  f"({md.get('pub_year', '')}, {md.get('journal', '')[:30]}, "
                  f"{md.get('token_count', '')} token)")

        stats = json.load(open(STATS_PATH, encoding="utf-8")) \
            if os.path.exists(STATS_PATH) else {}
        print(f"\nindex_stats.json: total_chunks={stats.get('total_chunks')} "
              f"model={stats.get('embedding_model')} "
              f"dim={stats.get('embedding_dimension')} "
              f"space={stats.get('hnsw_space')}")
        self.report["basic"] = {
            "index_count": n_idx, "chunk_count": n_chunks, "match": bool(ok),
            "dim": dim, "zero_vectors_in_sample": zero,
            "missing_meta_fields": len(missing),
            "stats_file_total": stats.get("total_chunks"),
        }

    # ---------------- 2) 自相似检索 ----------------
    def _self_test(self, samples, mode, enc, n_results=10, tag=""):
        ranks, sims = [], []
        for _, r in samples.iterrows():
            text = r["text"]
            if mode == "first_sentence":
                text = SENT.split(text.strip())[0][:300]
            elif mode == "full":
                text = text[:2000]          # 超过也没意义, 省时间
            vec = enc.embed_query(text)
            res = self.col.query(query_embeddings=[vec.tolist()],
                                 n_results=n_results)
            ids = res["ids"][0]
            rank = ids.index(r["chunk_id"]) + 1 if r["chunk_id"] in ids else 0
            ranks.append(rank)
            if rank == 1:
                sims.append(1 - res["distances"][0][0])
        hit1 = sum(1 for x in ranks if x == 1) / len(ranks)
        hit5 = sum(1 for x in ranks if 0 < x <= 5) / len(ranks)
        mrr = st.fmean([1 / x if x else 0 for x in ranks])
        print(f"  {mode:<15}{tag:<10} hit@1={hit1 * 100:5.1f}%  "
              f"hit@5={hit5 * 100:5.1f}%  MRR={mrr:.3f}  "
              f"命中时 sim 均值 {st.fmean(sims) if sims else 0:.3f}")
        return {"hit@1": round(hit1, 4), "hit@5": round(hit5, 4),
                "mrr": round(mrr, 4), "n": len(ranks)}

    def v2_self_similarity(self, n=20, do_ins=True):
        hr("2) 相似性检索验证 (自相似性)")
        rng = random.Random(42)
        n = min(n, len(self.in_index))
        samples = self.in_index.iloc[rng.sample(range(len(self.in_index)), n)]
        print(f"随机抽 {n} 个块(限定在索引内), 用其文本作查询, "
              f"看原块能否被检索回来\n")
        out = {}
        for mode in ("full", "first_sentence"):
            out[mode] = self._self_test(samples, mode, self.enc)
            if do_ins and mode == "first_sentence":
                out["first_sentence+指令"] = self._self_test(
                    samples, mode, self.enc_ins, tag="+ins")
        self.report["self_similarity"] = out

    # ---------------- 3) 边界情况 ----------------
    def v3_edge_cases(self):
        hr("3) 边界情况验证")
        cases = {
            "空查询": "",
            "空白字符": "   \n\t  ",
            "单字符": "a",
            "纯标点": "?!。？",
            "超长查询": ("heart failure treatment and outcomes in patients "
                        "with reduced ejection fraction. " * 800)[:40000],
            "乱码": "\u00e4\u00f6\u00fc\u00df" * 50,
        }
        out = {}
        for name, q in cases.items():
            t = time.perf_counter()
            try:
                vec = self.enc.embed_query(q)
                res = self.col.query(query_embeddings=[vec.tolist()], n_results=3)
                dim = len(vec)
                norm = float(np.linalg.norm(vec))
                top = res["ids"][0][0] if res["ids"][0] else None
                sim = round(1 - res["distances"][0][0], 4) if res["distances"][0] else None
                print(f"  {name:<10} OK   维度={dim} 模长={norm:.2f} "
                      f"top1={top} sim={sim} ({time.perf_counter() - t:.1f}s)")
                out[name] = {"status": "ok", "dim": dim, "norm": round(norm, 3),
                             "top1": top, "sim": sim}
            except Exception as e:                       # noqa: BLE001
                print(f"  {name:<10} 异常 {type(e).__name__}: "
                      f"{str(e)[:90]}")
                out[name] = {"status": "error",
                             "error": f"{type(e).__name__}: {str(e)[:200]}"}
        # 不可能满足的过滤条件
        try:
            res = self.col.query(query_embeddings=[self.enc.embed_query("diabetes")],
                                 n_results=3, where={"lang": "xx"})
            n = len(res["ids"][0])
            print(f"  不可能满足的过滤 (lang=xx): 返回 {n} 条 "
                  f"{'✅ 正确为空' if n == 0 else '❌ 应为空'}")
            out["impossible_filter"] = {"returned": n, "ok": n == 0}
        except Exception as e:                           # noqa: BLE001
            print(f"  不可能满足的过滤: 异常 {e}")
            out["impossible_filter"] = {"error": str(e)}
        self.report["edge_cases"] = out

    # ---------------- 4) 元数据过滤 ----------------
    def v4_metadata_filter(self):
        hr("4) 元数据过滤验证")
        out = {}
        checks = [("lang", "zh"), ("lang", "en"), ("pub_year", "2023")]
        n_idx = self.col.count()
        full = n_idx == len(self.chunks)     # 索引是否覆盖全量
        for field, val in checks:
            # 过滤命中数以索引自身为准: 索引可能是全量的子集(试跑时),
            # 这时拿全量 parquet 当期望值会误判
            expect = int((self.index_meta[field].astype(str) == val).sum())
            got = self.col.get(where={field: val}, limit=max(n_idx, 1))
            n = len(got["ids"])
            extra = ""
            if full:     # 全量时再做一次与 parquet 的交叉校验
                ds = int((self.chunks[field].astype(str) == val).sum())
                extra = f" / 数据集 {ds} 条"
                if ds != expect:
                    extra += " ❌索引与数据集不一致"
            print(f"  where {field}={val}: 过滤命中 {n} 条, 索引内实际 {expect} 条"
                  f"{extra}  {'✅' if n == expect else '❌ 不一致'}")
            out[f"{field}={val}"] = {"returned": n, "in_index": expect,
                                     "match": n == expect}

        # 带过滤的语义检索: 返回结果的元数据必须条条满足
        vec = self.enc.embed_query("diabetes treatment outcomes")
        res = self.col.query(query_embeddings=[vec.tolist()], n_results=10,
                             where={"lang": "en"})
        langs = {m.get("lang") for m in res["metadatas"][0]}
        print(f"\n  语义检索 + 过滤 lang=en: 返回 {len(res['ids'][0])} 条, "
              f"lang 取值集合 {langs}  {'✅' if langs == {'en'} else '❌ 混入其他'}")
        out["query_with_filter"] = {"returned": len(res["ids"][0]),
                                    "langs": sorted(langs),
                                    "ok": langs == {"en"}}

        # 过滤确实生效(与不过滤结果对比)
        res2 = self.col.query(query_embeddings=[vec.tolist()], n_results=10)
        diff = len(set(res2["ids"][0]) - set(res["ids"][0]))
        print(f"  同一查询不加过滤: top10 中有 {diff} 条不同 -> "
              f"{'✅ 过滤确实改变了结果集' if diff else '⚠️ 结果相同(可能语料本身单一)'}")
        out["filter_changes_result"] = {"diff": diff}
        self.report["metadata_filter"] = out

    # ---------------- 5) 指令实验 ----------------
    def v5_instruction(self, n=20):
        hr("5) BGE 查询指令实验")
        print("bge v1.5 系列官方建议查询加指令前缀; bge-m3 官方说明查询无需前缀。")
        print("这里用同一批查询实测, 决定线上默认是否开启。\n")
        rng = random.Random(7)
        n = min(n, len(self.in_index))
        samples = self.in_index.iloc[rng.sample(range(len(self.in_index)), n)]
        r_no = self._self_test(samples, "first_sentence", self.enc, tag="无指令")
        r_yes = self._self_test(samples, "first_sentence", self.enc_ins, tag="有指令")
        verdict = ("建议 不加指令" if r_no["hit@1"] >= r_yes["hit@1"]
                   else "建议 加指令")
        print(f"\n结论: {verdict} "
              f"(hit@1 {r_no['hit@1']:.3f} vs {r_yes['hit@1']:.3f}, "
              f"MRR {r_no['mrr']:.3f} vs {r_yes['mrr']:.3f})")
        self.report["instruction_exp"] = {"no_ins": r_no, "with_ins": r_yes,
                                          "verdict": verdict}

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.report["verified_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        json.dump(self.report, open(path, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print(f"\n[saved] {path}")


def main():
    ap = argparse.ArgumentParser(description="向量索引质量验证")
    ap.add_argument("--collection", default=COLLECTION)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--n", type=int, default=20, help="自相似/指令实验抽样块数")
    ap.add_argument("--skip-instruction", action="store_true")
    ap.add_argument("--json", default=DEFAULT_JSON)
    args = ap.parse_args()

    v = Verifier(args.collection, args.model)
    v.v1_basic()
    v.v2_self_similarity(args.n, do_ins=not args.skip_instruction)
    v.v3_edge_cases()
    v.v4_metadata_filter()
    if not args.skip_instruction:
        v.v5_instruction(args.n)
    v.save(args.json)

    # 总结
    hr("验证结论")
    b = v.report.get("basic", {})
    ss = v.report.get("self_similarity", {})
    mf = v.report.get("metadata_filter", {})
    print(f"向量数量一致      : {'✅' if b.get('match') else '❌'} "
          f"{b.get('index_count')} / {b.get('chunk_count')}")
    print(f"维度正确          : {'✅' if b.get('dim') == 1024 else '❌'} {b.get('dim')}")
    print(f"自相似 hit@1(首句): {ss.get('first_sentence', {}).get('hit@1')}")
    print(f"元数据过滤        : "
          f"{'✅' if all(v2.get('match', v2.get('ok', True))
                       for k, v2 in mf.items() if isinstance(v2, dict)) else '❌'}")
    errs = [k for k, x in v.report.get("edge_cases", {}).items()
            if isinstance(x, dict) and x.get("status") == "error"]
    print(f"边界情况异常      : {errs if errs else '无 ✅'}")


if __name__ == "__main__":
    main()
