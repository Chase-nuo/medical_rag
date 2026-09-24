# -*- coding: utf-8 -*-
"""索引质量验证 (阶段④ 第 3 步)
=================================================
建库不等于能用。这一脚本回答四个问题:

  1 数量对不对   : collection.count() 是否等于文本块数
                   (少一条说明入库批次漏了, 多一条说明重复 add)
  2 绑没绑对     : 自相似性 —— 拿库中原文当查询, top1 必须命中它自己。
                   这是唯一能一次验证「id / 向量 / 文本 / 元数据四者对齐」的测试:
                   只要有一处错位(比如 emb 行序与 df 行序不一致), top1 就会变成别的块。
  3 语义有没有用 : 人工可读的医学问题, 看 top-K 是否切题
  4 过滤灵不灵   : where 过滤后, 检查返回条目的字段值是否真的满足条件
                   (只"不报错"不算通过, 必须验证值)
  5 会不会崩     : 空串 / 超长 / 乱码 等边界输入

用法:
    python scripts/verify_index.py                    # 全量验证
    python scripts/verify_index.py --samples 20       # 加大自相似抽样
    python scripts/verify_index.py --log              # 同时写日志文件
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import embed_chunks as E  # noqa: E402

# 人工设计的探针查询: 覆盖「药物-结局」「指南推荐」「中文」三类,
# 便于肉眼判断语义检索是否真的在工作(而不是靠相似度数字自欺)
PROBE_QUERIES = [
    "Does metformin reduce cardiovascular mortality in type 2 diabetes?",
    "What is the recommended first-line treatment for heart failure with "
    "reduced ejection fraction?",
    "抗生素治疗社区获得性肺炎的疗程",
]

FILTERS = [{"lang": "en"}, {"lang": "zh"}, {"is_table": True}, {"pub_year": "2023"}]

EDGE_CASES = [
    ("空查询", ""),
    ("纯空格", "   "),
    ("超长查询(约6000字符)", "cardiac failure treatment " * 250),
    ("无意义串", "asdfghjkl"),
]


def check_basic(col, expect=None):
    print("=" * 72)
    print("1) 基础统计")
    print("=" * 72)
    print(f"  collection      {col.name}   metadata={col.metadata}")
    cnt = col.count()
    print(f"  向量总数        {cnt}")
    if expect is not None:
        ok = cnt == expect
        print(f"  期望            {expect}   -> {'PASS' if ok else 'FAIL'}")
    peek = col.peek(2)
    for i, (cid, md) in enumerate(zip(peek["ids"], peek["metadatas"])):
        print(f"  样本{i}  id={cid}")
        print(f"          {json.dumps(md, ensure_ascii=False)[:160]}")
    return cnt


def check_semantic(ix, n=3):
    print()
    print("=" * 72)
    print("2) 语义检索 (无指令前缀)")
    print("=" * 72)
    for q in PROBE_QUERIES:
        t = time.perf_counter()
        r = ix.query(q, n_results=n)
        print(f"\n  查询: {q[:70]}")
        print(f"  耗时: {time.perf_counter() - t:.2f}s")
        for i in range(n):
            md = r["metadatas"][i]
            print(f"    [{i}] sim={r['similarities'][i]:.4f} {r['ids'][i]}")
            print(f"        {md.get('source_title', '')[:64]} "
                  f"({md.get('pub_year', '')}, {md.get('lang', '')})")


def check_instruction(ix, n=3):
    """BGE 指令前缀对比实验。

    任务书说「BGE 查询加指令更好」, 但该结论来自 bge v1.5 系列。
    bge-m3 训练时已内化检索指令, 官方明确查询端无需前缀。
    这一节用实测决定本项目到底开不开 --instruction, 不靠文档背书。
    """
    print()
    print("=" * 72)
    print("3) 指令前缀对比 (bge-m3 是否需要查询前缀)")
    print("=" * 72)
    q = PROBE_QUERIES[0]
    a = ix.query(q, n_results=n, instruction=False)
    b = ix.query(q, n_results=n, instruction=True)
    print(f"  查询: {q[:70]}")
    print(f"  无指令 top1: {a['ids'][0]}  sim={a['similarities'][0]:.4f}")
    print(f"  加指令 top1: {b['ids'][0]}  sim={b['similarities'][0]:.4f}")
    delta = b["similarities"][0] - a["similarities"][0]
    verdict = "无指令更好 -> 默认关闭 --instruction" if delta <= 0 else "加指令更好"
    print(f"  top1 相似度变化: {delta:+.4f}   {verdict}")
    return delta


def check_self_similarity(ix, col, samples=8, clip=1200):
    """自相似性: 唯一能同时验证 id/向量/文本/元数据对齐的测试。

    命中分三级, 不能只看「是不是自己」:
      exact      自己                  —— 理想
      same_doc   同篇文档的相邻块       —— 可接受: 切块有 overlap, 相邻块本来
                                          就共享文本; 对 RAG 无害, 取回来的
                                          仍是同一篇的相关内容
      cross_doc  别的文档              —— 危险信号: 说明 emb 行序与 df 行序
                                          错位, 或 metadata 串了行
    所以判据是「cross_doc == 0」, 而不是「exact == samples」。
    """
    print()
    print("=" * 72)
    print("4) 自相似性 (取库中原文作查询, top1 应命中本块或同篇)")
    print("=" * 72)
    total = col.count()
    exact = same_doc = cross = 0
    sims = []
    for k in range(samples):
        # 均匀抽样, 覆盖库的首/中/尾, 避免只验证局部
        offset = k * (total // samples)
        got = col.get(limit=1, offset=offset)
        doc_id, text = got["ids"][0], got["documents"][0]
        r = ix.query(text[:clip], n_results=1)
        top1 = r["ids"][0]
        is_exact = top1 == doc_id
        is_same = top1.rsplit("_", 1)[0] == doc_id.rsplit("_", 1)[0]
        exact += is_exact
        same_doc += is_same
        cross += not is_same
        sims.append(r["similarities"][0])
        tag = "exact" if is_exact else ("same-doc" if is_same else "CROSS-DOC!")
        print(f"  [{k}] {doc_id} -> top1 {top1} "
              f"sim={r['similarities'][0]:.4f}  {tag}")
    print(f"  exact {exact}/{samples}   同篇 {same_doc}/{samples}   "
          f"跨篇 {cross}/{samples}   平均自相似 {np.mean(sims):.4f}")
    return exact, same_doc, cross, samples


def check_filter(ix, n=2):
    """元数据过滤必须验证「返回值满足条件」, 仅仅不报错不算通过。"""
    print()
    print("=" * 72)
    print("5) 元数据过滤")
    print("=" * 72)
    q = PROBE_QUERIES[0]
    allpass = True
    for w in FILTERS:
        key = list(w.keys())[0]
        try:
            r = ix.query(q, n_results=n, where_filter=w)
            vals = [m.get(key) for m in r["metadatas"]]
            ok = all(v == w[key] for v in vals)
            allpass &= ok
            print(f"  where={json.dumps(w, ensure_ascii=False):<24} "
                  f"返回 {len(r['ids'])} 条  字段值={vals}  "
                  f"sim={r['similarities'][0]:.4f}  {'PASS' if ok else 'FAIL'}")
        except Exception as e:                                # noqa: BLE001
            allpass = False
            print(f"  where={json.dumps(w, ensure_ascii=False):<24} "
                  f"ERR {type(e).__name__}: {e}")
    return allpass


def check_edge(ix):
    print()
    print("=" * 72)
    print("6) 边界情况")
    print("=" * 72)
    for label, q in EDGE_CASES:
        try:
            r = ix.query(q, n_results=1)
            print(f"  {label:<22} -> OK  sim={r['similarities'][0]:.4f} "
                  f"id={r['ids'][0]}")
        except Exception as e:                                # noqa: BLE001
            print(f"  {label:<22} -> ERR {type(e).__name__}: {e}")
    print("\n  注: 空查询仍能返回结果 —— Ollama 对空串也给出非零向量, "
          "Chroma 照常检索。\n      不崩溃即可, 但上层(r3 生成)应加非空校验, "
          "避免空查询污染上下文。")


def main():
    ap = argparse.ArgumentParser(description="索引质量验证")
    ap.add_argument("--collection", default="medical_chunks")
    ap.add_argument("--samples", type=int, default=8, help="自相似抽样数")
    ap.add_argument("--n", type=int, default=3, help="每条查询返回数")
    ap.add_argument("--log", action="store_true", help="写日志到 data/chunks/")
    args = ap.parse_args()

    if args.log:
        sys.stdout = E.Tee(os.path.join(ROOT, "data", "chunks",
                                        "verify_index.log"))

    ix = E.MedicalIndex(args.collection)
    col = ix.col

    try:
        df = E.pd.read_parquet(E.CHUNKS_PARQUET)
        expect = len(df)
    except Exception:                                         # noqa: BLE001
        expect = None

    cnt = check_basic(col, expect)
    check_semantic(ix, args.n)
    check_instruction(ix, args.n)
    exact, same_doc, cross, tot = check_self_similarity(ix, col, args.samples)
    filter_ok = check_filter(ix)
    check_edge(ix)

    print()
    print("=" * 72)
    print("汇总")
    print("=" * 72)
    print(f"  向量数量        {'PASS' if (expect is None or cnt == expect) else 'FAIL'} ({cnt})")
    print(f"  自相似性        {'PASS' if cross == 0 else 'FAIL'} "
          f"(exact {exact}/{tot}, 同篇 {same_doc}/{tot}, 跨篇 {cross}/{tot})")
    print(f"  元数据过滤      {'PASS' if filter_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
