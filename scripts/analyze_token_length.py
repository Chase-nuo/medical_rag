# -*- coding: utf-8 -*-
"""
文本 token 长度量化分析与分位数 (RAG 任务4)
==========================================
回答 4 个问题:
  1) 单篇正文的 token 长度分布如何? 各分位(P50/P75/P90/P95/P99)是多少?
  2) 语料的语言构成如何? 中英混排对 token 计量有什么影响?
  3) 段落级 token 长度分布如何? 有多少段落会顶破常见 chunk_size?
  4) 按 chunk_size 切块时, 各候选尺寸下的块数与块长分布如何? 该选多大?

与任务2/3的关系:
  - 任务2的 n_chars、任务3的分层抽样都用"字符数"充当 token 的代理;
    本任务改用本文嵌入模型的真实 tokenizer 做精确计量, 并量化该代理的误差。
  - 计量对象分三段, 因为 PMC 纯文本自带大量非论文主体内容:
      full : 整个 txt 文件(含元数据头块 + 参考文献 + 同行评议附录)
      body : \x9f====...====\x9f 分隔线(PMC 用于隔离元数据头块)之后的正文
      core : body 再截断到首个 References / Supporting information /
             Comments to the Author 独立行 -> 去掉参考文献与评议附录,
             即真正会被切块入库的内容
  - 头块与尾部噪声本就该在切块前剔除, 三段对比正好量化这份"需要剔除的体积"。

实测要点(2026-09, bge-m3 / 30 篇):
  - 英文篇 4.02 字符/token, 中文篇 1.82(纯中文约 1.4); 混排语料必须按语言分别换算,
    若统一按 3.8 换算, 中文篇的 token 数会被低估约 2.2 倍。
  - 元数据头块+参考文献+评议附录平均占全文 token 的 35%, 切块前应先剔除。
  - 表格被 PMC 纯文本展平成单段数千 token 的"巨型段落"(最大 6393), 切块时会被硬切断。
  - 推荐 chunk_size=512 token(超限段落占比 2.15%, 含 overlap 约 660 块)。

用法:
    python analyze_token_length.py
    python analyze_token_length.py --tokenizer bge-m3 --max-input 8192
    python analyze_token_length.py --chunk-sizes 256,512,1024
"""
import argparse
import csv
import glob
import os
import re
import statistics

from tokenizers import Tokenizer

ROOT = r"d:\medical_rag\data\pmc_oa"
TOKENIZER_ROOT = r"d:\medical_rag\data\tokenizer"
FIELDS_CSV = os.path.join(ROOT, "corpus_fields.csv")
OUT_LENGTH_CSV = os.path.join(ROOT, "corpus_token_lengths.csv")
OUT_CHUNK_CSV = os.path.join(ROOT, "chunk_size_simulation.csv")

# PMC 正文分隔线: 一行由 0x9F(C1 控制符) 包住若干 '=' 组成, 每篇唯一
BODY_MARK_RE = re.compile(r"^\x9f=+\x9f$")
# 正文结束标志: 出现以下独立行即认为主体内容结束(PMC 里大写/小写混用, 故忽略大小写)
TAIL_HEAD_RE = re.compile(
    r"^(references?|bibliography|literature cited|supporting information|"
    r"supplementary materials?|comments to the author|"
    r"review comments to the author)$", re.I)
# 句子切分(用于超长段落的降级处理)
SENT_RE = re.compile(r"(?<=[.;!?])\s+")
# 判定为 CJK 的字符区间
CJK_RANGE = ("\u4e00", "\u9fff")

QUANTILES = [0, 5, 25, 50, 75, 90, 95, 99, 100]
QNAMES = ["min", "p5", "p25", "p50", "p75", "p90", "p95", "p99", "max"]
LANGS = ("en", "mixed", "zh")


# ---------------------------------------------------------------- 基础工具
def quantile(vals, q):
    """线性插值分位数(等价 numpy.percentile 默认方式), 避免依赖 numpy"""
    xs = sorted(vals)
    if not xs:
        return 0.0
    if len(xs) == 1:
        return float(xs[0])
    pos = (len(xs) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def print_quantile_table(title, rows):
    """rows: [(名称, [值...])], 空值自动跳过"""
    print(f"\n{title}")
    print(f"{'item':<18}{'N':>5}" + "".join(f"{q:>9}" for q in QNAMES)
          + f"{'mean':>10}{'sd':>8}")
    for name, vals in rows:
        if not vals:
            continue
        cells = "".join(f"{quantile(vals, q):>9.0f}" for q in QUANTILES)
        print(f"{name:<18}{len(vals):>5}{cells}"
              f"{statistics.fmean(vals):>10.0f}{statistics.pstdev(vals):>8.0f}")


# ---------------------------------------------------------------- 文本分段
def split_regions(lines):
    """按 PMC 结构切成 (full, body, core, tail_cut); tail_cut 表示尾部噪声是否被剔除"""
    idx = next((i for i, l in enumerate(lines) if BODY_MARK_RE.match(l.strip())), None)
    if idx is None:                      # 没有分隔线: 退化为整篇
        return lines, lines, lines, False
    body = lines[idx + 1:]
    end = next((k for k, l in enumerate(body) if TAIL_HEAD_RE.match(l.strip())), None)
    if end is None:
        return lines, body, body, False
    return lines, body, body[:end], True


def to_paragraphs(lines):
    """空行分段; 段内多行用空格拼接"""
    paras, buf = [], []
    for l in lines:
        if l.strip():
            buf.append(l.strip())
        elif buf:
            paras.append(" ".join(buf))
            buf = []
    if buf:
        paras.append(" ".join(buf))
    return paras


# ---------------------------------------------------------------- 切块模拟
def pack_chunks(paras, tok, chunk_size):
    """贪心按段落装箱成 <= chunk_size token 的块(不计 overlap)。
    超长段落先按句子拆; 句子仍超长则按 token 滑窗硬切。
    返回 (块 token 长度列表, 被强行打断的段落数)"""
    units = []          # [(n_tokens, text)]
    hard_split = 0
    for p in paras:
        ids = tok.encode(p, add_special_tokens=False).ids
        if len(ids) <= chunk_size:
            units.append((len(ids), p))
            continue
        hard_split += 1
        for s in SENT_RE.split(p):
            s = s.strip()
            if not s:
                continue
            sid = tok.encode(s, add_special_tokens=False).ids
            if len(sid) <= chunk_size:
                units.append((len(sid), s))
            else:
                for i in range(0, len(sid), chunk_size):
                    part = sid[i:i + chunk_size]
                    units.append((len(part), tok.decode(part)))
    lens, cur = [], 0
    for n, _ in units:
        if cur and cur + n > chunk_size:
            lens.append(cur)
            cur = 0
        cur += n
    if cur:
        lens.append(cur)
    return lens, hard_split


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser(description="文本 token 长度量化分析与分位数")
    ap.add_argument("--tokenizer", default="bge-m3",
                    help="data/tokenizer/<名称>/tokenizer.json (默认 bge-m3)")
    ap.add_argument("--tokenizer-json", default=None, help="直接指定 tokenizer.json 路径")
    ap.add_argument("--max-input", type=int, default=8192,
                    help="嵌入模型最大输入 token 数 (bge-m3 默认 8192)")
    ap.add_argument("--chunk-sizes", default="128,256,384,512,768,1024",
                    help="待评估的 chunk_size 候选(逗号分隔)")
    args = ap.parse_args()

    tok_path = args.tokenizer_json or os.path.join(
        TOKENIZER_ROOT, args.tokenizer, "tokenizer.json")
    if not os.path.exists(tok_path):
        raise SystemExit(f"[error] 找不到 tokenizer: {tok_path}")

    tok = Tokenizer.from_file(tok_path)
    candidates = [int(x) for x in args.chunk_sizes.split(",") if x.strip()]

    def ntok(text):
        """正文长度计量: 不计特殊 token(模型推理时会自动加 <s>/</s>, 约 +2)"""
        return len(tok.encode(text, add_special_tokens=False).ids)

    # ---- 0) tokenizer 基本信息 ----
    print("== 0) Tokenizer ==")
    print(f"file={tok_path}")
    print(f"vocab_size={tok.get_vocab_size()}  max_input_tokens={args.max_input} "
          f"(特殊 token 另计, 通常 +2)")
    for label, probe in [("en", "Type 2 diabetes mellitus (T2DM) is associated "
                               "with a two-fold increased risk of pre-eclampsia."),
                         ("zh", "2型糖尿病与先兆子痫风险增加两倍相关。")]:
        pe = tok.encode(probe, add_special_tokens=False)
        print(f"probe[{label}] chars={len(probe)} tokens={len(pe.ids)} "
              f"chars_per_token={len(probe) / max(1, len(pe.ids)):.3f}")
    print(f"probe_pieces={tok.encode('Type 2 diabetes mellitus', add_special_tokens=False).tokens}")

    # ---- 1) 逐篇量化 ----
    recs = list(csv.DictReader(open(FIELDS_CSV, encoding="utf-8-sig")))
    rows = []
    for r in recs:
        pmc = r["pmcid"]
        hits = glob.glob(os.path.join(ROOT, pmc + ".*", "*.txt"))
        if not hits:
            print(f"[warn] {pmc}: 找不到 txt, 跳过")
            continue
        raw = open(hits[0], encoding="utf-8", errors="replace").read()
        lines = raw.splitlines()
        full, body, core, tail_cut = split_regions(lines)

        core_text = "\n".join(core)
        t_core = ntok(core_text)
        c_core = len(core_text)
        cjk = sum(1 for ch in core_text if CJK_RANGE[0] <= ch <= CJK_RANGE[1])
        cjk_ratio = cjk / max(1, c_core)
        lang = "zh" if cjk_ratio >= 0.20 else ("mixed" if cjk_ratio >= 0.02 else "en")

        paras = to_paragraphs(core)
        p_tokens = [ntok(p) for p in paras]

        rows.append({
            "pmcid": pmc,
            "title": r["title"][:70],
            "lang": lang,
            "lines_full": len(lines),
            "chars_core": c_core,
            "cjk_chars_core": cjk,
            "cjk_ratio": round(cjk_ratio, 3),
            "tokens_full": ntok("\n".join(full)),
            "tokens_body": ntok("\n".join(body)),
            "tokens_core": t_core,
            "chars_per_token_core": round(c_core / max(1, t_core), 3),
            "removed_pct": round((ntok("\n".join(full)) - t_core)
                                 / max(1, ntok("\n".join(full))) * 100, 1),
            "tail_cut": tail_cut,
            "n_paras": len(paras),
            "para_tok_p50": int(quantile(p_tokens, 50)),
            "para_tok_p90": int(quantile(p_tokens, 90)),
            "para_tok_max": max(p_tokens) if p_tokens else 0,
            "tab_lines": sum(1 for l in core if "\t" in l),
            "core_paras": paras,
            "_p_tokens": p_tokens,
        })

    rows.sort(key=lambda x: x["tokens_core"])

    print("\n== 1) 逐篇 token 长度 (按 core 升序) ==")
    print(f"{'pmcid':<14}{'chars_c':>9}{'tok_full':>9}{'tok_body':>9}{'tok_core':>9}"
          f"{'drop%':>7}{'c/t':>6}{'cjk%':>6}{'paras':>7}{'p_max':>7}{'tab':>6}{'tail':>6}")
    for x in rows:
        print(f"{x['pmcid']:<14}{x['chars_core']:>9}{x['tokens_full']:>9}"
              f"{x['tokens_body']:>9}{x['tokens_core']:>9}{x['removed_pct']:>7}"
              f"{x['chars_per_token_core']:>6}{x['cjk_ratio'] * 100:>6.0f}"
              f"{x['n_paras']:>7}{x['para_tok_max']:>7}{x['tab_lines']:>6}"
              f"{('Y' if x['tail_cut'] else '-'):>6}")

    # ---- 2) 文档级分位数 ----
    print("\n== 2) 文档级分位数 (N=篇数) ==")
    print_quantile_table("token 长度:", [
        ("tokens_full", [x["tokens_full"] for x in rows]),
        ("tokens_body", [x["tokens_body"] for x in rows]),
        ("tokens_core", [x["tokens_core"] for x in rows]),
    ])
    print_quantile_table("辅助量:", [
        ("chars_core", [x["chars_core"] for x in rows]),
        ("n_paras", [x["n_paras"] for x in rows]),
        ("chars/token*100", [x["chars_per_token_core"] * 100 for x in rows]),
        ("removed_pct", [x["removed_pct"] for x in rows]),
    ])
    n_over = sum(1 for x in rows if x["tokens_core"] > args.max_input)
    print(f"\n超过模型单次上限({args.max_input} token)的篇数: {n_over}/{len(rows)}"
          f" -> 超限篇目必须切块后才能入库")

    # ---- 3) 语言构成与尾部剔除覆盖率 ----
    print("\n== 3) 语言构成与 core 剔除覆盖率 ==")
    for lg in LANGS:
        grp = [x for x in rows if x["lang"] == lg]
        if grp:
            print(f"  {lg:<6} n={len(grp):<3} {','.join(x['pmcid'] for x in grp)}")
    print_quantile_table("按语言分组的 core token 长度:", [
        (f"tokens_core[{lg}]", [x["tokens_core"] for x in rows if x["lang"] == lg])
        for lg in LANGS])
    print_quantile_table("按语言分组的 chars/token*100:", [
        (f"c/t[{lg}]*100", [x["chars_per_token_core"] * 100
                            for x in rows if x["lang"] == lg]) for lg in LANGS])
    not_cut = [x["pmcid"] for x in rows if not x["tail_cut"]]
    print(f"\n尾部噪声成功剔除(识别到 References/... 标题) 的篇数: "
          f"{sum(1 for x in rows if x['tail_cut'])}/{len(rows)}")
    if not_cut:
        print(f"  未识别尾部边界(仍含参考文献/附录, token 偏高): {','.join(not_cut)}")

    # ---- 4) 段落级分位数 ----
    all_p = [t for x in rows for t in x["_p_tokens"]]
    print("\n== 4) 段落级分布 (core 段, 单位 token) ==")
    print_quantile_table("段落:", [("para_tokens", all_p)])
    giant = [t for t in all_p if t > 512]
    tab_docs = [x["pmcid"] for x in rows if x["tab_lines"] > 0]
    print(f"\n巨型段落(>512 token) {len(giant)} 个, 最大 {max(all_p)}"
          f" -> 多由表格/参考文献展平而来, 切块时会被硬切断")
    print(f"含制表符(疑似表格)的篇目 {len(tab_docs)}/{len(rows)}: {','.join(tab_docs)}")

    # ---- 5) 切块模拟 ----
    print(f"\n== 5) 切块模拟 (仅计 core 正文, 不含 overlap) ==\n{'-' * 76}")
    print(f"{'chunk_size':>10}{'超限段落':>10}{'占比%':>8}{'被打断篇数':>12}"
          f"{'总块数':>9}{'块/篇p50':>10}{'块/篇max':>10}{'填充率p50%':>12}")
    sim_rows = []
    for cs in candidates:
        over = sum(1 for t in all_p if t > cs)
        chunk_lens, n_split_docs, per_doc = [], 0, []
        for x in rows:
            lens, hard = pack_chunks(x["core_paras"], tok, cs)
            chunk_lens.extend(lens)
            per_doc.append(len(lens))
            if hard:
                n_split_docs += 1
        sim_rows.append({
            "chunk_size": cs,
            "paras_over": over,
            "paras_over_pct": round(over / max(1, len(all_p)) * 100, 2),
            "docs_with_split": n_split_docs,
            "n_chunks": len(chunk_lens),
            "chunks_per_doc_p50": int(quantile(per_doc, 50)),
            "chunks_per_doc_max": max(per_doc) if per_doc else 0,
            "chunk_len_p50": int(quantile(chunk_lens, 50)),
            "chunk_len_p90": int(quantile(chunk_lens, 90)),
            "chunk_len_p99": int(quantile(chunk_lens, 99)),
            "chunk_len_max": max(chunk_lens) if chunk_lens else 0,
            "fill_rate_p50": round(quantile(chunk_lens, 50) / cs * 100, 1),
            "est_chunks_overlap125": int(len(chunk_lens) / 0.875),
        })
        s = sim_rows[-1]
        print(f"{cs:>10}{over:>10}{s['paras_over_pct']:>8}{n_split_docs:>12}"
              f"{s['n_chunks']:>9}{s['chunks_per_doc_p50']:>10}"
              f"{s['chunks_per_doc_max']:>10}{s['fill_rate_p50']:>12}")

    print(f"\n块长分布 (overlap 按 stride=87.5%*chunk_size 估算块数):")
    print(f"{'chunk_size':>10}{'p50':>7}{'p90':>7}{'p99':>7}{'max':>7}"
          f"{'overlap后块数':>14}")
    for s in sim_rows:
        print(f"{s['chunk_size']:>10}{s['chunk_len_p50']:>7}{s['chunk_len_p90']:>7}"
              f"{s['chunk_len_p99']:>7}{s['chunk_len_max']:>7}"
              f"{s['est_chunks_overlap125']:>14}")

    # ---- 6) 结论 ----
    print("\n== 6) 结论与建议 ==")
    tok_core = [x["tokens_core"] for x in rows]
    cpt_en = [x["chars_per_token_core"] for x in rows if x["lang"] == "en"]
    cpt_zh = [x["chars_per_token_core"] for x in rows if x["lang"] == "zh"]
    cpt_all = statistics.fmean([x["chars_per_token_core"] for x in rows])
    m_en = statistics.fmean(cpt_en) if cpt_en else cpt_all
    m_zh = statistics.fmean(cpt_zh) if cpt_zh else cpt_all
    print(f"1) core 段 token: 中位 {quantile(tok_core, 50):.0f}, "
          f"P95 {quantile(tok_core, 95):.0f}, 最大 {max(tok_core)}; "
          f"超 {args.max_input} token 的有 {n_over} 篇 -> 单篇必须切块。")
    print(f"2) 字符/token: 全语料 {cpt_all:.2f}, 英文 {m_en:.2f}, 中文 {m_zh:.2f}"
          f" -> 任务2的 n_chars 若统一按 {cpt_all:.1f} 换算, 中文篇 token 数会被低估约 "
          f"{m_en / m_zh:.1f} 倍; 建议英文用 {m_en:.1f}、中文用 {m_zh:.1f} 分别换算。")
    print(f"3) 元数据头块+参考文献+评议附录平均占全文 token 的 "
          f"{statistics.fmean([x['removed_pct'] for x in rows]):.1f}% "
          f"-> 切块前先剔除; 另有 {len(giant)} 个 >512 token 的表格型巨型段落需单独处理。")
    band = {lim: [cs for cs in candidates
                  if sum(1 for t in all_p if t > cs) / max(1, len(all_p)) <= lim]
            for lim in (0.01, 0.05, 0.10)}
    print(f"4) 段落保留度: 超限段落占比 <=1% 需 chunk_size>={min(band[0.01])}, "
          f"<=5% 需 >={min(band[0.05])}, <=10% 需 >={min(band[0.10])}。")
    rec = min(band[0.05])
    print(f"   推荐 chunk_size = {rec} token"
          f"(英文约 {rec * m_en:.0f} 字符 / 中文约 {rec * m_zh:.0f} 字符), "
          f"overlap 取 10%~15%; 该尺寸下总块数约 "
          f"{[s['n_chunks'] for s in sim_rows if s['chunk_size'] == min(band[0.05])][0]}"
          f" (含 overlap 约 "
          f"{[s['est_chunks_overlap125'] for s in sim_rows if s['chunk_size'] == min(band[0.05])][0]})。")
    print(f"5) chunk_size 上限须 < {args.max_input}(模型最大输入); "
          f"若后续换用 512 维以下小模型需按新 tokenizer 重跑本脚本。")

    # ---- 存档 ----
    fields = [k for k in rows[0].keys()
              if not k.startswith("_") and k not in ("core_paras",)]
    with open(OUT_LENGTH_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for x in rows:
            w.writerow({k: x[k] for k in fields})
    with open(OUT_CHUNK_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sim_rows[0].keys()))
        w.writeheader()
        w.writerows(sim_rows)
    print(f"\n[saved] {OUT_LENGTH_CSV}  ({len(rows)} rows)")
    print(f"[saved] {OUT_CHUNK_CSV}  ({len(sim_rows)} rows)")


if __name__ == "__main__":
    main()
