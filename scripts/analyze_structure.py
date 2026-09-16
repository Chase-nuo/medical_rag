# -*- coding: utf-8 -*-
"""
语料结构分析 (RAG 任务2)
========================
目标：回答 3 个问题
  1) 数据集有哪些字段？每个字段的缺失率是多少？（缺失 > 阈值要定清洗策略）
  2) 文本质量体检：极短 / 乱码 / 中文混杂 / 编码异常
  3) 关键字段 title / journal / pub_date / pmid 能否作为元数据过滤器

数据源(三者字段不同)：
  - manifest.csv    : 抓取清单(我们生成的)
  - <pmcid>.json    : PMC OA 官方元数据(json 包)
  - <pmcid>.txt     : PMC OA 全文(自带 JOURNAL/ARTICLE INFORMATION 头)
本脚本以 JSON 为字段基准、以 TXT 头块为补充、以正文做质量体检。
"""
import csv, glob, json, os, re
from collections import Counter

ROOT = r"d:\medical_rag\data\pmc_oa"
OUT_FIELDS_CSV = r"d:\medical_rag\data\pmc_oa\corpus_fields.csv"

# TXT 头块里值得提取的键(键名在文件里是 "Key: value" 形式)
HEADER_KEYS = [
    "NLM Title Abbreviation", "NLM Journal ID", "ISSN", "EISSN", "Publisher",
    "PMCID", "PMID", "DOI", "Article ID", "Article version", "Subjects",
    "Electronic publication date", "Publication date",
    "Volume", "Issue", "First page", "Last page",
]

# 摘要可能用的标题词(结构化摘要常没有 Abstract 一行, 直接 Background:/Objective: 开头)
# 注意: re.M 让 ^ 匹配每一行行首, 否则只会看整个文本开头
ABSTRACT_HEAD_RE = re.compile(
    r"^\s*(abstract|summary)\b", re.I | re.M)

# 常见的章节标题词(做 IMRaD 观察用)
SECTION_RE = re.compile(
    r"^\s*[0-9.]*\s*(introduction|methods|materials and methods|results|"
    r"discussion|conclusion|conclusions|references|background|objective|"
    r"author contributions)\b\s*:?\s*$", re.I | re.M)

def clean_byte_metrics(raw: bytes):
    """编码体检: 是否含非法字节、替换符 U+FFFD、可疑 mojibake 串"""
    n_bad_bytes = 0
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError:
        n_bad_bytes = 1
    text = raw.decode("utf-8", errors="replace")
    mojibake = sum(text.count(p) for p in ("\ufffd", "Ã©", "Ã¨", "â€™", "â€", "ï¿½"))
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return text, n_bad_bytes, mojibake, cjk

def parse_txt_header(text: str, lines_limit: int = 80):
    """从 TXT 开头抓 'Key: value' 形式的元数据(仅前 lines_limit 行, 避免误读正文)"""
    kv = {}
    for ln in text.splitlines()[:lines_limit]:
        m = re.match(r"^\s*([A-Za-z][A-Za-z /()]*?):\s*(.+?)\s*$", ln)
        if m and m.group(1) in HEADER_KEYS and m.group(1) not in kv:
            kv[m.group(1)] = m.group(2)
    return kv

def analyze_one(json_path: str) -> dict:
    d = json.load(open(json_path, encoding="utf-8"))
    pmc = d["pmcid"]
    txt_path = os.path.join(os.path.dirname(json_path), f"{pmc}.{d['version']}.txt")
    if not os.path.exists(txt_path):
        txt_path = glob.glob(os.path.join(os.path.dirname(json_path), "*.txt"))[0]
    raw = open(txt_path, "rb").read()
    text, n_bad, mojibake, cjk = clean_byte_metrics(raw)
    h = parse_txt_header(text)
    lines = text.splitlines()
    n_lines = len(lines)
    n_chars = len(text)
    # 从全文看是否存在摘要标题、章节标题
    has_abstract = bool(ABSTRACT_HEAD_RE.search(text))
    has_section = bool(SECTION_RE.search(text))
    # 出版年份: 优先引文(citation)中的年份, 再退到 Publication date 行
    cit = d.get("citation") or ""
    m_year = re.search(r"(19|20)\d{2}", cit)
    year = m_year.group(0) if m_year else ""
    # 期刊: JSON 无独立期刊字段, 从 TXT 头取 NLM 缩写
    journal = h.get("NLM Title Abbreviation", "")
    return {
        "pmcid": pmc,
        "title": d.get("title", ""),
        "pmid": str(d.get("pmid") or ""),       # 空串视为缺失
        "doi": str(d.get("doi") or ""),
        "license": d.get("license_code", ""),
        "journal_nlm": journal,
        "pub_year": year,
        "pub_date_line": (h.get("Electronic publication date") or h.get("Publication date") or ""),
        "issn": (h.get("ISSN") or h.get("EISSN") or ""),
        "publisher": h.get("Publisher", ""),
        "has_abstract_head": has_abstract,
        "has_section_head": has_section,
        "n_lines": n_lines,
        "n_chars": n_chars,
        "size_kb": round(len(raw) / 1024, 1),
        "bad_bytes": n_bad,
        "mojibake_cnt": mojibake,
        "cjk_chars": cjk,
    }

def main():
    jsons = sorted(glob.glob(os.path.join(ROOT, "*", "*.json")))
    recs = [analyze_one(p) for p in jsons]
    fields = list(recs[0].keys())

    # 1) Field missingness matrix
    print("== 1) Field missingness ==")
    print(f"{'field':<20}{'missing':>8}{'rate%':>8}   missing samples")
    # 缺失率只对"文本型字段"有意义。has_* 是脚本算出来的布尔量, 写入 CSV 后
    # 是 "True"/"False" 两种非空字符串, 放进缺失矩阵会恒为 0%, 毫无信息量
    # —— 因此拆出来单独按"阳性率"统计。
    for f in ["title", "pmid", "doi", "license", "journal_nlm", "pub_year",
              "pub_date_line", "issn", "publisher"]:
        miss = [r["pmcid"] for r in recs if not r[f]]
        # 千篇级别下 missing 名单可能有几百个, 只展示前 15 个, 全量落 CSV
        shown = ",".join(miss[:15]) + (f" ...(+{len(miss) - 15})" if len(miss) > 15 else "")
        print(f"{f:<20}{len(miss):>8}{len(miss) / len(recs) * 100:>8.1f}   {shown}")
    print()
    for f in ("has_abstract_head", "has_section_head"):
        # recs 是内存里的 dict, 值是真正的 bool; 若从 CSV 读回来则是 "True" 字符串。
        # 统一 str() 再比, 两种来源都成立。
        pos = sum(1 for r in recs if str(r[f]) == "True")
        print(f"{f:<20}{pos:>8}{pos / len(recs) * 100:>8.1f}   (阳性率, 非缺失率)")

    # 2) Quality check summary
    print("\n== 2) Quality check ==")
    bad = [r for r in recs if r["bad_bytes"] or r["mojibake_cnt"] or r["cjk_chars"]]
    print(f"decode_bad={sum(r['bad_bytes'] for r in recs)} "
          f"mojibake_articles={sum(1 for r in recs if r['mojibake_cnt'])} "
          f"cjk_articles={sum(1 for r in recs if r['cjk_chars']>0)}")
    for r in bad[:20]:  # 只列前 20 篇异常, 全量明细见 corpus_fields.csv
        print(f"  {r['pmcid']}: lines={r['n_lines']} size={r['size_kb']}KB "
              f"moji={r['mojibake_cnt']} cjk={r['cjk_chars']}")
    if len(bad) > 20:
        print(f"  ...(共 {len(bad)} 篇异常, 明细见 corpus_fields.csv)")
    short = [r["pmcid"] for r in recs if r["n_lines"] < 100]
    print(f"short_text(<100 lines): {len(short)} 篇"
          + (f"  示例 {','.join(short[:15])}" if short else ""))

    # 3) Can key fields serve as metadata filters?
    print("\n== 3) Metadata filter feasibility ==")
    jc = Counter(r["journal_nlm"] for r in recs if r["journal_nlm"])
    yc = Counter(r["pub_year"] for r in recs if r["pub_year"])
    print(f"distinct_journals={len(jc)}  journal_nlm_missing={sum(1 for r in recs if not r['journal_nlm'])}")
    print(f"pub_year_missing={sum(1 for r in recs if not r['pub_year'])}  year_range={min(yc)}-{max(yc)}")
    print(f"pmid_missing={sum(1 for r in recs if not r['pmid'])}  pmid_link_ok={sum(1 for r in recs if r['pmid'])}")

    # 存档: 供后续任务(切块/长度分析)复用
    with open(OUT_FIELDS_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(recs)
    print(f"\n[saved] {OUT_FIELDS_CSV}  ({len(recs)} rows)")

if __name__ == "__main__":
    main()
