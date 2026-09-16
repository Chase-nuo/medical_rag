# -*- coding: utf-8 -*-
"""
领域内容理解 (RAG 任务3)
========================
回答 4 个问题:
  1) 正文是否遵循 IMRaD/结构化写作? 章节标题样式?
  2) 术语/缩写密度(EGFR/PCI 这类缩写多不多) + 常见缩写及其展开
  3) 同一概念的不同表述(heart attack vs myocardial infarction)
  4) 短/中/长分层抽样(按正文长度分位), 供人工精读

说明:
  - 抽样按"字符数"做分位是 token 的轻量代理;
    任务4会换成本文采用的嵌入模型 tokenizer 做精确长度量化。
"""
import csv, glob, json, os, re
from collections import Counter

ROOT = r"d:\medical_rag\data\pmc_oa"
FIELDS_CSV = os.path.join(ROOT, "corpus_fields.csv")
# 产物(供报告与后续评估复用)
OUT_SECTIONS = os.path.join(ROOT, "corpus_sections.csv")
OUT_ACRONYMS = os.path.join(ROOT, "corpus_acronyms.csv")
OUT_SYNONYMS = os.path.join(ROOT, "corpus_synonyms.csv")
OUT_STRATA = os.path.join(ROOT, "corpus_sample_strata.csv")

# ---- 1) 章节标题 ---------------------------------------------
# 把各种写法归一化到 7 类: intro/methods/results/discussion/conclusion/refs/objective
SECTION_ALIAS = {
    "introduction": "intro", "background": "intro",
    "methods": "methods", "materials and methods": "methods",
    "patients and methods": "methods", "subjects and methods": "methods",
    "study design": "intro",
    "results": "results",
    "discussion": "discussion",
    "conclusion": "conclusion", "conclusions": "conclusion",
    "references": "refs", "bibliography": "refs",
    "objective": "objective", "objectives": "objective",
    "aim": "objective",
}
SECTION_LINE = re.compile(
    r"^\s*(?:[0-9]+[.)]?\s*)?(" + "|".join(SECTION_ALIAS) + r")\s*[:\-]?\s*$",
    re.I)

# 中文编号章节标题: 如 "1 资料与方法" / "2.2 CT和SPECT显像改变"
CJK_HEAD = re.compile(r"^\s*\d+(\.\d+)*\s+[\u4e00-\u9fff]")

# ---- 2) 缩写/词频 ---------------------------------------------
# 先取出候选 token, 再要求 "全大写(允许带数字)" 且 "非纯数字" -> 过滤掉 10/001 这类
WORD = re.compile(r"[A-Za-z][A-Za-z'\-]{1,}")
TOKEN = re.compile(r"\b[A-Za-z0-9]{2,12}\b")
NOISE_ACRO = {"IT", "OR", "IN", "TO", "AT", "AS", "BY", "USA", "PLOS", "ONE",
              "NLM", "MD", "PHD", "ID", "NA", "VOL", "NO", "ET", "AL", "VS",
              # PMC txt 头部块重复词, 非正文内容
              "INFORMATION", "JOURNAL", "ARTICLE", "ABBREVIATION", "SUBJECTS",
              "ISSN", "EISSN", "PMCID", "VERSION", "PUBLICATION", "ELECTRONIC",
              "PUBLISHER", "DOI", "VOLUME", "ISSUE", "PAGE", "LICENSE"}
def is_acro(tok):
    return tok.isupper() and not tok.isdigit() and tok not in NOISE_ACRO
STOP = set("""a an the and or but of to in on for with by from as at be is are was were
              this that these those it its we our their his her not no if than then so such
              also more most other between among into during after before while using used
              use may might can could should would will results result study studies data
              patients patient group groups year years old new methods method model models
              analysis analyzed associated significant significantly increase increased
              decrease decreased higher lower compared comparison association level levels
              time times effect effects due risk total number percentage percent within
              table figure fig et al vs""".split())

def read_text(pmcid):
    """按 PMCID 找正文 txt。必须通配版本号: 语料扩容后会出现 .2/.3 版本,
    写死 '.1' 会静默读空。找不到时返回空串而不是抛异常(便于批量跑完再排查)。"""
    hits = glob.glob(os.path.join(ROOT, pmcid + ".*", "*.txt"))
    if not hits:
        return ""
    return open(hits[0], encoding="utf-8", errors="replace").read()

def section_scan(text):
    """扫描全文, 返回归一化章节类别出现顺序(可能含摘要+正文两份)"""
    order, seen_head = [], []
    for ln in text.splitlines():
        s = ln.strip()
        if not s or len(s) > 80:
            continue
        if CJK_HEAD.search(s):
            order.append("cjk_sec")
            seen_head.append(s[:40])
            continue
        m = SECTION_LINE.match(s)
        if m:
            order.append(SECTION_ALIAS[m.group(1).lower()])
            seen_head.append(s[:40])
    return order, seen_head

# 一次扫描抓出全文所有 "全称 (ACRO)" 定义式。
# 旧写法对每个缩写各编译一条正则并扫一遍全文 -> 每篇要做 O(缩写数) 次全文扫描。
# 30 篇时无感, 千篇规模下这是主要瓶颈(实测 24 分钟仍未跑完), 故改为单趟扫描。
# 括号内容限定 [A-Z] 开头的大写串, 天然满足 is_acro 的"全大写"条件。
ACRO_DEF_RE = re.compile(
    r"([A-Za-z][A-Za-z\-]*(?:[ \t]+[A-Za-z][A-Za-z\-]*){1,7})[ \t]*"
    r"\(([A-Z][A-Z0-9\-]{1,11})\)")


def acronym_stats(text):
    """缩写频次 + 在文中的展开短语示例(形如: full name (ACRO))"""
    ac = Counter(t for t in TOKEN.findall(text) if is_acro(t))
    total_words = len(WORD.findall(text))
    # 用 [ \t] 而非 \s 连接词, 避免跨行把无关词卷进"全称"
    examples = {}
    for m in ACRO_DEF_RE.finditer(text):
        ph, a = m.group(1), m.group(2)
        if a in examples or a not in ac:   # 只保留首次出现; 未计入 ac 的(如噪声)跳过
            continue
        if len(ph) <= 70 and " " in ph:    # 排除超长误匹配与单词(全称至少 2 词)
            examples[a] = ph
    return ac, examples, total_words

def synonym_probe(text):
    """统计几个概念的不同书面表述"""
    hits = {}
    pats = [
        (r"\bmyocardial infarction\b", "myocardial infarction"),
        (r"\bheart attack\b", "heart attack"),
        (r"\bacute myocardial infarction\b", "acute MI(full)"),
        (r"\bhypertension\b", "hypertension"),
        (r"\bhigh blood pressure\b", "high blood pressure"),
        (r"\bdiabetes mellitus\b", "diabetes mellitus"),
        (r"\btype 2 diabetes\b", "type 2 diabetes"),
        (r"\bT2DM\b", "T2DM"),
        (r"\bAlzheimer'?s disease\b", "Alzheimer's disease"),
        (r"\bdementia\b", "dementia"),
        (r"\blung cancer\b", "lung cancer"),
        (r"\bnonsmall cell lung cancer\b|non-small cell lung cancer", "NSCLC(full)"),
    ]
    for pat, name in pats:
        hits[name] = len(re.findall(pat, text, re.I))
    return hits

def main():
    # load per-article char length from task2 archive
    recs = list(csv.DictReader(open(FIELDS_CSV, encoding="utf-8-sig")))
    recs.sort(key=lambda r: int(r["n_chars"]))
    n = len(recs)

    # ---- (1)+(2) 结构与缩写: 同一遍读完, 每篇只读一次磁盘 ----
    # 原实现每篇要读 3 次(结构 / 缩写 / 含缩写判定)。30 篇无所谓,
    # 扩容到千篇后重复 IO 是主要耗时, 故合并为单趟。
    print("== 1) Section structure ==")
    order_all = []
    rows = []
    all_ac = Counter()
    all_ex = {}
    acro_articles = 0
    for r in recs:
        pmc = r["pmcid"]
        text = read_text(pmc)
        order, heads = section_scan(text)
        order_all.append((pmc, order))
        c = Counter(order)
        rows.append((pmc,
                     c.get("intro", 0), c.get("methods", 0), c.get("results", 0),
                     c.get("discussion", 0), c.get("conclusion", 0),
                     c.get("refs", 0), c.get("objective", 0), c.get("cjk_sec", 0)))
        ac, ex, tw = acronym_stats(text)
        all_ac.update(ac)
        all_ex.update(ex)
        if any(is_acro(t) for t in TOKEN.findall(text)):
            acro_articles += 1
    print(f"{'pmc':<12}{'intro':>6}{'meth':>6}{'res':>6}{'disc':>6}{'conc':>6}{'refs':>6}{'obj':>6}{'cjk':>6}")
    for t in rows[:30]:  # 千篇级别只打前 30 行, 避免刷屏; 全量明细落 CSV
        print(f"{t[0]:<12}" + "".join(f"{x:>6}" for x in t[1:]))
    if len(rows) > 30:
        print(f"  ...(共 {len(rows)} 篇, 明细见 {os.path.basename(OUT_SECTIONS)})")
    imrad_full = sum(1 for _, o in order_all
                     if "intro" in o and "methods" in o and "results" in o
                     and ("discussion" in o or "conclusion" in o))
    has_refs = sum(1 for _, o in order_all if "refs" in o)
    cjk_any = sum(1 for _, o in order_all if "cjk_sec" in o)
    print(f"full_IMRaD_like={imrad_full}/{n}  has_references={has_refs}/{n}  has_cjk_sections={cjk_any}/{n}")

    # ---- (2) 缩写 / 词频 ----
    print("\n== 2) Acronyms & lexicon ==")
    print(f"distinct_acronyms={len(all_ac)}  articles_with_acro={acro_articles}/{n}")
    print("top30 acronyms:", dict(all_ac.most_common(30)))
    print("\nwith expansions (examples):")
    for a, c in all_ac.most_common(25):
        print(f"  {a:<14} x{c:<5} {all_ex.get(a, '(no explicit expansion)')[:70]}")

    # ---- (3) 同义表述 ----
    print("\n== 3) Synonym probe (corpus-wide counts) ==")
    agg = Counter()
    for r in recs:
        agg.update(synonym_probe(read_text(r["pmcid"])))
    for k, v in agg.items():
        print(f"  {k:<28} {v}")

    # ---- (4) 短/中/长分层抽样 ----
    print("\n== 4) Stratified sample (by n_chars) ==")
    q1, q2 = n // 3, 2 * n // 3
    strata = {"short": recs[:q1], "mid": recs[q1:q2], "long": recs[q2:]}
    for name, group in strata.items():
        print(f"\n[{name}] n={len(group)}")
        for r in group[:10]:  # 每档只展示 10 篇, 全量清单落 CSV 供人工精读
            print(f"  {r['pmcid']:<13} lines={int(r['n_lines']):<5} chars={int(r['n_chars']):<7} {r['title'][:60]}")
        if len(group) > 10:
            print(f"  ...(共 {len(group)} 篇, 见 {os.path.basename(OUT_STRATA)})")

    # ---- (5) 存档: 供报告与后续评估复用 ----
    imrad_map = {
        pmc: int("intro" in o and "methods" in o and "results" in o
                 and ("discussion" in o or "conclusion" in o))
        for pmc, o in order_all
    }
    with open(OUT_SECTIONS, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pmcid", "intro", "methods", "results", "discussion",
                    "conclusion", "references", "objective", "cjk_section",
                    "imrad_like", "n_headings"])
        for t in rows:
            w.writerow(list(t) + [imrad_map[t[0]], sum(t[1:])])

    with open(OUT_ACRONYMS, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["acronym", "count", "expansion_example"])
        for a, c in all_ac.most_common():
            w.writerow([a, c, all_ex.get(a, "")])

    with open(OUT_SYNONYMS, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["concept", "count"])
        for k, v in agg.most_common():
            w.writerow([k, v])

    with open(OUT_STRATA, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stratum", "pmcid", "n_lines", "n_chars", "title"])
        for name, group in strata.items():
            for r in group:
                w.writerow([name, r["pmcid"], r["n_lines"], r["n_chars"], r["title"]])

    print(f"\n[saved] {OUT_SECTIONS}")
    print(f"[saved] {OUT_ACRONYMS}  ({len(all_ac)} rows)")
    print(f"[saved] {OUT_SYNONYMS}")
    print(f"[saved] {OUT_STRATA}")


if __name__ == "__main__":
    main()
