# -*- coding: utf-8 -*-
"""查询理解与增强（检索系统开发 · 第一部分）
=====================================================================
输入：用户自然语言查询，如「二甲双胍对心血管疾病有何影响？」
输出：EnhancedQuery —— 清洗后的查询、识别出的医学实体、扩展的同义词、
      向量检索用查询、关键词检索用查询、元数据过滤条件、多版本查询。

为什么需要这一层（不做会怎样）:
  1 缩写    : 用户写 "MI"，语料里主要写 "myocardial infarction"。
              缩写不展开，向量检索会召回无关段落（"MI" 在英文里还可能是
              Michigan / milliliter）。
  2 中英    : 语料 27196 块 en / 153 块 zh。中文查询若不映射到英文术语，
              等于拿中文向量去检索英文库，跨语言检索质量显著下降。
  3 过滤    : "近五年" 是自然语言，Chroma 需要 {"pub_year": {"$gte": "2021"}}。
              不做提取就只能全库检索，无法回答时间限定的问题。
  4 多版本  : 单一查询只命中一个语义点；生成若干版本供后续多路召回 + 融合。

设计原则:
  - 词典分两层：curated（人工精选，中英对照，高质量）
                + corpus（从本项目语料挖出的缩略语，数据驱动，可查证）
    二者在输出里标注来源 provenance，便于判断置信度。
  - 指令前缀默认关闭：阶段④实测 bge-m3 加前缀反而更差（0.7422 → 0.6833），
    详见 docs/向量化与索引构建报告.md 第 4 节。开关保留，换模型时可重开。

用法:
    python scripts/query_understanding.py                      # 跑内置示例
    python scripts/query_understanding.py "二甲双胍对心血管疾病有何影响？"
    python scripts/query_understanding.py "MI mortality after 2020" --json
    python scripts/query_understanding.py "metformin" --instruction   # 强制加前缀
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACRONYM_CSV = os.path.join(ROOT, "data", "pmc_oa", "corpus_acronyms.csv")

# ---------------------------------------------------------------- 词典
# curated 同义词词典: key = canonical 术语(英文小写), value = 别名(含缩写与中文)
# 结构说明: 匹配时把 key 和所有 value 都视为同一概念的表层形式,
#           统一映射到 canonical, 因此双向都可查(MI -> 心肌梗死 -> mi)。
MEDICAL_SYNONYMS: Dict[str, List[str]] = {
    # 疾病 / 状态
    "myocardial infarction": ["mi", "ami", "heart attack", "心肌梗死", "心梗"],
    "heart failure": ["hf", "chf", "cardiac failure", "心力衰竭", "心衰"],
    "cardiovascular disease": ["cvd", "cardiovascular diseases", "心血管疾病"],
    "atrial fibrillation": ["af", "afib", "房颤", "心房颤动"],
    "stroke": ["cerebrovascular accident", "cva", "脑卒中", "中风"],
    "hypertension": ["htn", "high blood pressure", "高血压"],
    "diabetes mellitus": ["dm", "diabetes", "t2dm", "type 2 diabetes",
                          "type ii diabetes", "糖尿病", "2型糖尿病"],
    "chronic kidney disease": ["ckd", "renal insufficiency", "慢性肾病", "慢性肾脏病"],
    "chronic obstructive pulmonary disease": ["copd", "慢阻肺"],
    "covid-19": ["sars-cov-2", "coronavirus disease 2019", "新冠", "新型冠状病毒"],
    "sepsis": ["septicemia", "脓毒症", "败血症"],
    "pneumonia": ["lung infection", "肺炎"],
    "community-acquired pneumonia": ["cap", "社区获得性肺炎"],
    "coronary artery disease": ["cad", "coronary heart disease", "冠心病"],
    "acute kidney injury": ["aki", "急性肾损伤"],
    "asthma": ["哮喘"],
    "anemia": ["anaemia", "贫血"],
    "hyperlipidemia": ["dyslipidemia", "高脂血症"],
    "intensive care unit": ["icu", "重症监护室", "重症监护"],
    "obesity": ["obese", "肥胖", "肥胖症"],
    "cancer": ["malignancy", "tumor", "neoplasm", "癌症", "恶性肿瘤"],
    "dementia": ["cognitive decline", "痴呆", "认知障碍"],
    # 药物
    "metformin": ["二甲双胍", "glucophage"],
    "aspirin": ["asa", "acetylsalicylic acid", "阿司匹林"],
    "atorvastatin": ["立普妥", "lipitor"],
    "warfarin": ["coumadin", "华法林"],
    "insulin": ["胰岛素"],
    "statin": ["statins", "hmgr inhibitor", "他汀"],
    "sglt2 inhibitor": ["sglt2i", "empagliflozin", "dapagliflozin", "canagliflozin",
                        "sglt2抑制剂", "列净类"],
    "clopidogrel": ["plavix", "氯吡格雷"],
    "heparin": ["肝素"],
    "amoxicillin": ["阿莫西林"],
    "antibiotic": ["antibiotics", "抗菌药", "抗生素"],
    # 结局 / 指标
    "mortality": ["death", "fatal", "死亡", "死亡率"],
    "hospitalization": ["admission", "住院"],
    "bleeding": ["hemorrhage", "出血"],
    "blood pressure": ["bp", "血压"],
    "hba1c": ["glycated hemoglobin", " hemoglobin a1c", "糖化血红蛋白"],
    "egfr": ["glomerular filtration rate", "肾小球滤过率"],
    "myocardial infarction or stroke": ["mace", "major adverse cardiovascular event",
                                        "主要心血管事件"],
    # 人群 / 其他
    "randomized controlled trial": ["rct", "randomised controlled trial", "随机对照试验"],
    "meta-analysis": ["meta analysis", "systematic review", "荟萃分析", "系统综述"],
    "mendelian randomization": ["mr", "孟德尔随机化"],
}

# 中文术语 -> 英文 canonical（用于把中文查询桥接到英文语料）
# 语料 27196 en / 153 zh，中文查询必须翻译到英文术语才能检索到内容
ZH_TERM_MAP: Dict[str, str] = {
    "二甲双胍": "metformin",
    "阿司匹林": "aspirin",
    "华法林": "warfarin",
    "胰岛素": "insulin",
    "他汀": "statin",
    "抗生素": "antibiotic",
    "抗菌药": "antibiotic",
    "心肌梗死": "myocardial infarction",
    "心梗": "myocardial infarction",
    "心力衰竭": "heart failure",
    "心衰": "heart failure",
    "房颤": "atrial fibrillation",
    "心房颤动": "atrial fibrillation",
    "脑卒中": "stroke",
    "中风": "stroke",
    "高血压": "hypertension",
    "糖尿病": "diabetes mellitus",
    "2型糖尿病": "diabetes mellitus",
    "慢性肾病": "chronic kidney disease",
    "慢性肾脏病": "chronic kidney disease",
    "慢阻肺": "chronic obstructive pulmonary disease",
    "新冠": "covid-19",
    "新型冠状病毒": "covid-19",
    "脓毒症": "sepsis",
    "败血症": "sepsis",
    "肺炎": "pneumonia",
    "社区获得性肺炎": "community-acquired pneumonia",
    "冠心病": "coronary artery disease",
    "急性肾损伤": "acute kidney injury",
    "哮喘": "asthma",
    "贫血": "anemia",
    "高脂血症": "hyperlipidemia",
    "重症监护": "intensive care unit",
    "重症监护室": "intensive care unit",
    "肥胖": "obesity",
    "肥胖症": "obesity",
    "癌症": "cancer",
    "恶性肿瘤": "cancer",
    "痴呆": "dementia",
    "认知障碍": "dementia",
    "死亡率": "mortality",
    "死亡": "mortality",
    "住院": "hospitalization",
    "出血": "bleeding",
    "血压": "blood pressure",
    "糖化血红蛋白": "hba1c",
    "肾小球滤过率": "egfr",
    "随机对照试验": "randomized controlled trial",
    "荟萃分析": "meta-analysis",
    "系统综述": "meta-analysis",
    "孟德尔随机化": "mendelian randomization",
    "心血管疾病": "cardiovascular disease",
}

# 实体识别模式（英文/数字，大小写不敏感）
# \b 保证完整单词匹配：避免 "aspirin" 里的 "pirin" 被误命中
MEDICAL_PATTERNS: Dict[str, str] = {
    "drug": r"\b(aspirin|metformin|atorvastatin|warfarin|insulin|statins?|"
            r"clopidogrel|heparin|amoxicillin|antibiotics?|"
            r"empagliflozin|dapagliflozin|canagliflozin|sglt2\s?inhibitors?)\b",
    "condition": r"\b(myocardial infarction|heart failure|cardiac failure|"
                 r"atrial fibrillation|stroke|hypertension|diabetes mellitus|"
                 r"diabetes|chronic kidney disease|copd|"
                 r"chronic obstructive pulmonary disease|covid-19|sars-cov-2|"
                 r"sepsis|pneumonia|community-acquired pneumonia|"
                 r"coronary artery disease|acute kidney injury|asthma|anemia|"
                 r"hyperlipidemia|obesity|cancer|malignancy|dementia|"
                 r"cardiovascular disease)\b",
    "outcome": r"\b(mortality|death|hospitalization|admission|bleeding|"
               r"hemorrhage|blood pressure|hba1c|egfr|mace)\b",
    "study_type": r"\b(randomi[sz]ed controlled trial|meta-analysis|"
                  r"systematic review|mendelian randomi[sz]ation|"
                  r"cohort study|case-control|cross-sectional)\b",
}

# curated 缩写（大写敏感，避免小写 "mi"/"mr" 误命中普通英文词）
MEDICAL_ABBREVIATIONS: Dict[str, str] = {
    "MI": "myocardial infarction",
    "AMI": "myocardial infarction",
    "HF": "heart failure",
    "CHF": "heart failure",
    "CVD": "cardiovascular disease",
    "AF": "atrial fibrillation",
    "CVA": "stroke",
    "HTN": "hypertension",
    "DM": "diabetes mellitus",
    "T2DM": "diabetes mellitus",
    "CKD": "chronic kidney disease",
    "COPD": "chronic obstructive pulmonary disease",
    "BP": "blood pressure",
    "RCT": "randomized controlled trial",
    "MACE": "myocardial infarction or stroke",
    "SGLT2I": "sglt2 inhibitor",
    "eGFR": "egfr",
    "HbA1c": "hba1c",
}

# 语料缩略语的最低出现次数（低于此值的缩写多半是噪声或极窄领域用法）
CORPUS_ACRONYM_MIN_COUNT = 500
# expansion_example 的质量过滤：语料挖出来的展开常是截断片段（如 "s disease"）
_BAD_EXPANSION = re.compile(r"[\t\\]|^\W|\d")

STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "for", "to", "and", "or", "with",
    "is", "are", "was", "were", "be", "been", "does", "do", "did", "what",
    "how", "why", "when", "which", "who", "any", "there", "their", "its",
    "有什么", "有何", "影响", "是否", "什么", "如何", "的", "了", "吗", "请",
}


# ---------------------------------------------------------------- 数据类
@dataclass
class EnhancedQuery:
    """查询理解的结果。字段设计对应下游检索的每一个入口。"""

    original: str                                # 用户原始输入
    cleaned: str                                 # 清洗后
    lang: str                                    # zh / en / mixed
    entities: Dict[str, List[str]] = field(default_factory=dict)   # 类型 -> 实体
    canonical_terms: List[str] = field(default_factory=list)       # 规范化术语
    synonyms: Dict[str, List[str]] = field(default_factory=dict)   # 术语 -> 同义词
    expansions: Dict[str, str] = field(default_factory=dict)       # 表层形式 -> 展开
    vector_query: str = ""                       # 送嵌入模型的文本
    keyword_query: str = ""                      # 送关键词检索(BM25)的文本
    filters: Dict = field(default_factory=dict)  # Chroma where 过滤
    filter_notes: List[str] = field(default_factory=list)          # 过滤的可读解释
    variants: List[str] = field(default_factory=list)              # 多版本查询
    provenance: Dict[str, str] = field(default_factory=dict)       # 术语来源

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        lines = [
            f"原始查询   : {self.original}",
            f"清洗后     : {self.cleaned}",
            f"语言       : {self.lang}",
            f"医学实体   : {self.entities or '{}'}",
            f"规范术语   : {self.canonical_terms or '[]'}",
            f"缩写展开   : {self.expansions or '{}'}",
            f"同义扩展   : {self.synonyms or '{}'}",
            f"向量查询   : {self.vector_query}",
            f"关键词查询 : {self.keyword_query}",
            f"过滤条件   : {self.filters or '{}'}",
        ]
        if self.filter_notes:
            lines.append(f"过滤释义   : {'; '.join(self.filter_notes)}")
        lines.append(f"查询变体   : {len(self.variants)} 个")
        for i, v in enumerate(self.variants):
            lines.append(f"   [{i}] {v}")
        return "\n".join(lines)


# ---------------------------------------------------------------- 语料缩略语
_corpus_acronyms: Optional[Dict[str, str]] = None


def load_corpus_acronyms(min_count: Optional[int] = None) -> Dict[str, str]:
    """从本项目语料挖出的缩略语表构建补充词典（惰性加载 + 缓存）。

    为什么用它:  curated 词典只能覆盖常见术语, 而这份表有 43 334 条真实
    出现在语料里的缩写(带频次)。用它做扩展是"数据驱动"的——扩展出来的词
    确实在本库里出现过, 而不是凭空猜测。

    为什么必须过滤: 表里有大量噪声, 例如 AD -> "s disease"(截断)、
    BP -> "Biological Process"(释义错误)。所以既卡频次, 也卡展开文本质量。
    """
    global _corpus_acronyms
    if _corpus_acronyms is not None:
        return _corpus_acronyms
    # 默认参数在定义时绑定, 这里运行时读取, 使 --min-count 能生效
    if min_count is None:
        min_count = CORPUS_ACRONYM_MIN_COUNT
    out: Dict[str, str] = {}
    try:
        import pandas as pd
        df = pd.read_csv(ACRONYM_CSV)
    except Exception:                                        # noqa: BLE001
        _corpus_acronyms = {}
        return out
    for _, r in df.iterrows():
        try:
            if int(r["count"]) < min_count:
                continue
        except (TypeError, ValueError):
            continue
        abbr = str(r["acronym"]).strip()
        exp = str(r.get("expansion_example", "")).strip()
        # 质量闸门: 长度合理、非 NaN、无制表符、首词不是单字母残留
        if not abbr or not exp or exp.lower() == "nan":
            continue
        if _BAD_EXPANSION.search(exp) or not (5 <= len(exp) <= 40):
            continue
        if len(exp.split()[0]) < 3:
            continue
        out[abbr] = exp.lower()
    _corpus_acronyms = out
    return out


# ---------------------------------------------------------------- 基础处理
_CJK = re.compile(r"[\u4e00-\u9fff]")
_WS = re.compile(r"\s+")
# 全角转半角（中文查询常见）
_FULL2HALF = {i: i - 0xFEE0 for i in range(0xFF01, 0xFF5F)}
_FULL2HALF[0x3000] = 0x20


def clean_query(text: str) -> str:
    """基础清洗：全角转半角、压缩空白、去首尾空白。

    不去标点: 实测去掉句尾 "?" 会让 top1 相似度从 0.7422 降到 0.7273 ——
    疑问标点本身携带"这是一个问句"的语义，是检索信号而不是噪声。
    （见 scripts/bench_query_enhancement.py 的对照实验）

    不做的事: 不去停用词、不改大小写原样(大写信息对缩写识别有用)。
    """
    if not text:
        return ""
    t = text.translate(_FULL2HALF)
    return _WS.sub(" ", t).strip()


def detect_lang(text: str) -> str:
    """按 CJK 字符占比判定语言（与阶段③切块时的判定口径一致）。"""
    if not text:
        return "en"
    n = len(text)
    cjk = len(_CJK.findall(text))
    ratio = cjk / n
    if ratio > 0.5:
        return "zh"
    if ratio > 0.05:
        return "mixed"
    return "en"


# ---------------------------------------------------------------- 实体与扩展
def _build_alias_index() -> Dict[str, str]:
    """把 curated 词典摊平: 任意表层形式 -> canonical。"""
    idx: Dict[str, str] = {}
    for canon, aliases in MEDICAL_SYNONYMS.items():
        idx[canon.lower()] = canon
        for a in aliases:
            idx[a.strip().lower()] = canon
    for zh, canon in ZH_TERM_MAP.items():
        idx[zh.lower()] = canon
    return idx


_ALIAS_INDEX = _build_alias_index()


def extract_entities(text: str) -> Dict[str, List[str]]:
    """识别医学实体。英文走正则（\b 保证完整词匹配），缩写走大写敏感表，
    中文走词典子串匹配。"""
    ents: Dict[str, List[str]] = {}
    low = text.lower()

    for etype, pat in MEDICAL_PATTERNS.items():
        hits = [m.group(0).lower() for m in re.finditer(pat, low)]
        if hits:
            ents[etype] = sorted(set(hits))

    # 缩写：大小写敏感，只认全大写（或 eGFR/HbA1c 这类混合大小写）
    abbr_hits = []
    for ab in MEDICAL_ABBREVIATIONS:
        if re.search(r"(?<![A-Za-z0-9])" + re.escape(ab) + r"(?![A-Za-z0-9])", text):
            abbr_hits.append(ab)
    if abbr_hits:
        ents["abbreviation"] = sorted(set(abbr_hits))

    # 中文术语子串匹配
    zh_hits = [zh for zh in ZH_TERM_MAP if zh in text]
    if zh_hits:
        ents["zh_term"] = sorted(set(zh_hits))

    return ents


def expand_synonyms(entities: Dict[str, List[str]],
                    text: str) -> (Dict[str, List[str]], Dict[str, str], Dict[str, str]):
    """同义词扩展 + 缩写展开。

    Returns:
        (synonyms, expansions, provenance)
        synonyms   : canonical -> 同义词列表（供召回扩面）
        expansions : 表层形式 -> 展开后的规范形式（缩写 MI -> myocardial infarction）
        provenance : canonical -> curated / corpus
    """
    synonyms: Dict[str, List[str]] = {}
    expansions: Dict[str, str] = {}
    provenance: Dict[str, str] = {}

    # 1) curated：实体命中 -> canonical -> 同义词
    canons: List[str] = []
    for etype, terms in entities.items():
        for t in terms:
            if etype == "abbreviation":
                canon = MEDICAL_ABBREVIATIONS.get(t)
                if canon:
                    expansions[t] = canon
                    canons.append(canon)
                continue
            if etype == "zh_term":
                canon = ZH_TERM_MAP.get(t)
                if canon:
                    expansions[t] = canon
                    canons.append(canon)
                continue
            canon = _ALIAS_INDEX.get(t.lower())
            if canon:
                canons.append(canon)

    for c in sorted(set(canons)):
        syns = [s for s in MEDICAL_SYNONYMS.get(c, []) if s.lower() != c.lower()]
        if syns:
            synonyms[c] = syns
        provenance[c] = "curated"

    # 2) corpus：查询里出现的全大写 token 若在语料缩略语表中则补展开
    corpus = load_corpus_acronyms()
    for tok in set(re.findall(r"\b[A-Z][A-Z0-9]{1,7}\b", text)):
        if tok in expansions or tok in MEDICAL_ABBREVIATIONS:
            continue
        exp = corpus.get(tok)
        if exp:
            expansions[tok] = exp
            provenance.setdefault(exp, "corpus")

    return synonyms, expansions, provenance


# ---------------------------------------------------------------- 过滤提取
_YEAR = r"(19[89]\d|20[0-2]\d)"
_ZH_NUM = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
           "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}

# 时间表达：按"信息量从高到低"排列，先匹配最具体的
_TIME_PATTERNS = [
    # 2020-2023 / 2020 至 2023
    (re.compile(rf"({_YEAR})\s*(?:-|–|—|~|至|到)\s*({_YEAR})"), "range"),
    # 2020年以后 / since 2020 / after 2020
    (re.compile(rf"({_YEAR})\s*(?:年)?\s*(?:以后|之后|以来|起|至今)"), "gte_zh"),
    (re.compile(rf"(?:since|after|from)\s+({_YEAR})", re.I), "gte_en"),
    # 近5年 / 近三年 / recent years
    (re.compile(r"(?:近|最近|过去)\s*([0-9]{1,2}|[一二两三四五六七八九十])\s*年"), "recent_zh"),
    (re.compile(r"(?:last|past|recent)\s+([0-9]{1,2})\s+years?", re.I), "recent_en"),
    # 2019年之前 / before 2019
    (re.compile(rf"({_YEAR})\s*(?:年)?\s*(?:以前|之前)"), "lte_zh"),
    (re.compile(rf"(?:before|until|prior to)\s+({_YEAR})", re.I), "lte_en"),
    # 单年份 2023
    (re.compile(rf"({_YEAR})\s*年?"), "single"),
]


# 语料 pub_year 的实际取值范围（阶段③统计值），用于把范围条件展开成枚举
MIN_YEAR, MAX_YEAR = "1992", "2026"


def _year_list(y0: str, y1: str) -> List[str]:
    """把年份区间展开成显式列表。

    为什么用 $in 枚举而不是 $gte/$lte:
      pub_year 在 Chroma 里存的是字符串（阶段③就是 str），而 Chroma 对
      字符串类型的元数据不支持 $gte/$lte 比较，会报
      "Expected operand value to be ..."。$in 对字符串完全可用，
      且年份取值只有 35 个，展开成本可忽略。
    """
    return [str(y) for y in range(int(y0), int(y1) + 1)]


def extract_filters(text: str, lang: str = "en") -> (Dict, List[str]):
    """从自然语言里提取元数据过滤条件。

    只产出阶段④已验证可用的三种过滤: pub_year / is_table / lang。
    （journal 有 287 种取值，需另建映射表，本部分不处理。）
    """
    filters: Dict = {}
    notes: List[str] = []
    this_year = time.localtime().tm_year

    for pat, kind in _TIME_PATTERNS:
        m = pat.search(text)
        if not m:
            continue
        if kind == "range":
            y0, y1 = sorted([m.group(1), m.group(2)])
            filters["pub_year"] = {"$in": _year_list(y0, y1)}
            notes.append(f"发表年份 {y0}-{y1}")
        elif kind in ("gte_zh", "gte_en"):
            y = m.group(1)
            filters["pub_year"] = {"$in": _year_list(y, MAX_YEAR)}
            notes.append(f"发表年份 ≥ {y}")
        elif kind in ("recent_zh", "recent_en"):
            raw = m.group(1)
            n = _ZH_NUM.get(raw, None)
            if n is None:
                try:
                    n = int(raw)
                except ValueError:
                    continue
            y = str(this_year - n + 1)
            filters["pub_year"] = {"$in": _year_list(y, MAX_YEAR)}
            notes.append(f"近 {n} 年 → 发表年份 ≥ {y}")
        elif kind in ("lte_zh", "lte_en"):
            y = m.group(1)
            filters["pub_year"] = {"$in": _year_list(MIN_YEAR, y)}
            notes.append(f"发表年份 ≤ {y}")
        elif kind == "single":
            y = m.group(1)
            filters["pub_year"] = y
            notes.append(f"发表年份 = {y}")
        break                                   # 只取最具体的一条时间约束

    if re.search(r"\b(table|tables)\b|表格", text, re.I):
        filters["is_table"] = True
        notes.append("仅含表格的块")
    if re.search(r"中文|chinese", text, re.I):
        filters["lang"] = "zh"
        notes.append("仅中文块（注意：全库仅 153 块中文）")

    return filters, notes


# ---------------------------------------------------------------- 查询构造
def build_vector_query(cleaned: str, expansions: Dict[str, str],
                       synonyms: Dict[str, List[str]],
                       lang: str = "en",
                       canonical_terms: Optional[List[str]] = None,
                       use_instruction: bool = False,
                       instruction: str = "") -> str:
    """构造送嵌入模型的查询文本。

    只做"缩写展开"这一件事，不做同义词堆砌:
      展开缩写是纯增益(MI -> myocardial infarction，两者同义，不引入新语义)；
      把一堆同义词拼进查询会稀释语义重心，反而降低向量检索精度。
      同义词放到 variants 里供多路召回用。

    中英文分两种策略:
      en     : 原地展开缩写，保留原形式（MI -> myocardial infarction (MI)）
      zh/mix : 原句原样保留，**不**追加英文术语。
               两种"桥接"做法都试过并做了消融（bench_query_enhancement.py
               --ablation，4 条中文查询）：
                 A 原句          0.5953 / 0.5509 / 0.6343 / 0.6513
                 B 原句+英文术语 0.6908 / 0.4815 / 0.6810 / 0.7082
                 C 纯英文术语串  0.6943 / 0.4990 / 0.7010 / 0.7032
               B 在"抗生素…"这条上比 A、C 都低约 0.05，说明拼接会稀释语义；
               而 max(A, C) 在 4 条里 3 条优于 B。
               所以不做拼接，改为把 C 作为独立变体（见 build_variants），
               交给下游多路召回融合——每次都能拿到两路里更好的那个。

    替换必须一次性完成: 逐个 re.sub 会嵌套（"死亡率"→"mortality (死亡率)"
    之后再替换 "死亡"，得到 mortality (mortality (死亡)率)）。

    指令前缀默认关闭: 阶段④实测 bge-m3 加前缀后 top1 相似度
    由 0.7422 降到 0.6833（docs/向量化与索引构建报告.md 第 4 节）。
    """
    q = cleaned
    if lang == "en":
        surfaces = [s for s in expansions if s.lower() != expansions[s].lower()]
        if surfaces:
            pat = re.compile("|".join(re.escape(s) for s in
                                      sorted(surfaces, key=len, reverse=True)))
            q = pat.sub(lambda m: f"{expansions[m.group(0)]} ({m.group(0)})", q)
    # else: zh/mixed 保持原句，桥接交给 build_variants 生成的纯英文术语变体
    if use_instruction:
        prefix = instruction or ("Represent this sentence for searching "
                                 "relevant passages: ")
        q = prefix + q
    return _WS.sub(" ", q).strip()


def build_keyword_query(cleaned: str, lang: str,
                        entities: Dict[str, List[str]],
                        synonyms: Dict[str, List[str]]) -> str:
    """构造关键词检索（BM25）用的词串。

    中文查询要输出英文关键词：语料 99.4% 是英文块，中文词做 BM25
    几乎匹配不到任何东西。所以先映射到英文 canonical，再并入同义词。
    """
    toks: List[str] = []

    def add(w: str):
        w = w.strip().lower()
        if not w or w in STOPWORDS or len(w) < 2:
            return
        if w not in toks:
            toks.append(w)

    # 实体优先（信号最强）
    for terms in entities.values():
        for t in terms:
            canon = None
            if t in MEDICAL_ABBREVIATIONS:
                canon = MEDICAL_ABBREVIATIONS[t]
            elif t in ZH_TERM_MAP:
                canon = ZH_TERM_MAP[t]
            else:
                canon = _ALIAS_INDEX.get(t.lower(), t)
            for piece in (canon or t).split():
                add(piece)
    # 同义词补面
    for canon, syns in synonyms.items():
        for s in syns:
            if _CJK.search(s):          # 中文同义词对英文 BM25 无益
                continue
            for piece in s.split():
                add(piece)
    # 中文查询：补英文 canonical（若无任何英文词则保留原文，供中文 BM25）
    if lang in ("zh", "mixed") and not toks:
        for piece in re.findall(r"[A-Za-z]{2,}", cleaned):
            add(piece)
        if not toks:
            return cleaned
    return " ".join(toks)


def build_variants(cleaned: str, vector_query: str,
                   canonical_terms: List[str],
                   synonyms: Dict[str, List[str]],
                   lang: str = "en",
                   max_variants: int = 3) -> List[str]:
    """生成多版本查询，供后续多路召回 + 结果融合。

    变体 0 = 主查询（缩写展开 / 跨语言桥接后的版本）
    变体 1 = 纯英文术语串（中文查询时这是关键的跨语言一路）
    变体 2 = 用同义词重述的版本（换一种说法命中不同的语义点）
    """
    variants = [vector_query]
    en_terms = [c for c in canonical_terms if not _CJK.search(c)]

    if lang != "en" and en_terms:
        v = " ".join(en_terms)
        if v and v not in variants:
            variants.append(v)

    # 只取前几个有同义词的核心术语，避免组合爆炸
    for canon in canonical_terms:
        if len(variants) >= max_variants:
            break
        syns = [s for s in synonyms.get(canon, []) if not _CJK.search(s)]
        if not syns:
            continue
        v = cleaned
        for s in syns[:1]:
            v = re.sub(re.escape(canon), s, v, count=1, flags=re.I)
        v = _WS.sub(" ", v).strip()
        # 只在确实发生改变时才算一个变体
        if v and v != cleaned and v not in variants:
            variants.append(v)
    return variants[:max_variants]


# ---------------------------------------------------------------- 主入口
def enhance_query(query: str, use_instruction: bool = False,
                  instruction: str = "",
                  max_variants: int = 3) -> EnhancedQuery:
    """查询理解与增强的主入口：清洗 → 实体 → 扩展 → 生成各检索入口的查询。"""
    original = query or ""
    cleaned = clean_query(original)
    lang = detect_lang(cleaned)
    entities = extract_entities(cleaned)
    synonyms, expansions, provenance = expand_synonyms(entities, cleaned)
    filters, notes = extract_filters(cleaned, lang)

    canonical_terms = sorted(set(provenance.keys()))

    vector_q = build_vector_query(cleaned, expansions, synonyms, lang,
                                  canonical_terms, use_instruction, instruction)
    keyword_q = build_keyword_query(cleaned, lang, entities, synonyms)
    variants = build_variants(cleaned, vector_q, canonical_terms, synonyms,
                              lang, max_variants)

    return EnhancedQuery(
        original=original,
        cleaned=cleaned,
        lang=lang,
        entities=entities,
        canonical_terms=canonical_terms,
        synonyms=synonyms,
        expansions=expansions,
        vector_query=vector_q,
        keyword_query=keyword_q,
        filters=filters,
        filter_notes=notes,
        variants=variants,
        provenance=provenance,
    )


# ---------------------------------------------------------------- CLI
DEMO_QUERIES = [
    "二甲双胍对心血管疾病有何影响？",
    "Does metformin reduce cardiovascular mortality in type 2 diabetes?",
    "MI 患者的死亡率 2020年以后的研究",
    "SGLT2 inhibitor heart failure hospitalization",
    "近五年关于慢性肾病eGFR的随机对照试验",
    "抗生素治疗社区获得性肺炎的疗程",
]


def main():
    global CORPUS_ACRONYM_MIN_COUNT       # 必须在任何引用之前声明

    ap = argparse.ArgumentParser(description="查询理解与增强")
    ap.add_argument("query", nargs="?", default=None, help="待处理的查询")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    ap.add_argument("--instruction", action="store_true",
                    help="给向量查询加 BGE 指令前缀（实测 bge-m3 下更差，默认关闭）")
    ap.add_argument("--min-count", type=int, default=CORPUS_ACRONYM_MIN_COUNT,
                    help="语料缩略语的最低频次")
    args = ap.parse_args()

    CORPUS_ACRONYM_MIN_COUNT = args.min_count

    queries = [args.query] if args.query else DEMO_QUERIES
    for i, q in enumerate(queries):
        eq = enhance_query(q, use_instruction=args.instruction)
        if args.json:
            print(json.dumps(eq.to_dict(), ensure_ascii=False, indent=2))
        else:
            print("=" * 72)
            print(eq.summary())
            print()


if __name__ == "__main__":
    main()
