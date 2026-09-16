# -*- coding: utf-8 -*-
"""
语料扩容驱动器 (容量评估 → 落地)
=================================
在 fetch_pmc_oa.py 的单篇能力之上补齐三件事:
  1) 多主题: 按若干主题词分配配额, 避免语料只覆盖单一疾病
  2) 去重  : 跳过本机已存在的 PMCID(单脚本内 + 跨批次都去重)
  3) 并发  : 单篇耗时被 S3 网络往返主导(实测 3.75 s/篇里 3.45 s 是网络),
             串行拉取 1000 篇要 62 分钟; 并发后可压到 1/4 左右。

并发安全性:
  - 每篇写进自己独立的 <ROOT>/PMC<id>.<ver>/ 目录, 线程间无文件冲突;
  - manifest 行由主线程统一写盘, 不做并发写;
  - 并发度默认 4, 配合 --sleep 使总请求速率 < 3 req/s(NCBI 无 key 的软限制)。

用法:
    python scripts/expand_corpus.py --target 1000
    python scripts/expand_corpus.py --target 300 --workers 4 --dry-run
    python scripts/expand_corpus.py --queries "hypertension:200" "lung cancer:150"
"""
import argparse
import csv
import glob
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fetch_pmc_oa import (  # noqa: E402
    MANIFEST_HEADER,
    esearch_pmc,
    http_get,
    is_oa_comm,
    list_version_prefixes,
    md5_of,
    pick_latest_version,
    to_https_url,
)

ROOT_DEFAULT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "data", "pmc_oa")
# 默认主题配额: 与本语料已有 30 篇的疾病谱一致(糖/高压/心梗/痴呆/肺癌/肾病)
DEFAULT_QUERIES = [
    ("diabetes mellitus", 200),
    ("hypertension", 200),
    ("myocardial infarction", 150),
    ("Alzheimer disease", 150),
    ("lung cancer", 150),
    ("chronic kidney disease", 150),
]


def existing_pmcids(root: str) -> set[str]:
    """本机已有哪些 PMCID: 看子目录名 + 已有 manifest。"""
    have = set()
    for d in glob.glob(os.path.join(root, "*")):
        if os.path.isdir(d):
            m = re.match(r"(PMC\d+)\.\d+$", os.path.basename(d))
            if m:
                have.add(m.group(1))
    for mf in ("manifest.csv", "manifest_expand.csv"):
        p = os.path.join(root, mf)
        if os.path.exists(p):
            with open(p, encoding="utf-8-sig", newline="") as fh:
                for r in csv.DictReader(fh):
                    pid = (r.get("pmcid") or "").strip().upper()
                    if pid:
                        have.add(pid if pid.startswith("PMC") else "PMC" + pid)
    return have


def fetch_one(pmcid: str, out_root: str, with_xml: bool = False):
    """拉取单篇, 返回 (状态, manifest行 或 跳过原因)。"""
    try:
        prefixes = list_version_prefixes(pmcid)
        prefix = pick_latest_version(pmcid, prefixes)
        if not prefix:
            return "skip", pmcid, "S3 无版本前缀"
        ver = prefix.rstrip("/")
        meta_url = f"https://pmc-oa-opendata.s3.amazonaws.com/metadata/{ver}.json"
        meta = __import__("json").loads(http_get(meta_url).decode("utf-8"))
        if not is_oa_comm(meta):
            return "skip", pmcid, (
                f"非 oa_comm(license={meta.get('license_code')}, "
                f"ms={meta.get('is_manuscript')}, oa={meta.get('is_pmc_openaccess')})")

        local_dir = os.path.join(out_root, ver)
        os.makedirs(local_dir, exist_ok=True)
        objects = []

        text_url = to_https_url(meta.get("text_url") or "")
        if text_url:
            dest = os.path.join(local_dir, f"{ver}.txt")
            data = http_get(text_url)
            with open(dest, "wb") as fh:
                fh.write(data)
            objects.append(f"txt({len(data)}B)")

        meta_dest = os.path.join(local_dir, f"{ver}.json")
        with open(meta_dest, "w", encoding="utf-8") as fh:
            __import__("json").dump(meta, fh, ensure_ascii=False, indent=2)
        objects.append(f"json({os.path.getsize(meta_dest)}B)")

        row = [
            datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            meta.get("pmcid", pmcid), meta.get("version"), meta.get("pmid"),
            (meta.get("title") or "")[:200], meta.get("license_code"),
            meta.get("is_manuscript"), meta.get("is_pmc_openaccess"),
            (meta.get("citation") or "")[:150], local_dir, ";".join(objects),
        ]
        return "ok", pmcid, row
    except Exception as exc:  # noqa: BLE001
        return "fail", pmcid, f"{type(exc).__name__}: {str(exc)[:120]}"


def main():
    ap = argparse.ArgumentParser(description="PMC OA 语料多主题并发扩容(可续跑/去重)")
    ap.add_argument("--target", type=int, default=1000, help="目标新增篇数")
    ap.add_argument("--queries", nargs="*", default=None,
                    help='主题与配额, 形如 "hypertension:200"; 省略则用默认 6 主题')
    ap.add_argument("--workers", type=int, default=4, help="并发度(默认4)")
    ap.add_argument("--out", default=ROOT_DEFAULT, help="语料根目录")
    ap.add_argument("--sleep", type=float, default=0.3, help="同一 worker 内请求间隔(秒)")
    ap.add_argument("--dry-run", action="store_true", help="只规划不下载")
    args = ap.parse_args()

    out_root = os.path.abspath(args.out)
    os.makedirs(out_root, exist_ok=True)

    if args.queries:
        plan = []
        for q in args.queries:
            if ":" in q:
                name, n = q.rsplit(":", 1)
                plan.append((name, int(n)))
            else:
                plan.append((q, 0))
    else:
        plan = DEFAULT_QUERIES

    have = existing_pmcids(out_root)
    print(f"已有 {len(have)} 篇 -> 目标新增 {args.target} 篇, 并发={args.workers}")

    # 1) 按配额检索候选 ID(已去重)
    todo: list[str] = []
    seen: set[str] = set()
    for name, quota in plan:
        if quota <= 0 or len(todo) >= args.target:
            continue
        need = min(quota, args.target - len(todo))
        ids = esearch_pmc(name, retmax=max(50, need * 3))
        added = 0
        for pid in ids:
            pmcid = pid.upper() if pid.upper().startswith("PMC") else "PMC" + pid
            if pmcid in have or pmcid in seen:
                continue
            seen.add(pmcid)
            todo.append(pmcid)
            added += 1
            if added >= need:
                break
        print(f"  [{name}] 配额{quota} -> 取到 {added} 篇候选")
        time.sleep(args.sleep)

    print(f"候选合计 {len(todo)} 篇(已排除本机已有的 {len(have)} 篇)")
    if args.dry_run:
        print("--dry-run: 不下盘")
        return
    if not todo:
        print("无新增候选")
        return

    # 2) 并发拉取
    rows, n_ok, n_skip, n_fail = [], 0, 0, 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_one, p, out_root): p for p in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            status, pmcid, payload = fut.result()
            if status == "ok":
                n_ok += 1
                rows.append(payload)
            else:
                if status == "skip":
                    n_skip += 1
                else:
                    n_fail += 1
                    print(f"  [fail] {pmcid}: {payload}")
            if i % 25 == 0 or i == len(todo):
                el = time.time() - t0
                rate = n_ok / el * 60 if el else 0
                print(f"  进度 {i}/{len(todo)}  ok={n_ok} skip={n_skip} fail={n_fail} "
                      f"| {el/60:.1f} min | {rate:.1f} 篇/分")

    # 3) 写清单(主线程单写, 避免并发写坏文件)
    if rows:
        mf = os.path.join(out_root, "manifest_expand.csv")
        new = not os.path.exists(mf)
        with open(mf, "a", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(MANIFEST_HEADER)
            w.writerows(rows)
        print(f"\n[清单] {mf} (+{len(rows)} 行)")

    el = (time.time() - t0) / 60
    print(f"完成: 新增 {n_ok} 篇, 跳过 {n_skip}, 失败 {n_fail} | 耗时 {el:.1f} min")


if __name__ == "__main__":
    main()
