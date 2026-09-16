# -*- coding: utf-8 -*-
"""
PMC OA (PubMed Central Open Access) 数据源拉取工具 —— 环境准备阶段验证用。

背景（2026-08 起 PMC 数据重构）:
    旧的 oa_comm/ 目录与 oa_file_list.csv 已下线。新版按“文章版本”存放于 S3 桶
    pmc-oa-opendata (us-east-1, 匿名可读)，每个版本前缀 PMC<id>.<ver>/ 内含:
        <ver>.xml  (JATS), <ver>.txt (纯文本), <ver>.json (元数据, 含 license_code),
        以及许可允许时的 pdf / 图片 / 补充文件。
    元数据字段: is_pmc_openaccess / is_manuscript / is_historical_ocr /
                is_retracted / license_code / xml_url / text_url ...

本工具将“旧 oa_comm 子集”精确等价为:
    is_pmc_openaccess == true 且 is_manuscript == false 且
    license_code in {CC0, CC BY, CC BY-SA, CC BY-ND}
流程:
    1) eSearch (db=pmc) 限定 商业复用类 CC 许可 + OA 过滤 检索主题
    2) 对每个 PMCID 用 S3 ListObjectsV2 解析出可用版本前缀
    3) 拉取该版本元数据 JSON，校验是否符合上述 oa_comm 条件
    4) 通过元数据中的 text_url/xml_url(自带 md5 校验参) 下载文本/XML，校验 MD5
    5) 记录本地清单 manifest.csv

用法示例:
    python fetch_pmc_oa.py --query "diabetes mellitus" --max 5
    python fetch_pmc_oa.py --query "hypertension" --max 10 --with-xml
    python fetch_pmc_oa.py --pmcid 13901 13902        # 直接按 PMCID 拉取
"""
import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

S3_BUCKET_URL = "https://pmc-oa-opendata.s3.amazonaws.com"
EUTILS_URL = "https://eutils.ncbi.nlm.nih.gov/eutils/esearch.fcgi"
# 旧 oa_comm = 允许商业复用的开放许可（不含 CC BY-NC*，不含作者手稿 TDM）
COMMERCIAL_LICENSES = {"CC0", "CC BY", "CC BY-SA", "CC BY-ND"}
# eSearch 侧对应的许可过滤（与 COMMERCIAL_LICENSES 一一对应）
LICENSE_FILTER_TERM = (
    "(cc0_license[Filter] OR cc_by_license[Filter] OR "
    "cc_by-sa_license[Filter] OR cc_by-nd_license[Filter])"
)
HEADERS = {"User-Agent": "medical-rag-prep/0.1 (educational use)"}
MANIFEST_HEADER = [
    "fetched_at_utc", "pmcid", "version", "pmid", "title",
    "license_code", "is_manuscript", "is_openaccess",
    "journal", "local_dir", "objects",
]


def http_get(url: str, timeout: int = 90, retries: int = 3) -> bytes:
    """HTTP GET。优先用 curl.exe（本机实测 urllib 的 TLS 到 AWS S3 会被网络随机重置）。
    找不到 curl 时退回 urllib。"""
    last_err = None
    curl = shutil.which("curl.exe") or shutil.which("curl")
    for attempt in range(retries):
        try:
            if curl:
                proc = subprocess.run(
                    [
                        curl, "-sS", "-L", "--max-time", str(timeout),
                        "-H", "User-Agent: medical-rag-prep/0.1 (educational use)",
                        url,
                    ],
                    capture_output=True,
                )
                if proc.returncode == 0:
                    return proc.stdout
                last_err = RuntimeError(
                    f"curl rc={proc.returncode}: "
                    f"{proc.stderr.decode('utf-8', 'ignore')[:300]}"
                )
            else:
                req = urllib.request.Request(url, headers=HEADERS)
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    return resp.read()
        except Exception as exc:  # noqa: BLE001
            last_err = exc
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GET failed after {retries} retries: {url} -> {last_err}")


def md5_of(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def to_https_url(url: str) -> str:
    """把元数据里的 s3://bucket/key 转成 https 可下载 URL（桶在 us-east-1）。"""
    if url.startswith("s3://"):
        rest = url[len("s3://"):]
        bucket, _, key = rest.partition("/")
        return f"https://{bucket}.s3.amazonaws.com/{key}"
    return url


def esearch_pmc(
    query: str,
    retmax: int,
    mindate: str | None = None,
    maxdate: str | None = None,
    datetype: str = "pdat",
) -> list[str]:
    """按主题检索返回 PMC ID 列表（已附加商业复用 OA 许可过滤）。
    mindate/maxdate 形如 '2022/01/01'，按 datetype(默认 pdat=出版日期) 过滤。"""
    # open_access[Filter] = PMC Open Access Subset（不含作者手稿），配合商业许可过滤复刻旧 oa_comm
    term = f"({query}) AND {LICENSE_FILTER_TERM} AND open_access[Filter]"
    params = urllib.parse.urlencode(
        {"db": "pmc", "term": term, "retmax": retmax, "retmode": "json", "sort": "relevance"}
    )
    if mindate or maxdate:
        params += "&" + urllib.parse.urlencode(
            {"mindate": mindate or "1900/01/01", "maxdate": maxdate or "3000/12/31", "datetype": datetype}
        )
    data = http_get(f"{EUTILS_URL}?{params}")
    payload = json.loads(data.decode("utf-8"))
    result = payload.get("esearchresult", {})
    count = result.get("count", "0")
    ids = result.get("idlist", [])
    print(f"[esearch] hit count={count}, retmax={len(ids)}")
    return ids


def list_version_prefixes(pmcid: str) -> list[str]:
    """列出某个 PMCID 在 S3 桶中的版本前缀，如 ['PMC12810641.1/']。
    入参允许带/不带 'PMC' 前缀，内部统一归一化。"""
    num = pmcid[3:] if pmcid.upper().startswith("PMC") else pmcid
    url = f"{S3_BUCKET_URL}/?list-type=2&prefix=PMC{num}.&delimiter=/&max-keys=100"
    xml = http_get(url).decode("utf-8", errors="ignore")
    prefixes = re.findall(r"<Prefix>(PMC\d+\.\d+/)</Prefix>", xml)
    return prefixes


def pick_latest_version(pmcid: str, prefixes: list[str]) -> str | None:
    if not prefixes:
        return None
    def ver_of(p: str) -> int:
        m = re.search(r"\.(\d+)/$", p)
        return int(m.group(1)) if m else 0
    return max(prefixes, key=ver_of)


def is_oa_comm(meta: dict) -> bool:
    return (
        meta.get("is_pmc_openaccess") is True
        and meta.get("is_manuscript") is False
        and meta.get("license_code") in COMMERCIAL_LICENSES
    )


def download_and_verify(url: str, dest: str) -> tuple[int, bool]:
    """下载并校验 md5 参数，返回 (bytes_len, md5_ok)。URL 带 ?md5=... 参数。"""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    parsed = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qs(parsed.query)
    expected_md5 = query.get("md5", [None])[0]
    data = http_get(url)
    ok = (expected_md5 is None) or (md5_of(data) == expected_md5.lower())
    with open(dest, "wb") as fh:
        fh.write(data)
    return len(data), ok


def fetch_pmcid(
    pmcid: str,
    out_root: str,
    writer: csv.writer,
    with_xml: bool = False,
) -> str:
    """拉取单个 PMCID 的最新版本并返回状态描述。"""
    prefixes = list_version_prefixes(pmcid)
    prefix = pick_latest_version(pmcid, prefixes)
    if not prefix:
        return f"[skip] {pmcid} : S3 无可用版本前缀"
    ver_name = prefix.rstrip("/")  # e.g. PMC12810641.1

    meta_url = f"{S3_BUCKET_URL}/metadata/{ver_name}.json"
    meta = json.loads(http_get(meta_url).decode("utf-8"))

    if not is_oa_comm(meta):
        lic = meta.get("license_code")
        ms = meta.get("is_manuscript")
        oa = meta.get("is_pmc_openaccess")
        return (
            f"[skip] {ver_name} : 不符合 oa_comm(license={lic}, "
            f"manuscript={ms}, openaccess={oa})"
        )

    local_dir = os.path.join(out_root, ver_name)
    os.makedirs(local_dir, exist_ok=True)
    objects_done = []

    text_url = to_https_url(meta.get("text_url") or "")
    xml_url = to_https_url(meta.get("xml_url") or "")
    targets = []
    if text_url:
        targets.append(("txt", text_url, os.path.join(local_dir, f"{ver_name}.txt")))
    if with_xml and xml_url:
        targets.append(("xml", xml_url, os.path.join(local_dir, f"{ver_name}.xml")))
    if not text_url and xml_url:  # 兜底：没有 txt 就抓 xml
        targets.append(("xml", xml_url, os.path.join(local_dir, f"{ver_name}.xml")))

    for kind, url, dest in targets:
        size, ok = download_and_verify(url, dest)
        objects_done.append(f"{kind}({size}B,md5={'ok' if ok else 'MISMATCH'})")

    # 保存元数据 JSON 备份
    meta_dest = os.path.join(local_dir, f"{ver_name}.json")
    with open(meta_dest, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    objects_done.append(f"json({os.path.getsize(meta_dest)}B)")

    writer.writerow([
        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        meta.get("pmcid", pmcid), meta.get("version"), meta.get("pmid"),
        (meta.get("title") or "")[:200], meta.get("license_code"),
        meta.get("is_manuscript"), meta.get("is_pmc_openaccess"),
        (meta.get("citation") or "")[:150], local_dir, ";".join(objects_done),
    ])
    print(f"[ok ] {ver_name} | {meta.get('license_code')} | {(meta.get('title') or '')[:90]}")
    return "[ok]"


def main() -> int:
    ap = argparse.ArgumentParser(description="PMC OA(oa_comm 等价)子集拉取工具")
    ap.add_argument("--query", help="eSearch 主题词，如 'diabetes mellitus'")
    ap.add_argument("--pmcid", nargs="*", type=str, help="直接指定 PMCID(可多个)")
    ap.add_argument("--max", type=int, default=5, help="最多拉取篇数(默认5)")
    ap.add_argument("--out", default="data/pmc_oa", help="输出根目录")
    ap.add_argument("--with-xml", action="store_true", help="同时下载 JATS XML")
    ap.add_argument("--sleep", type=float, default=0.35, help="NCBI 请求间隔(秒)")
    ap.add_argument("--mindate", help="最早出版日期，如 2022/01/01")
    ap.add_argument("--maxdate", help="最晚出版日期，如 2026/12/31")
    args = ap.parse_args()

    if not (args.query or args.pmcid):
        ap.error("必须提供 --query 或 --pmcid")
    out_root = os.path.abspath(args.out)
    os.makedirs(out_root, exist_ok=True)
    manifest_path = os.path.join(out_root, "manifest.csv")
    new_manifest = not os.path.exists(manifest_path)
    mf = open(manifest_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(mf)
    if new_manifest:
        writer.writerow(MANIFEST_HEADER)

    pmcids = []
    if args.pmcid:
        pmcids = [x if x.upper().startswith("PMC") else "PMC" + x for x in args.pmcid]
    else:
        pmcids = esearch_pmc(
            args.query,
            retmax=max(10, args.max * 3),
            mindate=args.mindate,
            maxdate=args.maxdate,
        )
        if not pmcids:
            print("[warn] eSearch 未返回任何 PMCID")
            mf.close()
            return 1
        time.sleep(args.sleep)

    seen: set[str] = set()
    done, skipped = 0, 0
    for pid in pmcids:
        if done >= args.max:
            break
        pmcid = pid.upper() if pid.upper().startswith("PMC") else "PMC" + pid
        if pmcid in seen:
            continue
        seen.add(pmcid)
        status = fetch_pmcid(pmcid, out_root, writer, with_xml=args.with_xml)
        if status.startswith("[ok]"):
            done += 1
        else:
            skipped += 1
            print(status)
        time.sleep(args.sleep)

    mf.close()
    print("-" * 60)
    print(f"完成: 成功 {done} 篇, 跳过 {skipped} 篇 -> {out_root}")
    print(f"清单: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
