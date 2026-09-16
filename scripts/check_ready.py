# -*- coding: utf-8 -*-
"""准备阶段环境自检：依赖 / 数据源 / Chroma / 本地 LLM / 嵌入能力。

用途：验证「基于 RAG 的医学专业知识生成 LLM」的本地运行底座是否就绪。
不包含任何 RAG 业务逻辑（切块、检索、生成答案）。

用法（rag_medical 环境）：
    conda activate rag_medical
    python scripts/check_ready.py
"""
import glob
import os
import shutil
import tempfile
import time

RESULTS = []


def _ver(pkg):
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _v

    try:
        return _v(pkg)
    except PackageNotFoundError:
        return "MISSING"


def check(name, fn):
    try:
        RESULTS.append((name, "PASS", fn()))
    except Exception as e:  # noqa: BLE001
        RESULTS.append((name, "FAIL", f"{type(e).__name__}: {str(e)[:160]}"))


def deps():
    import torch

    return (
        f"torch={torch.__version__} cuda={torch.cuda.is_available()} "
        f"threads={torch.get_num_threads()} | "
        f"langchain={_ver('langchain')} langchain-ollama={_ver('langchain-ollama')} | "
        f"chromadb={_ver('chromadb')} | pandas={_ver('pandas')} numpy={_ver('numpy')} | "
        f"datasets={_ver('datasets')} | ollama-sdk={_ver('ollama')} | "
        f"hf-hub={_ver('huggingface_hub')} tokenizers={_ver('tokenizers')}"
    )


def datasource():
    files = sorted(glob.glob(r"d:\medical_rag\data\pmc_oa\*\*.txt"))
    total = sum(os.path.getsize(f) for f in files)
    sample = open(files[0], encoding="utf-8", errors="replace").read()
    sep = [i for i, ln in enumerate(sample.splitlines()) if chr(0x9F) in ln]
    return (
        f"{len(files)} articles | {total / 1024 / 1024:.2f} MB | "
        f"sample={os.path.basename(files[0])} chars={len(sample)} body_sep_line={sep}"
    )


def chroma_persist():
    import chromadb

    tmp = tempfile.mkdtemp(prefix="chroma_smoke_")
    try:
        client = chromadb.PersistentClient(path=tmp)
        col = client.get_or_create_collection("smoke", embedding_function=None)
        col.upsert(
            ids=["a", "b", "c"],
            documents=["metformin lowers hepatic glucose", "insulin therapy", "retinopathy screening"],
            embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        )
        res = col.query(query_embeddings=[[1.0, 0.0, 0.0]], n_results=1)
        return f"count={col.count()} top1={res['documents'][0][0]!r}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def llm_langchain():
    from langchain_ollama import ChatOllama

    llm = ChatOllama(model="deepseek-r1:7b", temperature=0, num_predict=256)
    t0 = time.time()
    msg = llm.invoke("用一句话回答：二甲双胍（Metformin）主要用于治疗什么疾病？")
    el = time.time() - t0
    content = (msg.content or "").strip()
    thinking = ""
    try:
        thinking = (getattr(msg, "additional_kwargs", None) or {}).get("thinking", "") or ""
    except Exception:  # noqa: BLE001
        pass
    head = content or thinking
    return (
        f"{el:.1f}s | content_len={len(content)} thinking_len={len(thinking)} | "
        f"head={head[:110]!r}"
    )


def embeddings_probe():
    import ollama as ollama_sdk

    out = []
    for model in ("bge-m3", "nomic-embed-text"):
        try:
            r = ollama_sdk.embeddings(model=model, prompt="type 2 diabetes")
            out.append(f"{model}=OK(dim={len(r['embedding'])})")
        except Exception as e:  # noqa: BLE001
            out.append(f"{model}=NOT_READY({str(e)[:60]})")
    return " | ".join(out)


check("deps", deps)
check("datasource", datasource)
check("chroma_persist", chroma_persist)
check("llm_langchain", llm_langchain)
check("embeddings_probe", embeddings_probe)

print("=" * 78)
for name, status, detail in RESULTS:
    print(f"[{status:4}] {name:18} {detail}")
print("=" * 78)
