"""本地 LLM 集成与生成（生成模块与提示词工程第二部分）。

设计要点（都是踩过坑之后定下来的）:

1. **NO_PROXY 必须在创建 client 之前设**。Windows 的系统代理写在注册表而非
   环境变量, httpx 默认 trust_env=True 会把 localhost 请求也发给代理,
   导致 ollama 全部 WinError 10061（阶段⑤就栽过一次）。

2. **num_ctx 必须显式放大**。Ollama 默认 2048, 我们的上下文动辄 2000+ token,
   不设会被**静默截断**——末段证据凭空消失且不报错。

3. **max_tokens 是 thinking + 正文的总预算**。deepseek-r1 的思考无法关闭
   （实测 think=False 在 client 与 HTTP 两路都被静默忽略）, 给少了会出现
   "思考完了、正文一个字没有"的空答案。

4. **JSON 不信任模型自觉**。要求输出JSON不等于会输出合法JSON, 所以生成后
   必须过一道"提取 + 修复", 修不好就如实返回 None 并保留原文,
   **绝不编造一个结构体冒充模型输出**。

用法:
    python scripts/llm_generator.py --test            # 连接测试(秒级)
    python scripts/llm_generator.py --demo            # 普通生成(约1分钟)
    python scripts/llm_generator.py --demo-json       # JSON生成+提取修复
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import time
from typing import Any, Dict, Iterable, List, Optional

# 必须在 import ollama / 创建 Client 之前设置
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")

DEFAULT_MODEL = "deepseek-r1:7b"
DEFAULT_BASE_URL = "http://localhost:11434"


class LLMGenerator:
    """本地 Ollama 模型的生成封装。"""

    def __init__(self, model_name: str = DEFAULT_MODEL,
                 base_url: str = DEFAULT_BASE_URL,
                 timeout: int = 300,
                 temperature: float = 0.1,
                 max_tokens: int = 1600,
                 num_ctx: int = 8192):
        """
        Args:
            model_name: Ollama 模型名称
            base_url: Ollama 服务地址
            timeout: 请求超时时间（秒）。生成在 CPU 上单次可达 3 分钟,
                     所以默认 300 而不是常见的 30/60
            temperature: 默认温度（各阶段可用自己的值覆盖）
            max_tokens: 默认最大生成 token（含 thinking）
            num_ctx: 上下文窗口。默认 2048 会静默截断, 这里显式放大
        """
        self.model_name = model_name
        self.base_url = base_url
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.num_ctx = num_ctx

        from ollama import Client
        self.client = Client(host=base_url, timeout=timeout)

    # ------------------------------------------------------------ 连接测试
    def test_connection(self, quick: bool = False) -> Dict[str, Any]:
        """初始化并测试链接。

        quick=False: 只验证服务可达 + 模型已拉取（秒级）
        quick=True:  再跑一次极短生成, 验证 temperature / max_tokens 参数链路
                     确实能走通（约 20~40 秒）
        """
        out: Dict[str, Any] = {"base_url": self.base_url, "model": self.model_name}
        try:
            info = self.client.list()
            models = info.get("models", []) if isinstance(info, dict) else info.models
            names = []
            for m in models:
                names.append(m.get("model") if isinstance(m, dict)
                             else getattr(m, "model", getattr(m, "name", "")))
            out["service_ok"] = True
            out["available_models"] = names
            out["model_present"] = any(self.model_name in (n or "") for n in names)
        except Exception as e:                           # noqa: BLE001
            out.update(service_ok=False, model_present=False,
                       error=f"{type(e).__name__}: {e}")
            return out

        if out["model_present"]:
            try:
                detail = self.client.show(self.model_name)
                d = detail if isinstance(detail, dict) else detail.model_dump()
                out["capabilities"] = d.get("capabilities")
            except Exception as e:                       # noqa: BLE001
                out["show_error"] = str(e)[:120]

        if quick and out["model_present"]:
            r = self.generate("用一句话回答：二甲双胍是什么药？",
                              system_prompt="你是医学助手。",
                              temperature=0.0, max_tokens=200)
            out["probe"] = {k: r[k] for k in
                            ("success", "seconds", "eval_count", "tokens_per_s")}
            out["probe"]["answer_preview"] = r["content"][:60]
        return out

    # ------------------------------------------------------------ 生成
    def generate(self, prompt: str, system_prompt: Optional[str] = None,
                 temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None,
                 json_mode: bool = False) -> Dict[str, Any]:
        """单次生成。

        Args:
            prompt: 用户提示
            system_prompt: 系统提示（按模型格式作为 system role 传入）
            temperature / max_tokens: 覆盖实例默认值
            json_mode: True 时启用 Ollama 的 format='json' 原生约束

        Returns:
            {success, content, thinking, raw_text, eval_count, prompt_eval_count,
             seconds, tokens_per_s, temperature, max_tokens, error}
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        temp = self.temperature if temperature is None else temperature
        mt = self.max_tokens if max_tokens is None else max_tokens
        options = {"temperature": temp, "num_predict": mt, "num_ctx": self.num_ctx}

        kwargs: Dict[str, Any] = {}
        if json_mode:
            kwargs["format"] = "json"

        t = time.perf_counter()
        try:
            resp = self.client.chat(model=self.model_name, messages=messages,
                                    options=options, **kwargs)
            dt = time.perf_counter() - t
            msg = resp["message"] if isinstance(resp, dict) else resp.message
            content = msg.get("content") if isinstance(msg, dict) else msg.content
            thinking = (msg.get("thinking") if isinstance(msg, dict)
                        else getattr(msg, "thinking", None)) or ""
            content = content or ""
            n = (resp.get("eval_count") if isinstance(resp, dict)
                 else getattr(resp, "eval_count", 0)) or 0
            nprompt = (resp.get("prompt_eval_count") if isinstance(resp, dict)
                       else getattr(resp, "prompt_eval_count", 0)) or 0
            return {
                "success": bool(content.strip()),
                "content": content,
                "thinking": thinking,
                # 有些调用路径会把 <think> 混进正文, 这里统一留一份原始文本供提取
                "raw_text": content,
                "eval_count": n,
                "prompt_eval_count": nprompt,
                "seconds": round(dt, 1),
                "tokens_per_s": round(n / dt, 2) if dt else 0.0,
                "temperature": temp, "max_tokens": mt,
                "error": None if content.strip() else "正文为空(thinking 可能吃光了 max_tokens)",
            }
        except Exception as e:                           # noqa: BLE001
            return {"success": False, "content": "", "thinking": "", "raw_text": "",
                    "eval_count": 0, "prompt_eval_count": 0,
                    "seconds": round(time.perf_counter() - t, 1),
                    "tokens_per_s": 0.0, "temperature": temp, "max_tokens": mt,
                    "error": f"{type(e).__name__}: {e}"}

    # ------------------------------------------------------------ JSON 提取与修复
    @staticmethod
    def extract_json(text: str) -> Optional[Any]:
        """从文本中提取 JSON 并尽量修复格式。

        依次尝试: 原文 → 去代码围栏 → 截取括号区间 → 修尾逗号/中文引号 →
        补全未闭合括号 → ast.literal_eval(兼容单引号)。
        全部失败返回 None（由调用方决定降级, 不在这里硬造数据）。
        """
        if not text or not text.strip():
            return None
        s = text.strip()
        # thinking 混进正文的情况（HTTP 路径或旧版客户端）
        s = re.sub(r"<think>.*?</think>", "", s, flags=re.S).strip()

        variants: List[str] = []

        def add(x: str) -> None:
            x = x.strip()
            if x and x not in variants:
                variants.append(x)

        add(s)
        # 去 markdown 代码围栏
        fence = re.sub(r"^```(?:json|JSON)?\s*", "", s)
        fence = re.sub(r"\s*```$", "", fence)
        add(fence)
        # 截取第一个开括号到最后一个闭括号之间（丢掉前后解释性文字）
        for base in list(variants):
            for op, cl in (("{", "}"), ("[", "]")):
                i, j = base.find(op), base.rfind(cl)
                if i != -1 and j > i:
                    add(base[i:j + 1])

        repaired: List[str] = []
        for v in variants:
            r = v
            # 中文引号 → 英文引号
            r = r.replace("“", '"').replace("”", '"')
            r = r.replace("‘", "'").replace("’", "'")
            # 尾逗号: {"a":1,} / [1,2,]
            r = re.sub(r",\s*([}\]])", r"\1", r)
            # 去掉字符串外的控制字符
            r = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", r)
            repaired.append(r)
            closed = LLMGenerator._close_brackets(r)
            if closed != r:
                repaired.append(closed)
        variants.extend(repaired)

        for v in variants:
            try:
                return json.loads(v)
            except Exception:                            # noqa: BLE001
                continue
        for v in variants:                               # 兼容单引号/True/False
            try:
                obj = ast.literal_eval(v)
                if isinstance(obj, (dict, list)):
                    return obj
            except Exception:                            # noqa: BLE001
                continue
        return None

    @staticmethod
    def _close_brackets(s: str) -> str:
        """补全未闭合的括号（生成被 max_tokens 截断时最常见）。

        用栈而不是分别统计 {} 和 [] 的个数：嵌套截断如 {"a":1,"b":[1,2
        两种括号都得补，且必须**后开的先闭**（先 ] 再 }），分开统计会补出
        两种各自都仍不合法的串。扫描时跳过字符串内部，避免把值里的括号算进去。
        """
        stack: List[str] = []
        in_str = False
        esc = False
        for ch in s:
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch in "{[":
                stack.append(ch)
            elif ch in "}]":
                if stack:
                    stack.pop()
        out = s
        if in_str:                                   # 字符串本身被截断
            out += '"'
        out = out.rstrip().rstrip(",")
        closers = {"{": "}", "[": "]"}
        while stack:
            out += closers[stack.pop()]
        return out

    def generate_json(self, prompt: str, system_prompt: Optional[str] = None,
                      temperature: Optional[float] = None,
                      max_tokens: Optional[int] = None,
                      use_native_json: bool = True) -> Dict[str, Any]:
        """生成并解析 JSON。

        系统提示里追加 JSON 格式要求（任务书要求）, 同时可用 Ollama 原生
        format='json' 双保险。解析失败时 success=False 但保留原文与错误,
        由上层决定降级策略。
        """
        sys_p = (system_prompt or "")
        sys_p += ("\n输出要求：只输出一个合法 JSON 对象，不要输出任何解释、"
                  "不要用 markdown 代码围栏、不要在 JSON 前后添加文字。")
        r = self.generate(prompt, system_prompt=sys_p, temperature=temperature,
                          max_tokens=max_tokens, json_mode=use_native_json)
        parsed = self.extract_json(r["content"]) if r["content"] else None
        r["json"] = parsed
        r["json_ok"] = parsed is not None
        if parsed is None and r["success"]:
            r["error"] = "JSON 提取/修复失败"
        return r

    # ------------------------------------------------------------ 批量生成
    def generate_batch(self, items: Iterable[Dict[str, Any]],
                       json_mode: bool = False,
                       progress: bool = True) -> List[Dict[str, Any]]:
        """批量生成。

        顺序执行而非并发：CPU 上只有一个模型实例, 并发只会互相抢核、
        让每条都变慢, 还可能导致 Ollama 排队超时。
        单条失败不会中断整批——失败的条目带 error 返回, 便于事后定位。
        """
        items = list(items)
        out: List[Dict[str, Any]] = []
        total = len(items)
        for i, it in enumerate(items, 1):
            if progress:
                print(f"[batch {i}/{total}] 生成中...", flush=True)
            r = self.generate(it["prompt"],
                              system_prompt=it.get("system_prompt"),
                              temperature=it.get("temperature"),
                              max_tokens=it.get("max_tokens"),
                              json_mode=json_mode or it.get("json_mode", False))
            r["index"] = i - 1
            if "name" in it:
                r["name"] = it["name"]
            out.append(r)
            if progress:
                print(f"   {r['seconds']}s  {r['eval_count']} tokens  "
                      f"输出 {len(r['content'])} 字符"
                      f"{'' if r['success'] else '  失败: ' + str(r['error'])}",
                      flush=True)
        return out


def main() -> None:
    ap = argparse.ArgumentParser(description="本地 LLM 生成器")
    ap.add_argument("--test", action="store_true", help="连接测试(秒级)")
    ap.add_argument("--test-quick", action="store_true", help="连接测试+短生成(约30秒)")
    ap.add_argument("--demo", action="store_true", help="普通生成演示")
    ap.add_argument("--demo-json", action="store_true", help="JSON生成与修复演示")
    args = ap.parse_args()

    gen = LLMGenerator()
    if args.test or args.test_quick:
        print(json.dumps(gen.test_connection(quick=args.test_quick),
                         ensure_ascii=False, indent=2)[:1500])
        return

    if args.demo_json:
        r = gen.generate_json(
            '对2条证据判定是否支持"二甲双胍降低心血管风险"。\n'
            "[1] 二甲双胍可降低2型糖尿病患者全因死亡率。\n"
            "[2] 本研究讨论阿司匹林在卒中二级预防中的剂量。\n"
            '格式: {"evaluations":[{"id":1,"verdict":"支持","reason":"..."}],"usable_ids":[1]}',
            system_prompt="你是医学循证评价助手。", temperature=0.0, max_tokens=700)
        print(f"耗时 {r['seconds']}s  JSON解析={'成功' if r['json_ok'] else '失败'}")
        print("正文:", r["content"][:600])
        print("解析结果:", json.dumps(r["json"], ensure_ascii=False)[:400])
        return

    if args.demo:
        r = gen.generate("用三句话说明二甲双胍的临床用途。",
                         system_prompt="你是严谨的医学助手。", max_tokens=600)
        print(f"耗时 {r['seconds']}s  {r['eval_count']} tokens  {r['tokens_per_s']} t/s")
        print("正文:", r["content"][:600])
        return

    # 默认: 离线验证 JSON 修复器（不需要模型）
    print("JSON 提取修复器自检（离线）:")
    cases = {
        "标准JSON": '{"a": 1, "b": [1, 2]}',
        "带代码围栏": '```json\n{"a": 1}\n```',
        "带解释文字": '好的，结果如下：\n{"a": 1}\n以上。',
        "尾逗号": '{"a": 1, "b": [1, 2,],}',
        "中文引号": '{"a": “二甲双胍”}',
        "未闭合(截断)": '{"a": 1, "b": [1, 2',
        "thinking混入": '<think>让我想想</think>{"a": 1}',
        "单引号": "{'a': 1, 'b': True}",
        "完全不是JSON": '我不知道答案。',
    }
    for name, txt in cases.items():
        obj = LLMGenerator.extract_json(txt)
        print(f"  {name:<16} -> {'解析成功 ' + json.dumps(obj, ensure_ascii=False)[:40] if obj is not None else 'None(如实失败)'}")


if __name__ == "__main__":
    main()
