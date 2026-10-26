"""医学提示词模板（生成模块与提示词工程第一部分）。

四个阶段不是"多此一举"，而是把一次生成拆成分工明确的四步:

    evidence_evaluator  先给逐条证据定级 —— 防止低质量证据混进答案
    answer_generator    只依据"可用"证据写答案 —— 每条陈述必须能溯源
    critical_reviewer   反过来找答案的漏洞 —— LLM 天然倾向说得比证据更肯定
    final_assembler     吸收意见出终稿 —— 补引用清单与免责声明

医学场景的三条硬要求，写进了每个模板:
    1. 不编造：证据没覆盖的，必须明说"未检索到相关证据"
    2. 可溯源：每句结论都要带 [n] 编号，n 对应上下文里的来源编号
    3. 有分寸：区分研究类型强度，不做超出证据强度的断言

max_tokens 为什么给得比实际答案长很多（实测约束）:
    deepseek-r1:7b 是推理模型，输出分 thinking 和正文两段，且**无法关闭
    thinking**（实测 think=False 无效）。一次问答里 thinking 约占 60~70%
    的 token，而 max_tokens 是两段的总预算 —— 给少了会出现"思考完了、
    正文一个字没生成"的空答案（实测 num_predict=150 时答案为空）。
    所以这里的值都按"thinking + 正文"的总量给。

用法:
    python scripts/medical_prompts.py --list
    python scripts/medical_prompts.py --show answer_generator
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Any, Dict, List

# 四阶段的执行顺序
PIPELINE_ORDER = ["evidence_evaluator", "answer_generator",
                  "critical_reviewer", "final_assembler"]

DISCLAIMER = ("以上回答基于检索到的文献片段生成，仅供学习与研究参考，"
              "不构成临床诊断或治疗建议；具体诊疗请以临床医师判断为准。")


@dataclass
class PromptStage:
    """一个提示词阶段。"""
    name: str
    system_prompt: str
    user_prompt_template: str
    temperature: float = 0.1
    max_tokens: int = 1200
    description: str = ""

    @property
    def required_vars(self) -> List[str]:
        """模板里需要的变量名。"""
        out = []
        for key in ("context", "question", "evidence_assessment",
                    "draft_answer", "review", "disclaimer"):
            if "{" + key + "}" in self.user_prompt_template:
                out.append(key)
        return out

    def render(self, **kwargs: Any) -> str:
        """填充模板。

        不用 str.format：证据文本里常常带花括号（如 "p{value}"），format 会
        直接抛 KeyError。逐个 replace 对含特殊字符的医学文本更稳。
        """
        text = self.user_prompt_template
        for k, v in kwargs.items():
            text = text.replace("{" + k + "}", str(v))
        return text

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name, "description": self.description,
            "temperature": self.temperature, "max_tokens": self.max_tokens,
            "required_vars": self.required_vars,
        }


PROMPT_STAGES: Dict[str, PromptStage] = {}

PROMPT_STAGES["evidence_evaluator"] = PromptStage(
    name="证据评估器",
    description="逐条判定证据是否支持回答该问题，并给出研究类型与证据强度",
    temperature=0.0,      # 判定任务要稳定、可复现，不给随机性
    # 2400 是踩过坑之后的值: 原来给 1400, 实测模型只输出了"可用证据编号:5、8、12"
    # 一行(31 字符)就停了 —— 注意它只用了 543 tokens, 远没到上限, 说明不是被截断,
    # 而是模型自己跳过逐条环节直接给结论。加预算同时, 更重要的是下面的强制措辞。
    max_tokens=2400,
    system_prompt=(
        "你是医学循证评价专家。你的任务是评估检索到的文献片段与用户问题的相关性，"
        "并判断每条证据的强度。\n"
        "严格遵循：\n"
        "0. 对上下文里的【每一个】编号都要给出一行判定，不得省略、不得合并；\n"
        "1. 只依据给出的片段本身判断，不得引入片段之外的医学知识；\n"
        "2. 研究类型强度顺序：系统评价/Meta分析 > 随机对照试验(RCT) > 队列研究 > "
        "病例对照研究 > 横断面研究 > 病例报告 > 综述/叙述性文献 > 动物或体外实验；\n"
        "3. 若片段未明确说明研究类型，标注为“未知”，不要猜测；\n"
        "4. 若片段与问题无关，直接判为“不支持”，不强行解释。"
    ),
    user_prompt_template=(
        "用户问题：{question}\n\n"
        "检索到的证据片段如下：\n"
        "----------------\n"
        "{context}\n"
        "----------------\n\n"
        "对每一个编号，用一行给出结论，格式为：\n"
        "编号 - 判定 - 理由\n"
        "判定只能用这四个词之一：支持 / 部分支持 / 不支持 / 无关；"
        "理由限 15 字以内。\n"
        "上下文有几个编号就写几行，不要省略、不要另起小标题。\n\n"
        "写完所有编号后，最后一行输出：\n"
        "可用证据编号：<列出判定为“支持”或“部分支持”的编号>"
    ),
)

PROMPT_STAGES["answer_generator"] = PromptStage(
    name="答案生成器",
    description="仅依据评估通过的证据生成带引用的答案",
    temperature=0.2,      # 需要一点表达多样性，但要远低于默认，避免发挥
    max_tokens=1600,
    system_prompt=(
        "你是严谨的医学问答助手。你的答案必须完全基于给定的证据片段。\n"
        "严格遵循：\n"
        "1. 每一条结论后必须紧跟引用编号，格式如 [3]，编号对应证据片段编号；\n"
        "2. 只能用给定片段中实际出现的信息，不得补全、推测或引入外部知识；\n"
        "3. 证据未涉及的内容，直接说明“现有检索到的证据未涉及此点”，不得回避也不得编造；\n"
        "4. 措辞强度要与证据强度匹配：RCT/Meta 分析可用“表明”，"
        "观察性研究用“提示”，动物实验必须说明“在动物实验中观察到”；\n"
        "5. 使用中文回答，语言平实，先给核心结论，再展开要点；\n"
        "6. 不要输出你的思考过程，直接给出答案。"
    ),
    user_prompt_template=(
        "用户问题：{question}\n\n"
        "证据片段：\n"
        "----------------\n"
        "{context}\n"
        "----------------\n\n"
        "证据评估结果：\n"
        "{evidence_assessment}\n\n"
        "请依据“可用证据编号”对应的片段作答（其余片段不要使用）。\n"
        "要求：核心结论先行，随后分点说明，每点末尾标注引用编号。"
    ),
)

PROMPT_STAGES["critical_reviewer"] = PromptStage(
    name="批判性审查器",
    description="挑出答案中缺乏证据支持或过度概括的表述",
    temperature=0.0,      # 挑错任务，要稳定
    max_tokens=1400,
    system_prompt=(
        "你是医学文献审评专家，负责挑出答案中的问题。\n"
        "重点检查以下几类错误：\n"
        "1. 无引用支持：某一句话后面没有 [n]，或引用的编号并不支持该说法；\n"
        "2. 过度概括：把特定人群、剂量、随访时间的结论说成普遍结论；\n"
        "3. 跨物种外推：把动物或体外实验的结论当作临床结论；\n"
        "4. 因果倒置：把相关性说成因果性；\n"
        "5. 遗漏限定：缺少样本量、研究时限、人群特征等关键限定条件；\n"
        "6. 反向证据被忽略：给定片段中若有相反或不一致的结论未被提及。\n"
        "若某一类没有问题，写“无”。不要重写答案，只列问题。"
    ),
    user_prompt_template=(
        "用户问题：{question}\n\n"
        "证据片段：\n"
        "----------------\n"
        "{context}\n"
        "----------------\n\n"
        "待审答案：\n"
        "{draft_answer}\n\n"
        "请按上述六类逐条列出发现的问题，"
        "每条注明属于哪一类、涉及原文的哪一句、应如何修正。"
    ),
)

PROMPT_STAGES["final_assembler"] = PromptStage(
    name="最终组装器",
    description="吸收审查意见输出终稿，附来源清单与免责声明",
    temperature=0.1,
    max_tokens=1600,
    system_prompt=(
        "你是医学问答的定稿编辑。你要把初稿和审查意见合并成最终答案。\n"
        "要求：\n"
        "1. 吸收审查意见：删除或弱化被指出无证据支持的句子，补上缺失的限定条件；\n"
        "2. 保留引用编号：终稿的每个结论仍须带 [n]；\n"
        "3. 不新增未经证据支持的内容；\n"
        "4. 结尾附上“参考来源”清单，按编号列出来源短语（每条一行）；\n"
        "5. 最后附上免责声明（原样输出，不要改写）。"
    ),
    user_prompt_template=(
        "用户问题：{question}\n\n"
        "证据片段：\n"
        "----------------\n"
        "{context}\n"
        "----------------\n\n"
        "答案初稿：\n"
        "{draft_answer}\n\n"
        "审查意见：\n"
        "{review}\n\n"
        "请输出最终答案，结尾包含参考来源清单与下面这段免责声明：\n"
        "{disclaimer}"
    ),
)


def get_stage(key: str) -> PromptStage:
    if key not in PROMPT_STAGES:
        raise KeyError(f"未知阶段: {key}，可选 {list(PROMPT_STAGES)}")
    return PROMPT_STAGES[key]


def main() -> None:
    ap = argparse.ArgumentParser(description="医学提示词模板")
    ap.add_argument("--list", action="store_true", help="列出四个阶段")
    ap.add_argument("--show", default=None, help="打印某个阶段的完整提示词")
    args = ap.parse_args()

    if args.show:
        st = get_stage(args.show)
        print(f"=== {args.show}  {st.name} ===")
        print(f"temperature={st.temperature}  max_tokens={st.max_tokens}  "
              f"需要变量={st.required_vars}")
        print(f"\n--- system_prompt ---\n{st.system_prompt}")
        print(f"\n--- user_prompt_template ---\n{st.user_prompt_template}")
        return

    print(f"{'key':<20} {'name':<12} {'temp':>5} {'max_tok':>8}  变量")
    print("-" * 88)
    for key in PIPELINE_ORDER:
        st = PROMPT_STAGES[key]
        print(f"{key:<20} {st.name:<12} {st.temperature:>5} {st.max_tokens:>8}  "
              f"{','.join(st.required_vars)}")
    print("\n说明: max_tokens 包含 thinking 与正文两段的总预算 "
          "(deepseek-r1 无法关闭 thinking)")


if __name__ == "__main__":
    main()
