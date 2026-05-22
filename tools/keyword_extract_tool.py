"""
LLM 关键词提取工具（核心工具）

封装 processor.py 中的 LLM 调用与 JSON 解析逻辑，
去掉兜底链（兜底决策交由 Agent），保留重试机制。
"""

import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

from .base_tool import BaseTool, ToolResult
from ..config import AgentConfig
from ..utils.json_parser import (
    extract_json_and_thinking,
    try_repair_truncated_json,
)
from ..core.post_process import (
    post_process_keywords_with_config,
    extract_keywords_from_json,
    normalize_keywords_data,
)

logger = logging.getLogger(__name__)


def _get_keyword_prompt_long(comment: str) -> tuple:
    system_prompt = """
你是一名中文电商评论关键词抽取专家，正在为训练一个AI模型生成高质量的"思考-输出"范例。

你的任务是为下面这条评论生成一个完整的输出。这个输出将直接用于AI模型的训练，因此必须严格遵循格式和规则。


【核心规则】
1. **忠于原文**：关键词必须能在原文逐字对齐，可去除程度副词（很/比较/挺/有点…），禁止改写、归纳总结、意译。
2. **先推理，后给词**：必须先进行明确的内部推理，再给出最终关键词；严禁直接凭直觉"拍关键词"。
3. **一条记录只对应一个关键词**：
   - JSON 中的每个三元组，只能有一个最终"关键词"。
   - 如果一句话里拆出了多个原子词（如"做工精致"→"做工""精致"），必须为每个原子词分别输出一条三元组，不能把多个关键词塞进同一条记录。
4. **对象 + 描述都要单独输出**：
   - 若存在"对象词 + 描述词"结构（如"物流很快""质量很好""客服态度差"），对象词和描述词都必须作为独立的关键词，各占一条记录。
5. **弱化纯时间 / 数字信息**：
   - 一般不把纯日期、纯时间长度、订单编号等（如"22号""25号""半个多月""15天"）作为关键词。
   - 这类信息可以出现在推理说明中，用来解释背景，但**不要**作为"关键词"字段输出，除非它直接构成规则限制（如"15天包退"中的"包退"才是关键词）。
6. **长度与数量限制**：
   - 每个关键词长度**≤4 个汉字**。
   - 关键词列表最多输出 **15 个**高分关键词，按重要性降序排列。

【关键词抽取规则（分层原子化）】
1) **主体定位**：
   - 通读全段，提取全局"主体对象"词。
   - 商品主体 / 属性 / 服务角色（如：衣服、手机、屏幕、做工、客服、物流等）若小于等于 4 字，必须单独输出为关键词。
2) **评价、描述定位**：
   - 分句扫描评价、描述片段，找到与主体相关的描述词 / 情绪词（好、差、快、慢、不错、失望等）。
   - 对每个"主体 + 描述"结构，同时为主体词和描述词各输出一条关键词记录。
   - 若文本出现描述成语（如"物美价廉""货真价实"），该成语应作为单一原子级关键词整体输出，不得拆分。
4) **补充名词**：
   - 若文本中出现了与主体对象相关的**有意义的名词**，不论是否存在明显评价词/描述词/情绪词，该名词**必须单独提取**作为关键词。
   - 优先保留与商品、质量、价格、功能、外观、物流、客服、使用体验直接相关的名词；像"购物""传送"等过于笼统的词，除非被反复强调，否则可以只出现在推理说明中，不必作为关键词输出。
5) **否定与问题场景**：
   - 保留"不/没 + 核心词"的紧凑结构（如：没掉色、不防摔、不满意、没送到）。
   - 对否定结构，同样拆成原子词：主体 + 否定描述（如"物流不给力"→"物流""不给力"）。

【核心输出格式要求】
你必须生成一个包含以下两部分的文本，中间用两个换行分隔：
1. **思考**：在此标题下，详细写出你如何应用规则一步步分析评论、拆分原子关键词的推理过程。这部分是你的"内心独白"，需要展示完整的逻辑链条。
2. **JSON输出**：在思考结束后，空两行，然后输出一个且仅一个JSON对象。

【JSON格式规范】
最终输出的JSON必须且只能是以下结构，不要有任何额外文本：
{
  "keywords": [
    ["这里是针对这个单一关键词的、一句话推理说明", "关键词", "重要性分数"],
    ...
  ]
}

*   "推理说明"需简明，解释为何提取此词。
*   "重要性分数"为0-1之间的小数(保留两位)，反映该词在评论中的突出程度。

"""
    user_prompt = f"""请分析以下评论并提取关键词：

【待处理评论】
{comment}

请严格按照上述规则和JSON格式输出。"""
    return system_prompt, user_prompt


def _get_keyword_prompt_short(comment: str) -> tuple:
    system_prompt = """你是一名中文电商评论关键词抽取专家。当前评论长度在 10 字以内，属于"短评"。

你的任务是：先通过简要推理，识别评论中的"对象词"和"描述/评价词"，再从中提取最能表达评论含义的关键词。

【核心概念】
- 对象词：被描述/被评价的东西（如：画质、质量、性价比、物流、做工、感受等）。**不可无中生有**
- 描述/评价词：对对象给出特征或态度的词（如：清楚、高、不错、给力、垃圾、满意等）。

【抽取规则】
1. 若短评中出现"对象词 + 描述/评价词"的结构，则：
   - 先在推理中明确指出：对象词是什么、对应的描述词是什么。
   - 然后将"对象词"和"描述/评价词"都作为独立关键词输出。
   例：
   - "画质清楚" → 对象：画质；描述：清楚 → 关键词："画质"、"清楚"
   - "性价比很高" → 对象：性价比；描述：高 → 关键词："性价比"、"高"
   - "画质清楚，质量保证" → 关键词："画质"、"清楚"、"质量"、"保证"

2. 若短评只有整体情绪或整体评价，且没有明确对象词，则：
   - 在推理中说明"未找到明确对象，只存在整体评价"。
   - 将整体评价短语作为一个关键词输出。
   例：
   - "挺好的" → 关键词："挺好的"
   - "还不错" → 关键词："还不错"

3. 若短评同时包含整体评价和"对象 + 描述"结构，则两部分都要覆盖：
   - 整体评价也可以作为关键词。
   - 每个 "对象 + 描述" 对中，对象词与描述词都要输出。
   例：
   - "还不错，性价比挺高的"
     → 整体评价："还不错"
     → 对象：性价比；描述：高
     → 关键词："还不错"、"性价比"、"高"

4. 不进行过度拆分：
   - 程度副词（如"很""挺"等）可以保留在短语内部，不必单独输出为关键词。
   - 如"挺好的"可以整体作为一个关键词，不拆成"挺""好"。

5. 所有关键词必须在原文中逐字出现（连续或可直连），单个关键词长度 ≤4 个汉字（专有名词除外）。

【输出要求】
对每一个关键词，都需要给出一条对应的"推理步骤"，用来说明你是如何从评论中找到该关键词的。

- 推理步骤中必须**显式说明**：
  - 本条关键词是否有对应的"对象词"；
  - 若有对象词，对象词是什么；若无对象词，要说明"无明确对象，仅为整体评价"；
  - 该关键词是对象词本身，还是对应的描述/评价词。

- 最终只输出严格 JSON，格式如下（不要添加任何其他内容）：
{
  "keywords": [
    ["推理步骤", "关键词", 重要性分数（0~1，两位小数）]
  ]
}

【输出示例（仅用于风格参考，实际以输入为准）】
- 对于"画质清楚，质量保证"，可能的输出为：
{
  "keywords": [
    ["识别到对象词'画质'及其描述词'清楚'，先提取对象词'画质'作为关键词", "画质", 0.95],
    ["对象词为'画质'，其描述/评价词为'清楚'，提取'清楚'作为画质表现的描述关键词", "清楚", 0.90],
    ["识别到对象词'质量'及其描述词'保证'，先提取对象词'质量'作为关键词", "质量", 0.93],
    ["对象词为'质量'，其描述/评价词为'保证'，提取'保证'作为质量可靠性的描述关键词", "保证", 0.88]
  ]
}


"""
    user_prompt = f"""请分析以下评论并提取关键词：

【待处理评论】
{comment}

请严格按照上述规则和JSON格式输出。"""
    return system_prompt, user_prompt


class KeywordExtractTool(BaseTool):

    def __init__(self, config: AgentConfig | None = None, llm_service=None):
        self._config = config or AgentConfig()
        self._client: OpenAI | None = None
        self._llm_service = llm_service

    def _get_client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(
                base_url=self._config.TOOL_LLM_BASE_URL,
                api_key=self._config.TOOL_LLM_API_KEY,
            )
        return self._client

    @property
    def name(self) -> str:
        return "keyword_extract"

    @property
    def description(self) -> str:
        return (
            "使用 LLM 从中文电商评论中提取结构化关键词。"
            "自动选择长评/短评 Prompt 模板，调用 Tool LLM 进行推理，"
            "解析 JSON 结果并进行后处理（去重、排序、过滤）。"
            "返回关键词三元组列表 [[推理说明, 关键词, 重要性分数], ...]。"
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "待提取关键词的评论文本（建议已预处理）",
                },
                "max_keywords": {
                    "type": "integer",
                    "description": "最多返回的关键词数量",
                    "default": 8,
                },
            },
            "required": ["text"],
        }

    def _save_debug_log(
        self,
        input_text: str,
        prompt_type: str,
        system_prompt: str,
        user_prompt: str,
        attempts: List[Dict[str, Any]],
        final_error: str,
    ) -> Optional[str]:
        """将 keyword_extract 的全部调试信息保存到 JSON 文件，返回文件路径。"""
        try:
            from ..cli.config_loader import DEFAULT_OUTPUT_DIR
            debug_dir = Path(DEFAULT_OUTPUT_DIR) / "debug"
        except Exception:
            debug_dir = Path("extract_agent_output") / "debug"

        try:
            debug_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = debug_dir / f"keyword_extract_debug_{ts}.json"
            debug_data = {
                "timestamp": datetime.now().isoformat(),
                "input_text": input_text,
                "input_text_length": len(input_text),
                "prompt_type": prompt_type,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "model": self._config.TOOL_LLM_MODEL,
                "max_tokens": self._config.TOOL_LLM_MAX_TOKENS,
                "final_error": final_error,
                "total_attempts": len(attempts),
                "attempts": attempts,
            }
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(debug_data, f, ensure_ascii=False, indent=2)
            logger.info(f"keyword_extract debug 日志已保存: {filepath}")
            return str(filepath)
        except Exception as e:
            logger.warning(f"保存 debug 日志失败: {e}")
            return None

    def _call_llm(self, text: str, attempt: int, cfg) -> tuple:
        """
        调用 LLM 获取关键词提取结果。
        优先使用 LLM Service + Skill 路径，无 LLM Service 时回退直接 OpenAI 调用。

        Returns:
            (raw_content, finish_reason, usage_dict)
        """
        use_penalty = attempt >= 2
        current_seed = cfg.TOOL_LLM_SEED + attempt if use_penalty else cfg.TOOL_LLM_SEED

        if self._llm_service:
            skill_name = (
                "keyword_extract_short"
                if len(text.strip()) < cfg.SHORT_TEXT_LEN
                else "keyword_extract_long"
            )
            extra_params: Dict[str, Any] = {"seed": current_seed}
            if use_penalty:
                extra_params["frequency_penalty"] = cfg.TOOL_LLM_FREQUENCY_PENALTY
                extra_params["extra_body"] = {
                    "repetition_penalty": cfg.TOOL_LLM_REPETITION_PENALTY,
                }
            llm_resp = self._llm_service.call_tool(
                skill_name=skill_name,
                variables={"comment": text},
                extra_params=extra_params,
            )
            return (
                (llm_resp.content or "").strip(),
                llm_resp.finish_reason,
                None,
            )

        if len(text.strip()) < cfg.SHORT_TEXT_LEN:
            system_prompt, user_prompt = _get_keyword_prompt_short(text)
        else:
            system_prompt, user_prompt = _get_keyword_prompt_long(text)

        call_kwargs: Dict[str, Any] = {
            "model": cfg.TOOL_LLM_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": cfg.TOOL_LLM_MAX_TOKENS,
            "temperature": cfg.TOOL_LLM_TEMPERATURE,
            "seed": current_seed,
        }
        if use_penalty:
            call_kwargs["frequency_penalty"] = cfg.TOOL_LLM_FREQUENCY_PENALTY
            call_kwargs["extra_body"] = {
                "repetition_penalty": cfg.TOOL_LLM_REPETITION_PENALTY,
            }

        client = self._get_client()
        resp = client.chat.completions.create(**call_kwargs)
        raw = resp.choices[0].message.content.strip()
        finish_reason = resp.choices[0].finish_reason
        usage = None
        if resp.usage:
            usage = {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "total_tokens": resp.usage.total_tokens,
            }
        return raw, finish_reason, usage

    def execute(self, **kwargs) -> ToolResult:
        text = kwargs.get("text", "")
        max_keywords = kwargs.get("max_keywords", self._config.N)

        start = time.time()

        if not text or not text.strip():
            return ToolResult(
                success=False,
                error="输入文本为空",
                metadata={"elapsed_ms": 0},
            )

        cfg = self._config
        prompt_type = "short" if len(text.strip()) < cfg.SHORT_TEXT_LEN else "long"
        max_retries = cfg.MAX_RETRIES
        last_error = None
        attempt_logs: List[Dict[str, Any]] = []

        for attempt in range(max_retries + 1):
            attempt_record: Dict[str, Any] = {
                "attempt": attempt + 1,
                "status": "unknown",
            }
            try:
                raw, finish_reason, usage = self._call_llm(text, attempt, cfg)

                attempt_record["raw_output_length"] = len(raw)
                attempt_record["finish_reason"] = finish_reason
                attempt_record["usage"] = usage

                json_str, thinking_text = extract_json_and_thinking(raw)

                if json_str is None and finish_reason == "length":
                    logger.info("finish_reason=length，尝试修复被截断的 JSON")
                    repaired = try_repair_truncated_json(raw)
                    if repaired:
                        json_str = repaired
                        thinking_text = ""
                        attempt_record["json_repaired"] = True

                if json_str is None:
                    raw_preview = raw[:200] + ("..." if len(raw) > 200 else "")
                    truncation_hint = ""
                    if finish_reason == "length":
                        truncation_hint = (
                            "（模型输出被 max_tokens 截断，JSON 不完整。"
                            f"当前 TOOL_LLM_MAX_TOKENS={cfg.TOOL_LLM_MAX_TOKENS}，"
                            "建议增大该值或缩短评论文本）"
                        )
                    last_error = f"无法从模型输出中提取 JSON{truncation_hint}，输出片段: {raw_preview}"
                    attempt_record["status"] = "json_extract_failed"
                    attempt_record["error"] = last_error
                    attempt_logs.append(attempt_record)
                    logger.warning(
                        "关键词提取无法提取 JSON (尝试 %d/%d)",
                        attempt + 1, max_retries + 1,
                    )
                    continue

                attempt_record["extracted_json"] = json_str
                attempt_record["thinking_text_length"] = len(thinking_text)

                keywords_json = json.loads(json_str)
                keywords_data = extract_keywords_from_json(keywords_json)
                keywords_data = normalize_keywords_data(keywords_data, json_format="new")

                processed = post_process_keywords_with_config(
                    keywords_data,
                    config=cfg,
                    original_text=text,
                    max_keywords=max_keywords,
                )

                if processed:
                    elapsed = round((time.time() - start) * 1000, 2)
                    data: Dict[str, Any] = {"keywords": processed}
                    if thinking_text:
                        data["thinking"] = thinking_text
                    attempt_record["status"] = "success"
                    attempt_record["keywords_count"] = len(processed)
                    attempt_logs.append(attempt_record)
                    return ToolResult(
                        success=True,
                        data=data,
                        metadata={
                            "elapsed_ms": elapsed,
                            "prompt_type": prompt_type,
                            "attempts": attempt + 1,
                            "model": cfg.TOOL_LLM_MODEL,
                        },
                    )

                last_error = "LLM 返回结果经后处理后为空"
                attempt_record["status"] = "post_process_empty"
                attempt_record["error"] = last_error

            except json.JSONDecodeError as e:
                last_error = f"JSON 解析错误: {e}"
                attempt_record["status"] = "json_decode_error"
                attempt_record["error"] = last_error
                logger.warning("关键词提取 JSON 解析失败 (尝试 %d/%d): %s",
                               attempt + 1, max_retries + 1, e)
            except Exception as e:
                last_error = str(e)
                attempt_record["status"] = "exception"
                attempt_record["error"] = last_error
                logger.warning("关键词提取异常 (尝试 %d/%d): %s",
                               attempt + 1, max_retries + 1, e)

            attempt_logs.append(attempt_record)

        elapsed = round((time.time() - start) * 1000, 2)
        return ToolResult(
            success=False,
            error=f"经 {max_retries + 1} 次尝试后仍失败: {last_error}",
            metadata={
                "elapsed_ms": elapsed,
                "attempts": max_retries + 1,
            },
        )
