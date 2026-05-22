"""
情感分析工具

调用 Tool LLM（SFT+DPO+GRPO 微调模型）进行情绪分析，
使用与微调训练时一致的 Prompt 模板。

微调模型输出格式为三元组：
  {"sentiment": [["推理说明", "情绪类别(正向/中立/负向)", 置信概率]]}

本工具负责将其转换为 Agent 统一的结构化格式：
  {"label": "positive/negative/neutral", "confidence": 0.95, "reasoning": "..."}
"""

import json
import logging
import re
import time
from typing import Any, Dict, Optional

from openai import OpenAI

from .base_tool import BaseTool, ToolResult
from ..config import AgentConfig
from ..utils.json_parser import extract_json_from_response

logger = logging.getLogger(__name__)


def _get_sentiment_prompt(comment: str) -> tuple:
    system_prompt = """你是一名中文电商评论情绪识别专家。你的任务是对给定的评论文本进行**逐步推理**，最终输出评论整体的**情绪类别**（正向 / 中立 / 负向）以及该判断的**置信概率**。

核心规则1：必须基于评论内容的真实语义与事实做判断，不得臆测或无中生有。
核心规则2：**动态推理原则（防止过度解读）**。根据评论的复杂度采用不同的判断标准：
   - **直白评论**：若评论仅由形容词、重复词或简单短句构成，**直接采信其字面含义**，无需强行寻找事实支撑。
   - **复杂评论**：若评论包含具体事实描述且与语气存在冲突，则以**事实对体验的影响**为准。
核心规则3：**电商语境补全**。对于语法缺失、无主语的口语，需还原买家真实的抱怨或夸赞意图，而非死扣语法。

【情绪判断逻辑】
1. **显式情绪识别（态度优先）**：
   - 正向：包括强烈的赞美（完美、很好），也包括**弱正向/及格**的表述（如：可以、还行、内容丰富、符合要求、没毛病）。**重复的褒义字（如：好好好）视为强烈正向。**
   - 负向：表示不满、抱怨、失望等。
   - 中立：只有客观描述（参数、颜色、尺码、时间、场景等），没有表态，**且无任何**主观评价词。

2. **隐式情绪与冲突修正（事实定性 - 关键）**：
   - 若评论没有明显态度词，或者态度词与事实冲突，需判断**事实本身对购物体验的影响**。
   - **负面事实通用规则**：凡是描述了**商品状态异常**（如：二手/被退过、破损/脏污）、**履约失败**（如：缺件、漏发、错发）、**使用体验受损**（如：失效、异物、甚至安全隐患）等等的事实，**无论语气多么平淡/客观**，均判定为**负向**。
   - **正面事实通用规则**：描述问题得到解决、或达到使用目的。

4. **反讽/反语检测启发式**
   - 当出现表面褒义词但同时有下列任一情形时，应将反讽/反语作为高怀疑项，并在推理步骤明确说明理由（之后再按事实优先或子句聚合判定）：
     - 表面褒义与"典型负面语义"（如"太小/很少/不够/便宜""有异物/二手"等）并列，且两者语义冲突；或
     - 出现夸张语气、反问、强烈标点（"！"）、或带有"其实/倒是/竟然"之类转折词；或
     - 商品属性在该语境下通常为负面（例如"包很小"通常不应被视作正向，除非明确表达"正是我需要的小"）。
   - 如果反讽线索明显，倾向把整体情绪向与事实一致的方向判定（并在推理里标注"反讽判定"）。

5. **最终判定路径**：
   - **路径A（直白类）**：显式态度明确，且无相反事实 → **直接采信显式态度**（避免过度推理）。
   - **路径B（事实类/冲突类）**：
     - 描述了符合"负面事实通用规则"的情况 → **负向**（事实 > 无情绪词）。
     - 若出现"表面褒义 + 明显负面事实或异常搭配" → **负向**（反讽）。
   - **路径C（语境补全）**：对于残缺句式，需还原买家真实的抱怨或夸赞意图，并按照真实意图判断。


【输出分两步】
**第一步：推理步骤（逐条展示**推理**过程）**
- "手机质量很好，物流也很快，真是一次满意的购物体验。" → 路径A：显式褒义词强烈，无冲突 → 最终情绪：正向。

- "好好好好好好好好好好" → 路径A：用户重复褒义词，意图直白明确 → 最终情绪：正向。

- "别人退的直接又发出" → 路径B：虽然无贬义词，但"退货重发"属于**把二手当新品卖**，严重违背购物契约（负面事实通则） → 最终情绪：负向。

- "电源没有差评" → 路径C：典型电商歧义句，基于常识补全为"电源[没收到]，[给]差评" → 最终情绪：负向。

- "能吃出纸是真的牛" → 路径B：字面夸赞与"食品中有异物"（安全隐患）冲突，判定为反讽 → 最终情绪：负向。

- "舒适度：可以，主要是镂空的。" → 路径A："可以"属于及格认可（弱正向），镂空是客观属性 → 最终情绪：正向。

- "有效果，一停立马复发" → 路径B：事实承认"有效果"，复发属于停药后的自然现象，未违背产品承诺 → 最终情绪：正向。


**第二步：输出严格 JSON，格式如下（不要添加其他内容）：**
- 必须包含键"sentiment"，其值为数组。
- 数组中的每个元素是一个三元组：["推理步骤概要说明", "情绪类别(正向/中立/负向)", 概率(0~1之间，保留两位小数)]，三元组之间不要有任何的除了用于分割的逗号之外的符号。
- 三元组中的"推理步骤说明"字段用一句话概括整体推理策略。

{
  "sentiment": [
    ["推理步骤说明", "情绪类别", 置信概率(0~1之间小数，保留两位)],
    ...
  ]
}"""
    user_prompt = f"""请分析以下评论的情绪：

【待判断评论】
{comment}
"""
    return system_prompt, user_prompt


_SENTIMENT_JSON_PATTERN = re.compile(
    r"\{[^{}]*\"sentiment\"\s*:\s*\[.*?\]\s*\}", re.DOTALL
)

_LABEL_MAP = {
    "正向": "positive",
    "负向": "negative",
    "中立": "neutral",
    "positive": "positive",
    "negative": "negative",
    "neutral": "neutral",
}


class SentimentTool(BaseTool):

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
        return "sentiment_analyze"

    @property
    def description(self) -> str:
        return (
            "分析中文电商评论的情感倾向。"
            "返回情感标签（positive/negative/neutral）、置信度和推理说明。"
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "待分析情感的评论文本",
                },
            },
            "required": ["text"],
        }

    @staticmethod
    def _parse_sentiment_response(result: Dict[str, Any]) -> Dict[str, Any]:
        """
        将微调模型的三元组输出转换为 Agent 统一格式。

        微调模型可能输出两种格式：
          嵌套: {"sentiment": [["推理", "正向/中立/负向", 0.95]]}
          扁平: {"sentiment": ["推理", "正向/中立/负向", 0.95]}
        统一转换为: {"label": "positive", "confidence": 0.95, "reasoning": "..."}
        """
        sentiment_data = result.get("sentiment")

        if isinstance(sentiment_data, list) and len(sentiment_data) > 0:
            triple = sentiment_data[0]
            if isinstance(triple, list) and len(triple) >= 3:
                reasoning = str(triple[0])
                raw_label = str(triple[1])
                confidence = float(triple[2])
                label = _LABEL_MAP.get(raw_label, "neutral")
                return {
                    "label": label,
                    "confidence": confidence,
                    "reasoning": reasoning,
                }

            if isinstance(triple, str) and len(sentiment_data) >= 3:
                reasoning = str(sentiment_data[0])
                raw_label = str(sentiment_data[1])
                try:
                    confidence = float(sentiment_data[2])
                except (ValueError, TypeError):
                    confidence = 0.5
                label = _LABEL_MAP.get(raw_label, "neutral")
                return {
                    "label": label,
                    "confidence": confidence,
                    "reasoning": reasoning,
                }

        if isinstance(sentiment_data, str):
            label = _LABEL_MAP.get(sentiment_data, sentiment_data)
        else:
            label = _LABEL_MAP.get(
                result.get("label", "neutral"), "neutral"
            )
        confidence = float(result.get("confidence", 0.5))
        reasoning = result.get("reasoning", "")

        return {
            "label": label,
            "confidence": confidence,
            "reasoning": reasoning,
        }

    def execute(self, **kwargs) -> ToolResult:
        text = kwargs.get("text", "")
        start = time.time()

        if not text or not text.strip():
            return ToolResult(
                success=False,
                error="输入文本为空",
                metadata={"elapsed_ms": 0},
            )

        cfg = self._config

        try:
            if self._llm_service:
                raw = self._call_via_llm_service(text, cfg)
            else:
                raw = self._call_via_direct_client(text, cfg)

            json_str = extract_json_from_response(raw, pattern=_SENTIMENT_JSON_PATTERN)
            if json_str is None:
                raise json.JSONDecodeError("无法从模型输出中提取 JSON", raw, 0)

            result = json.loads(json_str)
            elapsed = round((time.time() - start) * 1000, 2)

            parsed = self._parse_sentiment_response(result)

            return ToolResult(
                success=True,
                data=parsed,
                metadata={"elapsed_ms": elapsed, "model": cfg.TOOL_LLM_MODEL},
            )

        except json.JSONDecodeError as e:
            logger.warning("情感分析 JSON 解析失败: %s", e)
            return ToolResult(
                success=False,
                error=f"JSON 解析错误: {e}",
                metadata={"elapsed_ms": round((time.time() - start) * 1000, 2)},
            )
        except Exception as e:
            logger.exception("情感分析工具执行异常")
            return ToolResult(
                success=False,
                error=str(e),
                metadata={"elapsed_ms": round((time.time() - start) * 1000, 2)},
            )

    def _call_via_llm_service(self, text: str, cfg) -> str:
        """通过 LLM Service + Skill 调用情感分析"""
        extra_params: Dict[str, Any] = {
            "temperature": 0,
            "seed": cfg.TOOL_LLM_SEED,
        }
        if cfg.TOOL_LLM_RESPONSE_FORMAT:
            extra_params["response_format"] = cfg.TOOL_LLM_RESPONSE_FORMAT

        llm_resp = self._llm_service.call_tool(
            skill_name="sentiment_analyze",
            variables={"comment": text},
            extra_params=extra_params,
        )
        return (llm_resp.content or "").strip()

    def _call_via_direct_client(self, text: str, cfg) -> str:
        """通过直接 OpenAI client 调用情感分析（向后兼容）"""
        system_prompt, user_prompt = _get_sentiment_prompt(text)

        client = self._get_client()
        resp = client.chat.completions.create(
            model=cfg.TOOL_LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=cfg.TOOL_LLM_RESPONSE_FORMAT,
            max_tokens=cfg.TOOL_LLM_MAX_TOKENS,
            temperature=0,
            seed=cfg.TOOL_LLM_SEED,
        )
        return resp.choices[0].message.content.strip()
