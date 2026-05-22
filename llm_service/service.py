"""
LLM Service 统一调用层

收归所有 LLM 调用（Agent LLM 与 Tool LLM），提供：
- call_agent(): Agent LLM 调用（ReAct 循环中使用）
- call_tool(): Tool LLM 调用（通过 SkillLoader 自动构建 messages）
- detect_tool_calling_mode(): 探测 Function Calling 支持
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

from ..config import AgentConfig
from .models import LLMResponse, LLMStreamChunk, SkillPrompt
from .skill_loader import SkillLoader

logger = logging.getLogger(__name__)


class LLMService:
    """统一管理 Agent LLM 与 Tool LLM 的调用"""

    def __init__(
        self,
        config: AgentConfig | None = None,
        skill_loader: SkillLoader | None = None,
    ):
        self.config = config or AgentConfig()
        self.skill_loader = skill_loader or SkillLoader()

        self._agent_client: Optional[OpenAI] = None
        self._tool_client: Optional[OpenAI] = None
        self._native_tool_calling: Optional[bool] = None

        backend = self.config.get_backend_mode()
        if backend != "offline":
            self._agent_client = OpenAI(
                base_url=self.config.AGENT_LLM_BASE_URL,
                api_key=self.config.AGENT_LLM_API_KEY,
            )
            self._tool_client = OpenAI(
                base_url=self.config.TOOL_LLM_BASE_URL,
                api_key=self.config.TOOL_LLM_API_KEY,
            )

        self._trajectory_recorder = None

    def set_trajectory_recorder(self, recorder: Any) -> None:
        self._trajectory_recorder = recorder

    # ------------------------------------------------------------------
    # Agent LLM 调用
    # ------------------------------------------------------------------

    def call_agent(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
    ) -> LLMResponse:
        """
        调用 Agent LLM。

        Args:
            messages: OpenAI messages 格式列表
            tools: 工具定义列表 (native FC 模式)
            tool_choice: 工具选择策略 ("auto" / "none" / ...)
        """
        if self._agent_client is None:
            raise RuntimeError("Agent LLM client not initialized (offline mode)")

        cfg = self.config
        kwargs: Dict[str, Any] = {
            "model": cfg.AGENT_LLM_MODEL,
            "messages": messages,
            "max_tokens": cfg.AGENT_LLM_MAX_TOKENS,
            "temperature": cfg.AGENT_LLM_TEMPERATURE,
        }
        if tools:
            kwargs["tools"] = tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice

        start = time.time()
        resp = self._agent_client.chat.completions.create(**kwargs)
        elapsed_ms = round((time.time() - start) * 1000, 2)

        message = resp.choices[0].message
        tool_calls_data = None
        if message.tool_calls:
            tool_calls_data = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]

        usage_data = None
        if resp.usage:
            usage_data = {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "total_tokens": resp.usage.total_tokens,
            }

        llm_resp = LLMResponse(
            content=message.content,
            tool_calls=tool_calls_data,
            role=message.role,
            raw=resp,
            model=resp.model,
            usage=usage_data,
            finish_reason=resp.choices[0].finish_reason,
        )

        if self._trajectory_recorder:
            self._trajectory_recorder.record_agent_turn(
                messages=messages,
                response=llm_resp,
                elapsed_ms=elapsed_ms,
            )

        return llm_resp

    # ------------------------------------------------------------------
    # Agent LLM 流式调用
    # ------------------------------------------------------------------

    def call_agent_stream(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
    ):
        """
        流式调用 Agent LLM，逐 chunk yield LLMStreamChunk。

        流结束后额外 yield 一个 finish_reason 非 None 的终止 chunk。
        调用方可通过累积 delta_content 拼接完整响应，
        或通过 tool_call 相关字段累积工具调用参数。
        """
        if self._agent_client is None:
            raise RuntimeError("Agent LLM client not initialized (offline mode)")

        cfg = self.config
        kwargs: Dict[str, Any] = {
            "model": cfg.AGENT_LLM_MODEL,
            "messages": messages,
            "max_tokens": cfg.AGENT_LLM_MAX_TOKENS,
            "temperature": cfg.AGENT_LLM_TEMPERATURE,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice

        stream = self._agent_client.chat.completions.create(**kwargs)

        for chunk in stream:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta

            sc = LLMStreamChunk(
                finish_reason=choice.finish_reason,
                model=chunk.model,
            )

            if delta.content:
                sc.delta_content = delta.content

            if delta.tool_calls:
                tc = delta.tool_calls[0]
                sc.tool_call_index = tc.index
                if tc.id:
                    sc.tool_call_id = tc.id
                if tc.function:
                    if tc.function.name:
                        sc.tool_call_function_name = tc.function.name
                    if tc.function.arguments:
                        sc.tool_call_function_args_delta = tc.function.arguments

            yield sc

    # ------------------------------------------------------------------
    # Tool LLM 调用
    # ------------------------------------------------------------------

    def call_tool(
        self,
        skill_name: str,
        variables: Optional[Dict[str, Any]] = None,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        """
        调用 Tool LLM，通过 SkillLoader 自动构建 messages。

        Args:
            skill_name: 技能名称 (如 "keyword_extract_long")
            variables: 注入到 SKILL.md 模板中的变量
            extra_params: 额外的 API 参数 (如 seed, frequency_penalty 等)
        """
        if self._tool_client is None:
            raise RuntimeError("Tool LLM client not initialized (offline mode)")

        skill = self.skill_loader.load(skill_name, **(variables or {}))
        return self.call_tool_with_skill(skill, extra_params=extra_params)

    def call_tool_with_skill(
        self,
        skill: SkillPrompt,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        """使用已加载的 SkillPrompt 调用 Tool LLM"""
        if self._tool_client is None:
            raise RuntimeError("Tool LLM client not initialized (offline mode)")

        cfg = self.config
        messages = [
            {"role": "system", "content": skill.system},
            {"role": "user", "content": skill.user},
        ]

        kwargs: Dict[str, Any] = {
            "model": cfg.TOOL_LLM_MODEL,
            "messages": messages,
            "max_tokens": cfg.TOOL_LLM_MAX_TOKENS,
            "temperature": cfg.TOOL_LLM_TEMPERATURE,
            "seed": cfg.TOOL_LLM_SEED,
        }
        if extra_params:
            extra_body = extra_params.pop("extra_body", None)
            kwargs.update(extra_params)
            if extra_body:
                kwargs["extra_body"] = extra_body

        start = time.time()
        resp = self._tool_client.chat.completions.create(**kwargs)
        elapsed_ms = round((time.time() - start) * 1000, 2)

        usage_data = None
        if resp.usage:
            usage_data = {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "total_tokens": resp.usage.total_tokens,
            }

        llm_resp = LLMResponse(
            content=resp.choices[0].message.content,
            role="assistant",
            raw=resp,
            model=resp.model,
            usage=usage_data,
            finish_reason=resp.choices[0].finish_reason,
        )

        if self._trajectory_recorder:
            self._trajectory_recorder.record_tool_turn(
                skill_name=skill.name,
                messages=messages,
                response=llm_resp,
                elapsed_ms=elapsed_ms,
            )

        return llm_resp

    # ------------------------------------------------------------------
    # Function Calling 探测
    # ------------------------------------------------------------------

    def detect_tool_calling_mode(self) -> bool:
        """
        探测 vLLM 是否支持原生 tool calling。
        使用 httpx 直接发请求，设置严格超时。
        """
        if self._native_tool_calling is not None:
            return self._native_tool_calling

        mode_cfg = self.config.AGENT_TOOL_CALLING_MODE.lower()
        if mode_cfg == "native":
            self._native_tool_calling = True
        elif mode_cfg == "prompt":
            self._native_tool_calling = False
        else:
            self._native_tool_calling = self._probe_native_tool_calling()

        mode_name = "原生 Function Calling" if self._native_tool_calling else "Prompt-based"
        logger.info("Agent 工具调用模式: %s (配置=%s)", mode_name, mode_cfg)
        return self._native_tool_calling

    def _probe_native_tool_calling(self) -> bool:
        import httpx

        probe_timeout = 10
        url = f"{self.config.AGENT_LLM_BASE_URL}/chat/completions"
        payload = {
            "model": self.config.AGENT_LLM_MODEL,
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "_probe",
                    "description": "probe",
                    "parameters": {"type": "object", "properties": {}},
                },
            }],
            "tool_choice": "auto",
            "max_tokens": 1,
        }

        try:
            resp = httpx.post(
                url,
                json=payload,
                headers={"Authorization": f"Bearer {self.config.AGENT_LLM_API_KEY}"},
                timeout=probe_timeout,
            )
            if resp.status_code == 200:
                logger.info("探测成功：vLLM 支持原生 Function Calling")
                return True
            logger.info("探测返回 %d，降级为 prompt-based", resp.status_code)
            return False
        except Exception as e:
            logger.warning("探测 Function Calling 异常，降级为 prompt-based: %s", e)
            return False

    @property
    def is_native_tool_calling(self) -> Optional[bool]:
        return self._native_tool_calling

    # ------------------------------------------------------------------
    # Skill 层便捷方法
    # ------------------------------------------------------------------

    def load_skill(self, skill_name: str, **variables: Any) -> SkillPrompt:
        return self.skill_loader.load(skill_name, **variables)

    def build_tool_descriptions(self, tools: Dict[str, Any]) -> str:
        """将工具定义转为 Markdown 描述文本（用于 prompt-based 模式）"""
        from ..tools.base_tool import BaseTool
        parts: list[str] = []
        for tool in tools.values():
            if isinstance(tool, BaseTool):
                spec = tool.to_openai_tool()
            else:
                spec = tool
            fn = spec.get("function", spec)
            params = json.dumps(fn.get("parameters", {}), ensure_ascii=False, indent=2)
            parts.append(
                f"### `{fn.get('name', '?')}`\n{fn.get('description', '')}\n\n```json\n{params}\n```"
            )
        return "\n\n".join(parts)
