"""
独立测试脚本：验证 vLLM 是否正确支持 tool_choice="auto"

测试内容：
1. 不带 tools 的普通请求（基准对照）
2. 带 tools + tool_choice="auto" 的请求（核心测试）
3. 带 tools + tool_choice="none" 的请求（对照）

使用方法：
    cd extract_agent
    python examples/test_tool_calling.py
"""

import sys
import os
import json
import time

_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_ROOT = os.path.dirname(_EXAMPLES_DIR)
_REPO_ROOT = os.path.dirname(_PACKAGE_ROOT)
sys.path.insert(0, _REPO_ROOT)

from openai import OpenAI

BASE_URL = os.getenv("AGENT_LLM_BASE_URL", "http://192.168.12.42:8001/v1")
API_KEY = os.getenv("AGENT_LLM_API_KEY", "dummy")

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

# 先查询 vLLM 上可用的模型名称
print("=" * 60)
print("Step 0: 查询 vLLM 可用模型")
print("=" * 60)
try:
    models = client.models.list()
    for m in models.data:
        print(f"  模型ID: {m.id}")
    MODEL_ID = models.data[0].id
    print(f"\n将使用模型: {MODEL_ID}")
except Exception as e:
    print(f"  查询模型列表失败: {e}")
    print("  请检查 vLLM 服务是否启动、BASE_URL 是否正确")
    sys.exit(1)

SIMPLE_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "获取指定城市的天气",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "城市名称",
                }
            },
            "required": ["city"],
        },
    },
}

def test_request(label, messages, tools=None, tool_choice=None, timeout=30):
    print(f"\n{'=' * 60}")
    print(f"测试: {label}")
    print(f"{'=' * 60}")
    kwargs = {
        "model": MODEL_ID,
        "messages": messages,
        "max_tokens": 256,
        "temperature": 0,
    }
    if tools is not None:
        kwargs["tools"] = tools
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice

    print(f"  参数: tools={'有' if tools else '无'}, tool_choice={tool_choice}")
    start = time.time()
    try:
        resp = client.chat.completions.create(**kwargs, timeout=timeout)
        elapsed = time.time() - start
        msg = resp.choices[0].message
        print(f"  状态: 成功 ({elapsed:.2f}s)")
        print(f"  finish_reason: {resp.choices[0].finish_reason}")
        if msg.content:
            print(f"  content: {msg.content[:200]}")
        if msg.tool_calls:
            print(f"  tool_calls ({len(msg.tool_calls)}):")
            for tc in msg.tool_calls:
                print(f"    - {tc.function.name}({tc.function.arguments})")
        return True
    except Exception as e:
        elapsed = time.time() - start
        print(f"  状态: 失败 ({elapsed:.2f}s)")
        print(f"  错误: {e}")
        return False


messages_simple = [{"role": "user", "content": "你好"}]
messages_weather = [{"role": "user", "content": "北京今天天气怎么样？"}]

# 测试 1：普通请求（基准）
test_request(
    "普通请求（无 tools）",
    messages=messages_simple,
)

# 测试 2：tool_choice="none"（对照）
test_request(
    'tool_choice="none"',
    messages=messages_weather,
    tools=[SIMPLE_TOOL],
    tool_choice="none",
)

# 测试 3：tool_choice="auto"（核心）
test_request(
    'tool_choice="auto"（核心测试）',
    messages=messages_weather,
    tools=[SIMPLE_TOOL],
    tool_choice="auto",
    timeout=60,
)

# 测试 4：多个 tools + auto
MULTI_TOOLS = [
    SIMPLE_TOOL,
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "搜索信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"}
                },
                "required": ["query"],
            },
        },
    },
]

test_request(
    '多工具 + tool_choice="auto"',
    messages=messages_weather,
    tools=MULTI_TOOLS,
    tool_choice="auto",
    timeout=60,
)

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)
