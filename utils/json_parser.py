"""
LLM 响应 JSON 提取工具集

统一处理 LLM 输出中的 JSON 提取、修复和解析，解决以下常见问题：
- 模型输出包含 thinking 文本 + JSON 混合
- markdown ```json``` 代码块包裹
- 中文标点（逗号、冒号）混入 JSON
- max_tokens 截断导致 JSON 不完整
- 嵌套数组中存在非法尾部元素
"""

import json
import re
from typing import Optional, Tuple

_KEYWORDS_JSON_PATTERN = re.compile(
    r'\{\s*"keywords"\s*:\s*\[', re.DOTALL
)


def find_matching_brace(text: str, start: int) -> int:
    """
    从 start 位置的 '{' 开始，找到与之匹配的 '}'。
    正确处理嵌套和字符串引号内的转义。

    Returns:
        匹配的 '}' 的索引，未找到返回 -1。
    """
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == '\\' and in_string:
            escape = True
            continue
        if ch == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return i
    return -1


def sanitize_json(candidate: str) -> str:
    """
    修复微调模型 JSON 输出中常见的格式错误：
    1. 中文逗号 -> 英文逗号
    2. 中文冒号 -> 英文冒号（仅在 JSON 结构位置）
    3. 移除 keywords 数组中不合法的尾部元素
    """
    fixed = candidate.replace('，', ',').replace('：', ':')

    try:
        json.loads(fixed)
        return fixed
    except json.JSONDecodeError:
        pass

    kw_match = re.search(r'"keywords"\s*:\s*\[', fixed)
    if not kw_match:
        return fixed

    arr_start = kw_match.end() - 1
    depth = 0
    in_str = False
    esc = False
    last_valid_end = -1
    elem_start = -1

    for i in range(arr_start, len(fixed)):
        ch = fixed[i]
        if esc:
            esc = False
            continue
        if ch == '\\' and in_str:
            esc = True
            continue
        if ch == '"' and not esc:
            in_str = not in_str
            continue
        if in_str:
            continue

        if ch == '[':
            depth += 1
            if depth == 2:
                elem_start = i
        elif ch == ']':
            depth -= 1
            if depth == 1 and elem_start != -1:
                elem_text = fixed[elem_start:i + 1]
                try:
                    json.loads(elem_text)
                    last_valid_end = i
                except json.JSONDecodeError:
                    pass
                elem_start = -1
            elif depth == 0:
                break

    if last_valid_end > 0:
        rebuilt = fixed[:last_valid_end + 1].rstrip()
        if rebuilt.endswith(','):
            rebuilt = rebuilt[:-1]
        rebuilt += '\n  ]\n}'
        try:
            json.loads(rebuilt)
            return rebuilt
        except json.JSONDecodeError:
            pass

    return fixed


def _try_load(candidate: str) -> Optional[str]:
    """尝试解析 JSON 字符串。先原样解析，失败则 sanitize 修复后重试。"""
    try:
        json.loads(candidate)
        return candidate
    except json.JSONDecodeError:
        pass

    sanitized = sanitize_json(candidate)
    try:
        json.loads(sanitized)
        return sanitized
    except json.JSONDecodeError:
        return None


def extract_json_from_response(
    raw: str,
    pattern: Optional[re.Pattern] = None,
) -> Optional[str]:
    """
    从 LLM 的混合输出中提取 JSON 部分。

    策略（按优先级）：
    1. 剥离 markdown 代码块后整段解析
    2. 使用自定义 pattern 正则匹配（如 sentiment JSON 格式）
    3. 从前往后扫描第一个 '{' 到最后一个 '}'

    Args:
        raw: LLM 原始输出
        pattern: 可选的正则表达式，用于匹配特定 JSON 结构

    Returns:
        提取到的 JSON 字符串，失败返回 None。
    """
    text = raw.strip()

    if text.startswith("```json"):
        text = text.replace("```json", "").replace("```", "").strip()
    elif text.startswith("```"):
        text = text.replace("```", "").strip()

    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    if pattern:
        match = pattern.search(raw)
        if match:
            candidate = match.group(0)
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                pass

    last_brace = raw.rfind("}")
    if last_brace == -1:
        return None
    for i in range(len(raw)):
        if raw[i] == "{":
            candidate = raw[i:last_brace + 1]
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                continue

    return None


def extract_json_and_thinking(raw: str) -> Tuple[Optional[str], str]:
    """
    从模型输出中提取 JSON 部分和思考文本。

    策略（按优先级）：
    1. 整段直接解析为 JSON
    2. 剥离 markdown 代码块后解析
    3. 找到 {"keywords": [ 开头，用括号匹配定位完整 JSON
    4. 从后往前找最后一个 } 对应的 {，尝试解析

    每步都先原样解析，失败则用 sanitize_json 修复后重试。

    Returns:
        (json_str, thinking_text) 元组。
        json_str 为 None 表示提取失败；
        thinking_text 为 JSON 之前的推理文本（可能为空字符串）。
    """
    text = raw.strip()

    result = _try_load(text)
    if result is not None:
        return result, ""

    cleaned = text
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()
    if cleaned != text:
        result = _try_load(cleaned)
        if result is not None:
            return result, ""

    match = _KEYWORDS_JSON_PATTERN.search(raw)
    if match:
        brace_start = match.start()
        brace_end = find_matching_brace(raw, brace_start)
        if brace_end != -1:
            candidate = raw[brace_start:brace_end + 1]
            result = _try_load(candidate)
            if result is not None:
                thinking = raw[:brace_start].strip()
                return result, thinking

    last_brace = raw.rfind("}")
    if last_brace == -1:
        return None, raw.strip()

    for i in range(len(raw) - 1, -1, -1):
        if raw[i] == "{":
            candidate = raw[i:last_brace + 1]
            result = _try_load(candidate)
            if result is not None:
                thinking = raw[:i].strip()
                return result, thinking

    return None, raw.strip()


def try_repair_truncated_json(raw: str) -> Optional[str]:
    """
    当 finish_reason == "length" 导致 JSON 被截断时，
    尝试找到 {"keywords": [...]} 的起始位置，然后补全缺失的括号。
    只处理最常见的截断情况：数组元素被截断。
    """
    match = _KEYWORDS_JSON_PATTERN.search(raw)
    if not match:
        return None

    start = match.start()
    fragment = raw[start:]

    depth_brace = 0
    depth_bracket = 0
    in_string = False
    escape = False

    for ch in fragment:
        if escape:
            escape = False
            continue
        if ch == '\\' and in_string:
            escape = True
            continue
        if ch == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            depth_brace += 1
        elif ch == '}':
            depth_brace -= 1
        elif ch == '[':
            depth_bracket += 1
        elif ch == ']':
            depth_bracket -= 1

    last_complete_bracket = fragment.rfind(']')
    if last_complete_bracket == -1:
        last_bracket_open = fragment.rfind('[')
        if last_bracket_open != -1:
            repaired = fragment[:last_bracket_open + 1] + ']}'
            try:
                json.loads(repaired)
                return repaired
            except json.JSONDecodeError:
                pass
        return None

    last_comma = fragment.rfind(',', 0, last_complete_bracket)
    if last_comma != -1:
        candidate = fragment[:last_complete_bracket + 1]
        candidate = candidate.rstrip().rstrip(',')
        candidate += ']}'
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass

    for pos in range(len(fragment) - 1, match.end() - match.start() - 1, -1):
        ch = fragment[pos]
        if ch == ']':
            suffix = fragment[:pos + 1] + '}'
            try:
                json.loads(suffix)
                return suffix
            except json.JSONDecodeError:
                candidate = fragment[:pos + 1].rstrip().rstrip(',') + ']}'
                try:
                    json.loads(candidate)
                    return candidate
                except json.JSONDecodeError:
                    continue

    repair = fragment.rstrip()
    if repair.endswith(','):
        repair = repair[:-1]
    repair += ']' * max(0, depth_bracket) + '}' * max(0, depth_brace)
    try:
        json.loads(repair)
        return repair
    except json.JSONDecodeError:
        pass

    return None
