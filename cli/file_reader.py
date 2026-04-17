"""
文件输入解析 — 支持 txt / csv / json 格式读取评论
"""

import csv
import json
from pathlib import Path
from typing import List, Optional


class FileReadError(Exception):
    """文件读取异常"""
    pass


def read_comments_from_file(
    path: str,
    column: Optional[str] = None,
    field: Optional[str] = None,
) -> List[str]:
    """
    从文件读取评论列表。

    根据文件扩展名自动选择解析方式：
    - .txt: 每行一条评论（跳过空行）
    - .csv: 需指定 column 参数（列名）
    - .json: 需指定 field 参数（字段名），格式为 [{field: text}, ...]
    """
    filepath = Path(path)

    if not filepath.is_file():
        raise FileReadError(f"文件不存在: {path}")

    ext = filepath.suffix.lower()

    if ext == ".txt":
        return _read_txt(filepath)
    elif ext == ".csv":
        return _read_csv(filepath, column)
    elif ext == ".json":
        return _read_json(filepath, field)
    else:
        raise FileReadError(
            f"不支持的文件格式: {ext}（支持 .txt / .csv / .json）"
        )


def _read_txt(filepath: Path) -> List[str]:
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    comments = [line.strip() for line in lines if line.strip()]
    if not comments:
        raise FileReadError(f"文件为空: {filepath}")
    return comments


def _read_csv(filepath: Path, column: Optional[str]) -> List[str]:
    if not column:
        raise FileReadError("CSV 文件需要指定 --column 参数来指定评论所在列名")

    comments = []
    with open(filepath, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames and column not in reader.fieldnames:
            raise FileReadError(
                f"CSV 中不存在列 '{column}'，"
                f"可用列: {', '.join(reader.fieldnames)}"
            )
        for row in reader:
            text = row.get(column, "").strip()
            if text:
                comments.append(text)

    if not comments:
        raise FileReadError(f"CSV 文件中列 '{column}' 没有有效数据")
    return comments


def _read_json(filepath: Path, field: Optional[str]) -> List[str]:
    if not field:
        raise FileReadError("JSON 文件需要指定 --field 参数来指定评论字段名")

    with open(filepath, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise FileReadError(f"JSON 解析失败: {e}")

    if not isinstance(data, list):
        raise FileReadError("JSON 文件应为数组格式: [{field: text}, ...]")

    comments = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        text = item.get(field, "")
        if isinstance(text, str) and text.strip():
            comments.append(text.strip())

    if not comments:
        raise FileReadError(f"JSON 文件中字段 '{field}' 没有有效数据")
    return comments
