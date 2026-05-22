"""关键词长度过滤"""

import re


def filter_by_length(keywords_data, keyword_idx=1, max_length=6):
    """过滤超过最大长度的关键词"""
    return [
        item for item in keywords_data
        if len(item) > keyword_idx
        and isinstance(item[keyword_idx], str)
        and len(item[keyword_idx]) <= max_length
    ]


def filter_english(keywords_data, keyword_idx=1):
    """过滤包含英文字母的关键词"""
    return [
        item for item in keywords_data
        if len(item) > keyword_idx
        and isinstance(item[keyword_idx], str)
        and not re.search(r'[a-zA-Z]', item[keyword_idx])
    ]
