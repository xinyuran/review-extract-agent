"""
文本预处理主入口

提供 preprocess_comment() 和 advanced_preprocess() 两个入口函数，
内部组合 cleaners 子模块完成各项清洗。
"""

import re

from .cleaners import (
    remove_time_expressions,
    remove_date_expressions,
    remove_dates,
    remove_urls,
    remove_emails,
    remove_phone_numbers,
    remove_emojis,
    remove_garbled_text,
    remove_special_symbols,
    normalize_numbers,
    remove_whitespace_chars,
    keep_chinese_only,
    clean_text,
)

# Re-export all cleaners for backward compatibility
__all__ = [
    "preprocess_comment",
    "advanced_preprocess",
    "remove_time_expressions",
    "remove_date_expressions",
    "remove_dates",
    "remove_urls",
    "remove_emails",
    "remove_phone_numbers",
    "remove_emojis",
    "remove_garbled_text",
    "remove_special_symbols",
    "normalize_numbers",
    "remove_whitespace_chars",
    "keep_chinese_only",
    "clean_text",
]


def preprocess_comment(comment, 
                       remove_english=True,
                       deduplicate_punctuation=True,
                       remove_html_entities=True,
                       normalize_whitespace=True,
                       remove_control_chars=True,
                       remove_dates_flag=True,
                       keep_chinese_only_flag=True,
                       keep_numbers=True,
                       keep_chinese_punctuation=True,
                       remove_whitespace_chars_flag=True,
                       max_length=None):
    """
    评论预处理函数（包含清洗和长度截断）
    """
    if not isinstance(comment, str):
        comment = str(comment)

    if remove_whitespace_chars_flag:
        comment = remove_whitespace_chars(comment)

    if remove_dates_flag:
        comment = remove_dates(comment)

    if keep_chinese_only_flag:
        comment = keep_chinese_only(
            comment,
            keep_numbers=keep_numbers,
            keep_chinese_punctuation=keep_chinese_punctuation
        )
    else:
        comment = clean_text(
            comment,
            remove_english=remove_english,
            deduplicate_punctuation=deduplicate_punctuation,
            remove_html_entities=remove_html_entities,
            normalize_whitespace=normalize_whitespace,
            remove_control_chars=remove_control_chars
        )

    if deduplicate_punctuation and not keep_chinese_only_flag:
        comment = re.sub(r'([!！？?。.，,、；;：:""\"\'\'（）()【】\[\]《》<>…~～@#￥$%^&*_+\-=｜|/\\])\1+', r'\1', comment)
    elif deduplicate_punctuation and keep_chinese_only_flag:
        comment = re.sub(r'([，。！？；：""''（）【】《》、…—·])\1+', r'\1', comment)

    if normalize_whitespace:
        comment = re.sub(r'\s+', ' ', comment)
        comment = comment.strip()

    if max_length is not None and len(comment) > max_length:
        comment = comment[:max_length]

    return comment


def advanced_preprocess(text, 
                       remove_urls_flag=True,
                       remove_emails_flag=True,
                       remove_phones_flag=True,
                       normalize_numbers_flag=True,
                       remove_emojis_flag=True,
                       remove_garbled_flag=True,
                       remove_special_symbols_flag=True):
    """
    高级预处理（可选功能）
    """
    if remove_garbled_flag:
        text = remove_garbled_text(text)
    if remove_urls_flag:
        text = remove_urls(text)
    if remove_emails_flag:
        text = remove_emails(text)
    if remove_phones_flag:
        text = remove_phone_numbers(text)
    if remove_emojis_flag:
        text = remove_emojis(text)
    if remove_special_symbols_flag:
        text = remove_special_symbols(text)
    if normalize_numbers_flag:
        text = normalize_numbers(text)
    return text
