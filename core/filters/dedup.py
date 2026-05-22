"""关键词去重"""


def deduplicate_keywords(keywords_data, keyword_idx=1):
    """
    对关键词列表进行去重，保留第一次出现的（分数最高的，假设已排序）。

    Args:
        keywords_data: 关键词数据列表
        keyword_idx: 关键词在子列表中的索引

    Returns:
        去重后的关键词列表
    """
    seen = set()
    result = []
    for item in keywords_data:
        if len(item) > keyword_idx:
            kw = item[keyword_idx]
            if kw not in seen:
                seen.add(kw)
                result.append(item)
    return result
