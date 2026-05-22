"""
关键词后处理主入口

提供 post_process_keywords() / post_process_keywords_with_config() / 
extract_keywords_from_json() / normalize_keywords_data() 四个对外 API，
内部组合 filters 子模块完成去重、停用词过滤、长度过滤、原文对齐等。
"""

import logging
import re

from .filters import (
    find_min_span_in_text,
    validate_keyword_chars_in_text,
    filter_keywords_not_in_original,
    load_stopwords,
    is_date_keyword,
    is_time_keyword,
    deduplicate_keywords,
    filter_by_length,
    filter_english,
    CHINESE_NUM_PATTERN,
    ARABIC_NUM_PATTERN,
    ANY_NUM_PATTERN,
)

# Re-export for backward compatibility
__all__ = [
    "post_process_keywords",
    "post_process_keywords_with_config",
    "extract_keywords_from_json",
    "normalize_keywords_data",
    "find_min_span_in_text",
    "validate_keyword_chars_in_text",
    "filter_keywords_not_in_original",
    "load_stopwords",
    "is_date_keyword",
    "is_time_keyword",
]


def normalize_keywords_data(keywords_data, json_format="new"):
    """
    规范化关键词数据，确保分数为 float 类型，并验证关键词为字符串
    支持二元组和三元组格式的自动识别和转换
    
    Args:
        keywords_data: 原始关键词数据
            - 新格式三元组: [["推理", "关键词", 分数], ...]
            - 新格式二元组: [["关键词", 分数], ...]  # 缺少推理说明
            - 旧格式: [["关键词", 分数, "理由"], ...]
        json_format: JSON格式类型，"new"(新格式) 或 "old"(旧格式)
        
    Returns:
        规范化后的关键词数据（统一为三元组格式）
    """
    if not keywords_data:
        return []
    
    normalized_data = []
    for item in keywords_data:
        if not isinstance(item, list) or len(item) < 2:
            # 跳过格式不正确的数据（至少需要2个元素）
            continue
        
        # 根据长度和格式判断数据结构
        if json_format == "new":
            if len(item) == 3:
                # 标准三元组: ["推理", "关键词", 分数]
                reasoning, keyword, score = item[0], item[1], item[2]
            elif len(item) == 2:
                # 二元组: ["关键词", 分数] - 缺少推理说明
                # 需要判断哪个是关键词，哪个是分数
                if isinstance(item[0], str) and isinstance(item[1], (int, float, str)):
                    # 假设第一个是关键词，第二个是分数
                    reasoning = ""  # 空推理说明
                    keyword = item[0]
                    score = item[1]
                else:
                    # 格式不明确，跳过
                    continue
            else:
                # 长度不符合预期，跳过
                continue
        else:
            # 旧格式: ["关键词", 分数, "理由"]
            if len(item) >= 2:
                keyword = item[0]
                score = item[1]
                reasoning = item[2] if len(item) > 2 else ""
            else:
                continue
        
        # 验证关键词是否为字符串类型
        if not isinstance(keyword, str):
            continue
        
        # 转换分数为 float
        try:
            score = float(score)
        except (ValueError, TypeError):
            # 如果转换失败，设置默认分数
            score = 0.5
        
        # 统一输出为三元组格式: ["推理", "关键词", 分数]
        if json_format == "new":
            normalized_data.append([reasoning, keyword, score])
        else:
            normalized_data.append([keyword, score, reasoning])
    
    return normalized_data


def post_process_keywords(
    keywords_data,
    deduplicate=True,
    sort_by_importance=True,
    filter_low_score=False,
    score_threshold=0.0,
    top_n=False,
    n=10,
    return_full_info=False,
    json_format="new",
    remove_english=False,
    filter_stopwords=False,
    stopwords_exact_match=True,
    stopwords_contain_match=False,
    stopwords_file="stopwords.txt",
    filter_time_keywords=False,
    filter_date_keywords=False,
    filter_long_keywords=False,
    max_keyword_length=6,
    backfill_topn=True,
    filter_not_in_original=True,
    original_text=None,
    max_span_ratio=2
):
    """
    对关键词提取结果进行后处理
    
    Args:
        keywords_data: 原始关键词数据
            - 新格式: [["推理", "关键词", 分数], ...]
            - 旧格式: [["关键词", 分数, "理由"], ...]
        deduplicate: 是否去重（完全相同的词）
        sort_by_importance: 是否按 importance 分数从高到低排序
        filter_low_score: 是否过滤低分词
        score_threshold: 分数阈值（当 filter_low_score=True 时生效）
        top_n: 是否只保留前 N 个关键词
        n: 保留的关键词数量（当 top_n=True 时生效）
        return_full_info: 是否返回完整信息（包括分数和理由），False 则只返回关键词列表
        json_format: JSON格式类型，"new"(新格式) 或 "old"(旧格式)
        remove_english: 是否去除包含英文字母的关键词
        filter_stopwords: 是否启用停用词过滤
        stopwords_exact_match: 是否启用精确匹配（关键词完全等于停用词时过滤）
        stopwords_contain_match: 是否启用包含匹配（关键词包含停用词时过滤）
        stopwords_file: 停用词文件路径（默认为 stopwords.txt）
        filter_time_keywords: 是否过滤时间相关的关键词（如"8点"、"早上"等）
        filter_date_keywords: 是否过滤日期相关的关键词（如"27号"、"5月15"等）
        filter_long_keywords: 是否过滤超长关键词（长度超过max_keyword_length的关键词）
        max_keyword_length: 关键词最大长度（当 filter_long_keywords=True 时生效，默认为6）
        backfill_topn: 当启用top_n且过滤后数量不足N时，是否从被过滤的关键词中回填
        filter_not_in_original: 是否过滤原文中不存在的关键词（检查关键词每个字符是否都在原文中）
        original_text: 原始文本（当 filter_not_in_original=True 时必须提供，建议传入预处理后的文本）
        max_span_ratio: 关键词字符在原文中的最大跨度倍数（默认为2，防止字符散落拼凑）
        
    Returns:
        处理后的关键词列表或完整信息列表
    """
    # 如果输入为空，直接返回空列表
    if not keywords_data:
        return []
    
    # 深拷贝一份数据，避免修改原始数据
    processed_data = [item[:] for item in keywords_data]
    
    # 确定关键词和分数的位置（根据格式）
    if json_format == "new":
        # 新格式: [推理, 关键词, 分数]
        keyword_idx = 1
        score_idx = 2
    else:
        # 旧格式: [关键词, 分数, 理由]
        keyword_idx = 0
        score_idx = 1
    
    # 定义分数获取函数（后续多处使用）
    def get_score(item):
        """安全地获取分数，处理类型转换"""
        if len(item) <= score_idx:
            return 0
        score = item[score_idx]
        try:
            return float(score)
        except (ValueError, TypeError):
            return 0
    
    # 0. 过滤空关键词（最先执行，避免空关键词参与后续处理）
    # 移除关键词为空字符串的项
    filtered_data = []
    # logging.info(f"过滤空关键词前: len(processed_data) = {len(processed_data)}")
    for item in processed_data:
        if len(item) > keyword_idx:
            keyword = item[keyword_idx]
            # 确保关键词是字符串且非空
            if isinstance(keyword, str) and keyword.strip():
                filtered_data.append(item)
    processed_data = filtered_data
    # logging.info(f"过滤空关键词后: len(processed_data) = {len(processed_data)}")
    
    # 1. 按 importance 分数排序（如果启用）
    if sort_by_importance:
        processed_data.sort(key=get_score, reverse=True)

    
    # 2. 去重（如果启用）
    if deduplicate:
        processed_data = deduplicate_keywords(processed_data, keyword_idx=keyword_idx)
    
    # 3. 过滤低分词（如果启用）
    if filter_low_score:
        def check_score(item):
            """安全地检查分数是否达到阈值"""
            if len(item) <= score_idx:
                return False
            try:
                score = float(item[score_idx])
                return score >= score_threshold
            except (ValueError, TypeError):
                return False
        
        processed_data = [item for item in processed_data if check_score(item)]
    
    # 注意：这里不再提前执行 top_n 截取
    # 4. top_n 截取移到所有过滤完成后（见后文）
    
    # 5. 去除包含英文字母的关键词（如果启用）
    if remove_english:
        processed_data = filter_english(processed_data, keyword_idx=keyword_idx)
    
    # 6. 过滤停用词（如果启用）
    if filter_stopwords and (stopwords_exact_match or stopwords_contain_match):
        # 加载停用词列表
        stopwords = load_stopwords(stopwords_file)
        
        if stopwords:  # 只有在成功加载停用词时才进行过滤
            filtered_data = []
            filtered_out = []  # 记录被过滤的关键词（用于调试）
            # logging.info(f"[停用词过滤] 过滤前关键词: {[item[keyword_idx] if len(item) > keyword_idx else '?' for item in processed_data]}")
            
            for item in processed_data:
                if len(item) > keyword_idx:
                    keyword = item[keyword_idx]
                    # 确保关键词是字符串类型
                    if not isinstance(keyword, str):
                        # logging.info(f"[停用词过滤] 跳过非字符串关键词: {keyword} (类型: {type(keyword)})")
                        continue  # 跳过非字符串类型的关键词
                    
                    should_filter = False
                    
                    # 精确匹配：关键词完全等于停用词
                    if stopwords_exact_match and keyword in stopwords:
                        # logging.info(f"[停用词过滤] '{keyword}' 匹配停用词（精确匹配）")
                        should_filter = True
                    
                    # 包含匹配：关键词包含停用词
                    if stopwords_contain_match and not should_filter:
                        for stopword in stopwords:
                            if stopword in keyword:
                                should_filter = True
                                # logging.info(f"[停用词过滤] '{keyword}' 包含停用词 '{stopword}'（包含匹配）")
                                break
                    
                    # 如果不需要过滤，则保留该关键词
                    if not should_filter:
                        filtered_data.append(item)
                    else:
                        filtered_out.append(keyword)

            # 调试输出（可选，取消注释以启用）
            # logging.info(f"[停用词过滤] 过滤前: {len(processed_data)} 个, 过滤后: {len(filtered_data)} 个, 被过滤: {filtered_out}")
            
            processed_data = filtered_data
    
    # 7. 过滤时间关键词（如果启用）
    if filter_time_keywords:
        filtered_data = []
        for item in processed_data:
            if len(item) > keyword_idx:
                keyword = item[keyword_idx]
                # 确保关键词是字符串类型
                if not isinstance(keyword, str):
                    continue  # 跳过非字符串类型的关键词
                
                # 检查是否为时间关键词
                if not is_time_keyword(keyword):
                    filtered_data.append(item)
        
        processed_data = filtered_data
    
    # 8. 过滤日期关键词（如果启用）
    if filter_date_keywords:
        filtered_data = []
        filtered_out_dates = []  # 记录被过滤的日期关键词
        for item in processed_data:
            if len(item) > keyword_idx:
                keyword = item[keyword_idx]
                # 确保关键词是字符串类型
                if not isinstance(keyword, str):
                    continue  # 跳过非字符串类型的关键词
                
                # 检查是否为日期关键词
                if not is_date_keyword(keyword):
                    filtered_data.append(item)
                else:
                    filtered_out_dates.append(keyword)
        
        # 调试输出（可选）
        # if filtered_out_dates:
        #     logging.info(f"[日期过滤] 过滤前: {len(processed_data)} 个, 过滤后: {len(filtered_data)} 个, 被过滤: {filtered_out_dates}")
        
        processed_data = filtered_data
    
    # 9. 过滤超长关键词（如果启用）
    if filter_long_keywords and max_keyword_length > 0:
        filtered_data = []
        filtered_out_long = []  # 记录被过滤的超长关键词
        for item in processed_data:
            if len(item) > keyword_idx:
                keyword = item[keyword_idx]
                # 确保关键词是字符串类型
                if not isinstance(keyword, str):
                    continue  # 跳过非字符串类型的关键词
                
                # 检查关键词长度是否超过最大长度
                if len(keyword) <= max_keyword_length:
                    filtered_data.append(item)
                else:
                    filtered_out_long.append(keyword)
        
        # 调试输出（可选）
        # if filtered_out_long:
        #     logging.info(f"[长度过滤] 过滤前: {len(processed_data)} 个, 过滤后: {len(filtered_data)} 个, 被过滤: {filtered_out_long}")
        
        processed_data = filtered_data

    # 10. 过滤原文中不存在的关键词（如果启用）
    if filter_not_in_original and original_text:
        processed_data = filter_keywords_not_in_original(
            processed_data, 
            original_text, 
            keyword_idx=keyword_idx,
            max_span_ratio=max_span_ratio
        )


    # ========== 重要：在所有过滤完成后，保存干净的数据用于回填 ==========
    # 这样回填时不会回填被停用词/时间词/日期词/超长关键词过滤掉的关键词
    clean_sorted_data = [item[:] for item in processed_data]
    
    # 11. 智能 top_n 处理（所有过滤完成后）
    if top_n and n > 0:
        # 先截取前 N 个
        if len(processed_data) > n:
            processed_data = processed_data[:n]
        
        current_count = len(processed_data)
        
        if current_count < n and backfill_topn:
            # 当前数量不足 N 个，需要回填
            # 计算需要回填的数量
            needed_count = n - current_count
            
            # 获取当前已选中的关键词集合（用于去重）
            selected_keywords = set()
            for item in processed_data:
                if len(item) > keyword_idx:
                    keyword = item[keyword_idx]
                    if isinstance(keyword, str):
                        selected_keywords.add(keyword)
            
            # 从干净的数据中找回填候选，而不是从原始数据;确保回填的关键词也是经过停用词和时间词过滤的
            backfill_candidates = []
            for item in clean_sorted_data:
                if len(item) > keyword_idx:
                    keyword = item[keyword_idx]
                    # 确保是字符串且未被选中
                    if isinstance(keyword, str) and keyword not in selected_keywords:
                        backfill_candidates.append(item)
            
            # 从候选中选择分数最高的进行回填
            backfill_items = backfill_candidates[:needed_count]
            
            # 将回填的关键词追加到结果末尾
            processed_data.extend(backfill_items)
            
            # 调试输出
            if backfill_items:
                backfilled_keywords = [item[keyword_idx] for item in backfill_items if len(item) > keyword_idx]
                logging.info(f"[智能回填] 当前: {current_count} 个, 目标: {n} 个, 回填: {len(backfill_items)} 个 → {backfilled_keywords}")
        
        # 最终截取到 N 个（如果回填后超过 N 个，需要截断；正常情况下应该正好是 N 个或少于 N 个）
        processed_data = processed_data[:n]
    
    # 11. 根据 return_full_info 决定返回格式
    if return_full_info:
        # 返回完整信息
        return processed_data
    else:
        # 只返回关键词列表（确保是字符串）
        keywords_list = []
        for item in processed_data:
            if len(item) > keyword_idx:
                keyword = item[keyword_idx]
                # 只添加字符串类型的关键词
                if isinstance(keyword, str):
                    keywords_list.append(keyword)
        return keywords_list


def extract_keywords_from_json(keywords_json, return_raw=False):
    """
    从 JSON 格式的关键词数据中提取关键词列表
    
    Args:
        keywords_json: JSON 格式的关键词数据（字典）
        return_raw: 是否返回原始格式（包含分数和理由）
        
    Returns:
        关键词数据列表，格式为 [["词1", 0.90, "理由"], ...] 或 []
    """
    # 统一处理返回的关键词部分（兼容中英文键名）
    if '关键词' in keywords_json:
        keywords_data = keywords_json['关键词']
    elif 'keywords' in keywords_json:
        keywords_data = keywords_json['keywords']
    else:
        return []
    
    return keywords_data


def post_process_keywords_with_config(
    keywords_data,
    config,
    original_text=None,
    max_keywords=None,
    json_format="new",
):
    """
    使用 PostprocessConfig（或 AgentConfig）驱动后处理，减少调用方参数传递。

    Args:
        keywords_data: 原始关键词数据
        config: PostprocessConfig 或 AgentConfig 实例
        original_text: 原文文本（用于原文对齐校验）
        max_keywords: 覆盖 config.n 的最大关键词数
        json_format: JSON 格式类型
    """
    from ..config import PostprocessConfig

    if isinstance(config, PostprocessConfig):
        pp = config
    else:
        pp = getattr(config, "postprocess", config)

    return post_process_keywords(
        keywords_data,
        deduplicate=pp.deduplicate,
        sort_by_importance=pp.sort_by_importance,
        filter_low_score=pp.filter_low_score,
        score_threshold=pp.score_threshold,
        top_n=pp.top_n,
        n=max_keywords if max_keywords is not None else pp.n,
        return_full_info=pp.return_full_info,
        json_format=json_format,
        remove_english=pp.remove_english,
        filter_stopwords=pp.filter_stopwords,
        stopwords_exact_match=pp.stopwords_exact_match,
        stopwords_contain_match=pp.stopwords_contain_match,
        stopwords_file=pp.stopwords_file,
        filter_time_keywords=pp.filter_time_keywords,
        filter_date_keywords=pp.filter_date_keywords,
        filter_long_keywords=pp.filter_long_keywords,
        max_keyword_length=pp.max_keyword_length,
        backfill_topn=pp.backfill_topn,
        filter_not_in_original=pp.filter_not_in_original,
        original_text=original_text,
        max_span_ratio=pp.max_span_ratio,
    )
