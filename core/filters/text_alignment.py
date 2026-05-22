"""关键词原文对齐过滤"""

import logging


def find_min_span_in_text(keyword, original_text):
    """
    在原始文本中找到能匹配关键词所有字符的最小连续范围
    
    使用贪婪算法：从关键词第一个字符在原文中的每个出现位置开始，
    依次向后查找剩余字符，计算匹配所需的最小范围。
    
    Args:
        keyword: 待匹配的关键词
        original_text: 原始文本
        
    Returns:
        最小匹配范围的长度，如果无法匹配则返回 -1
    """
    if not keyword or not original_text:
        return -1
    
    # 过滤掉关键词中的空格
    keyword_chars = [c for c in keyword if c != ' ']
    if not keyword_chars:
        return -1
    
    min_span = float('inf')
    
    # 找到第一个字符在原文中的所有位置
    first_char = keyword_chars[0]
    start_positions = [i for i, c in enumerate(original_text) if c == first_char]
    
    # 对于每个起始位置，尝试贪婪匹配
    for start_pos in start_positions:
        current_pos = start_pos
        matched = True
        
        # 依次匹配剩余字符
        for i, char in enumerate(keyword_chars):
            if i == 0:
                continue  # 第一个字符已经匹配
            
            # 从当前位置向后查找下一个字符
            found = False
            for j in range(current_pos + 1, len(original_text)):
                if original_text[j] == char:
                    current_pos = j
                    found = True
                    break
            
            if not found:
                matched = False
                break
        
        if matched:
            # 计算这次匹配的范围（从起始位置到最后匹配位置）
            span = current_pos - start_pos + 1
            min_span = min(min_span, span)
    
    return min_span if min_span != float('inf') else -1


def validate_keyword_chars_in_text(keyword, original_text, max_span_ratio=2):
    """
    验证关键词中的每个字符是否都在原始文本中紧凑地存在
    
    该函数用于过滤掉模型"自行推理总结"产生的、原文中不存在的关键词。
    
    验证规则：
    1. 关键词的每个字符都必须在原文中存在
    2. 这些字符在原文中的最小匹配范围不能超过关键词长度的 max_span_ratio 倍
    
    例如（假设 max_span_ratio=2）：
    - 原文"已经退货"，关键词"没退货" → 无效（"没"不在原文中）
    - 原文"衣服不怎么粘肉"，关键词"不粘肉" → 有效（范围5，关键词长度3，比例1.67<2）
    - 原文"聊个不停...一天"，关键词"聊天" → 无效（字符散落太远，范围远超2倍）
    
    Args:
        keyword: 待验证的关键词
        original_text: 原始文本（建议传入预处理后的文本）
        max_span_ratio: 最大跨度倍数（默认为2，即匹配范围不超过关键词长度的2倍）
        
    Returns:
        True 如果关键词验证通过，否则 False
    """
    if not isinstance(keyword, str) or not isinstance(original_text, str):
        return False
    
    if not keyword or not original_text:
        return False
    
    # 过滤掉关键词中的空格
    keyword_chars = [c for c in keyword if c != ' ']
    if not keyword_chars:
        return False
    
    keyword_len = len(keyword_chars)
    
    # 1. 首先检查每个字符是否都在原文中存在
    original_chars = set(original_text)
    for char in keyword_chars:
        if char not in original_chars:
            return False
    
    # 2. 检查字符在原文中的紧凑性（最小匹配范围）
    min_span = find_min_span_in_text(keyword, original_text)
    
    if min_span < 0:
        # 无法找到匹配，说明字符顺序不对或不存在
        return False
    
    # 计算允许的最大范围
    max_allowed_span = keyword_len * max_span_ratio
    
    if min_span > max_allowed_span:
        logging.debug(f"[原文验证] 关键词'{keyword}'跨度过大: 最小范围={min_span}, 允许最大={max_allowed_span}")
        return False
    
    return True


def filter_keywords_not_in_original(keywords_data, original_text, keyword_idx=1, max_span_ratio=2):
    """
    过滤掉不在原始文本中的关键词
    
    遍历关键词列表，检查每个关键词的所有字符是否都在原始文本中紧凑地存在，
    过滤掉包含原文中不存在字符的关键词，以及字符散落过远的拼凑关键词。
    
    Args:
        keywords_data: 关键词数据列表，格式为 [[推理, 关键词, 分数], ...]
        original_text: 原始文本（建议传入预处理后的文本）
        keyword_idx: 关键词在列表中的索引位置（新格式为1，旧格式为0）
        max_span_ratio: 最大跨度倍数（默认为2，即匹配范围不超过关键词长度的2倍）
        
    Returns:
        过滤后的关键词数据列表
    """
    if not keywords_data or not original_text:
        return keywords_data
    
    filtered_data = []
    filtered_out = []  # 记录被过滤的关键词（用于调试）
    
    for item in keywords_data:
        if len(item) > keyword_idx:
            keyword = item[keyword_idx]
            
            # 确保关键词是字符串类型
            if not isinstance(keyword, str):
                continue
            
            # 验证关键词中的每个字符是否都在原文中紧凑地存在
            if validate_keyword_chars_in_text(keyword, original_text, max_span_ratio):
                filtered_data.append(item)
            else:
                filtered_out.append(keyword)
    
    # 调试输出
    if filtered_out:
        logging.info(f"[原文验证过滤] 过滤前: {len(keywords_data)} 个, 过滤后: {len(filtered_data)} 个, 被过滤: {filtered_out}")
    
    return filtered_data
