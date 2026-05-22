import re


def remove_emojis(text):
    """
    去除文本中的emoji表情符号
    
    包括：
    - 标准emoji (😀😁😂等)
    - 表情符号 (☺️♥️等)
    - 特殊符号 (🔥⭐等)
    
    Args:
        text: 待处理的文本
        
    Returns:
        去除emoji后的文本
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Emoji Unicode 范围
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # 表情符号 (Emoticons)
        "\U0001F300-\U0001F5FF"  # 符号和象形文字 (Symbols & Pictographs)
        "\U0001F680-\U0001F6FF"  # 交通和地图符号 (Transport & Map Symbols)
        "\U0001F1E0-\U0001F1FF"  # 旗帜 (Flags)
        "\U0001F900-\U0001F9FF"  # 补充符号和象形文字 (Supplemental Symbols and Pictographs)
        "\U0001FA00-\U0001FA6F"  # 扩展符号A (Extended Symbols A)
        "\U0001FA70-\U0001FAFF"  # 扩展符号B (Extended Symbols B)
        "\U0001F000-\U0001F02F"  # 麻将牌 (Mahjong Tiles)
        "\U0001F0A0-\U0001F0FF"  # 扑克牌 (Playing Cards)
        "]+",
        flags=re.UNICODE
    )
    
    return emoji_pattern.sub(r'', text)


def remove_garbled_text(text):
    """
    去除乱码字符
    
    包括：
    - 不可见字符（零宽字符等）
    - 特殊控制字符
    - 异常Unicode字符
    - 常见乱码模式（如：锟斤拷、烫烫烫等）
    
    Args:
        text: 待处理的文本
        
    Returns:
        去除乱码后的文本
    """
    if not isinstance(text, str):
        text = str(text)
    
    # 1. 去除零宽字符（Zero-Width Characters）
    # 零宽空格、零宽连接符、零宽非连接符等
    zero_width_chars = [
        '\u200b',  # 零宽空格 (Zero Width Space)
        '\u200c',  # 零宽非连接符 (Zero Width Non-Joiner)
        '\u200d',  # 零宽连接符 (Zero Width Joiner)
        '\ufeff',  # 零宽非断空格 (Zero Width No-Break Space / BOM)
        '\u2060',  # 字连接符 (Word Joiner)
    ]
    for char in zero_width_chars:
        text = text.replace(char, '')
    
    # 2. 去除常见乱码模式
    garbled_patterns = [
        '锟斤拷',  # UTF-8编码问题导致的乱码
        '烫烫烫',  # 未初始化内存显示
        '屯屯屯',  # 类似乱码
        '�',      # Unicode替换字符 (Replacement Character)
    ]
    for pattern in garbled_patterns:
        text = text.replace(pattern, '')
    
    # 3. 去除特殊格式字符（Variation Selectors）
    # 用于改变前面字符的显示方式
    text = re.sub(r'[\uFE00-\uFE0F]', '', text)  # Variation Selectors
    text = re.sub(r'[\U000E0100-\U000E01EF]', '', text)  # Variation Selectors Supplement
    
    # 4. 去除其他不可见或特殊Unicode字符
    # 格式控制字符
    text = re.sub(r'[\u200e\u200f]', '', text)  # Left-to-Right/Right-to-Left Mark
    text = re.sub(r'[\u202a-\u202e]', '', text)  # Directional Formatting
    
    # 5. 去除私有使用区字符（Private Use Area）
    # 这些字符通常用于自定义符号，可能显示为乱码
    text = re.sub(r'[\ue000-\uf8ff]', '', text)  # Private Use Area
    text = re.sub(r'[\U000F0000-\U000FFFFD]', '', text)  # Supplementary Private Use Area-A
    text = re.sub(r'[\U00100000-\U0010FFFD]', '', text)  # Supplementary Private Use Area-B
    
    return text


def remove_special_symbols(text):
    """
    去除特殊符号（保留常用标点）
    
    去除：
    - 数学符号 (±×÷≈等)
    - 货币符号 (€£¥$等，除了常用的￥)
    - 箭头符号 (→←↑↓等)
    - 几何图形 (■□●○等)
    - 音乐符号 (♪♫等)
    
    Args:
        text: 待处理的文本
        
    Returns:
        去除特殊符号后的文本
    """
    if not isinstance(text, str):
        text = str(text)
    
    # 定义要去除的特殊符号范围
    special_symbols_pattern = re.compile(
        "["
        "\u2190-\u21FF"  # 箭头 (Arrows)
        "\u2200-\u22FF"  # 数学运算符 (Mathematical Operators)
        "\u2300-\u23FF"  # 杂项技术符号 (Miscellaneous Technical)
        "\u2500-\u257F"  # 制表符 (Box Drawing)
        "\u2580-\u259F"  # 方块元素 (Block Elements)
        "\u25A0-\u25FF"  # 几何图形 (Geometric Shapes)
        "\u2600-\u26FF"  # 杂项符号 (Miscellaneous Symbols) - 包含部分emoji
        "\u2700-\u27BF"  # 装饰符号 (Dingbats)
        "\u2B00-\u2BFF"  # 杂项符号和箭头 (Miscellaneous Symbols and Arrows)
        "\u20A0-\u20CF"  # 货币符号 (Currency Symbols) - 但保留￥
        "]+",
        flags=re.UNICODE
    )
    
    text = special_symbols_pattern.sub(r'', text)
    
    # 额外处理：去除一些常见的特殊符号（不在上述范围内的）
    text = text.replace('™', '')  # 商标
    text = text.replace('®', '')  # 注册商标
    text = text.replace('©', '')  # 版权
    text = text.replace('§', '')  # 章节符号
    text = text.replace('¶', '')  # 段落符号
    text = text.replace('†', '')  # 剑标
    text = text.replace('‡', '')  # 双剑标
    text = text.replace('•', '')  # 项目符号
    text = text.replace('◦', '')  # 空心项目符号
    text = text.replace('‣', '')  # 三角项目符号
    
    return text
