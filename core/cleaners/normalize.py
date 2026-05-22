import re
import html
import unicodedata


def normalize_numbers(text):
    """规范化数字表示（将全角数字转为半角）"""
    # 全角数字转半角
    full_width = '０１２３４５６７８９'
    half_width = '0123456789'
    trans_table = str.maketrans(full_width, half_width)
    return text.translate(trans_table)


def remove_whitespace_chars(text):
    """
    去除文本中的空白字符和换行符
    
    去除的字符包括：
    - \n 换行符 (Line Feed)
    - \r 回车符 (Carriage Return)
    - \t 制表符 (Tab)
    - \v 垂直制表符 (Vertical Tab)
    - \f 换页符 (Form Feed)
    - \r\n Windows换行 (CRLF)
    
    Args:
        text: 待处理的文本
        
    Returns:
        去除空白字符后的文本
    """
    if not isinstance(text, str):
        text = str(text)
    
    # 定义需要去除的空白字符
    whitespace_chars = [
        '\r\n',  # Windows换行（CRLF），需要先处理，避免被拆分
        '\n',    # 换行符 (Line Feed)
        '\r',    # 回车符 (Carriage Return)
        '\t',    # 制表符 (Tab)
        '\v',    # 垂直制表符 (Vertical Tab)
        '\f',    # 换页符 (Form Feed)
    ]
    
    for char in whitespace_chars:
        text = text.replace(char, '')
    
    return text


def keep_chinese_only(text, keep_numbers=True, keep_chinese_punctuation=True):
    """
    只保留中文字符（可选保留数字和中文标点）
    
    Args:
        text: 待处理的文本
        keep_numbers: 是否保留数字（0-9）
        keep_chinese_punctuation: 是否保留中文标点符号
        
    Returns:
        只包含中文的文本
    """
    if not isinstance(text, str):
        text = str(text)
    
    # 定义要保留的字符范围
    result = []
    
    for char in text:
        # 1. 保留中文汉字（CJK统一表意文字）
        if '\u4e00' <= char <= '\u9fff':  # 基本汉字
            result.append(char)
        # 2. 保留中文标点符号（如果启用）
        elif keep_chinese_punctuation and char in '，。！？；：""''（）【】《》、…—·':
            result.append(char)
        # 3. 保留数字（如果启用）
        elif keep_numbers and char.isdigit():
            result.append(char)
        # 4. 保留空格（用于分隔）
        elif char == ' ':
            result.append(char)
    
    return ''.join(result)


def clean_text(text, remove_english=True, deduplicate_punctuation=True, 
               remove_html_entities=True, normalize_whitespace=True,
               remove_control_chars=True):
    """
    文本预处理主函数
    
    Args:
        text: 待处理的文本
        remove_english: 是否去除英文字母（保留数字）
        deduplicate_punctuation: 是否去除连续重复的标点符号
        remove_html_entities: 是否去除 HTML 实体
        normalize_whitespace: 是否规范化空白字符
        remove_control_chars: 是否去除控制字符
        
    Returns:
        清洗后的文本
    """
    if not isinstance(text, str):
        text = str(text)
    
    # 1. 去除 HTML 实体（如 &hellip; → ...，&nbsp; → 空格）
    if remove_html_entities:
        text = html.unescape(text)
        # 进一步处理常见的 HTML 实体
        text = text.replace('&nbsp;', ' ')
        text = text.replace('&quot;', '"')
        text = text.replace('&amp;', '&')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
    
    # 2. 去除所有英文字母（a-z, A-Z），保留数字
    if remove_english:
        text = re.sub(r'[a-zA-Z]+', '', text)
    
    # 3. 去除控制字符（如 \x00-\x1f，但保留常用的换行、制表符）
    if remove_control_chars:
        # 保留常用空白字符：空格、换行、制表符
        text = ''.join(char for char in text 
                      if unicodedata.category(char)[0] != 'C' 
                      or char in ['\n', '\r', '\t', ' '])
    
    # 4. 去除连续重复的标点符号（!!! → !，... → .）
    if deduplicate_punctuation:
        # 匹配连续重复的标点符号（包括中英文标点）
        # 标点符号类别：P (Punctuation)
        text = re.sub(r'([!！？?。.，,、；;：:""\"\'\'（）()【】\[\]《》<>…~～@#￥$%^&*_+\-=｜|/\\])\1+', r'\1', text)
    
    # 5. 规范化空白字符（多个空格/换行合并为一个空格）
    if normalize_whitespace:
        # 将多个连续的空白字符（空格、换行、制表符等）合并为一个空格
        text = re.sub(r'\s+', ' ', text)
        # 去除首尾空白
        text = text.strip()
    
    return text
