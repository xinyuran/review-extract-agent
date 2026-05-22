import re


def remove_urls(text):
    """去除文本中的 URL 链接"""
    # 匹配 http/https/ftp 开头的 URL
    text = re.sub(r'https?://\S+|ftp://\S+', '', text)
    # 匹配 www 开头的 URL
    text = re.sub(r'www\.\S+', '', text)
    return text


def remove_emails(text):
    """去除文本中的邮箱地址"""
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
    return text


def remove_phone_numbers(text):
    """去除文本中的手机号码"""
    # 匹配中国手机号（11位数字）
    text = re.sub(r'1[3-9]\d{9}', '', text)
    return text
