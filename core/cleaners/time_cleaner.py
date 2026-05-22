import re


def remove_time_expressions(text):
    """
    去除文本中的时间表达式（系统化处理）
    
    支持的时间格式：
    - 标准时间：8:40、08:40:30、8.40、8点40分30秒
    - 口语时间：8点多、8点半、8点左右、差5分8点
    - 时间段：早上、上午、中午、下午、晚上、凌晨、夜里
    - 模糊时间：刚才、现在、马上、立刻、稍后
    
    Args:
        text: 待处理的文本
        
    Returns:
        去除时间表达后的文本
    """
    if not isinstance(text, str):
        text = str(text)
    
    # ===== 1. 标准时间格式 =====
    
    # 1.1 冒号分隔的时间：8:40、08:40:30、23:59:59
    text = re.sub(r'\d{1,2}:\d{1,2}(:\d{1,2})?', '', text)
    
    # 1.2 点号分隔的时间：8.40、8.40.30（需要确保不是小数）
    # 使用前瞻/后顾断言确保不是价格等小数
    text = re.sub(r'(?<!\d)\d{1,2}\.\d{1,2}(\.\d{1,2})?(?!\d)', '', text)
    
    # 1.3 中文完整时间：8点40分30秒、8点40分、8点40
    # 匹配：数字+点+数字+分+数字+秒
    text = re.sub(r'\d+\s*点\s*\d+\s*分\s*\d+\s*秒', '', text)
    # 匹配：数字+点+数字+分
    text = re.sub(r'\d+\s*点\s*\d+\s*分', '', text)
    # 匹配：数字+点+数字（后面没有单位）
    text = re.sub(r'\d+\s*点\s*\d+(?![分秒])', '', text)
    
    # 1.4 中文时间：8点、40分、30秒、8时、40分钟
    text = re.sub(r'\d+\s*[点时]\s*(?:钟)?', '', text)
    text = re.sub(r'\d+\s*分\s*(?:钟)?', '', text)
    text = re.sub(r'\d+\s*秒\s*(?:钟)?', '', text)
    
    # 1.5 中文数字时间：八点、四十分、三十秒
    chinese_num_pattern = r'[零一二三四五六七八九十百千万亿壹贰叁肆伍陆柒捌玖拾佰仟萬億两]+'
    text = re.sub(chinese_num_pattern + r'\s*[点时]\s*(?:钟)?', '', text)
    text = re.sub(chinese_num_pattern + r'\s*分\s*(?:钟)?', '', text)
    text = re.sub(chinese_num_pattern + r'\s*秒\s*(?:钟)?', '', text)
    
    # ===== 2. 口语化时间表达 =====
    
    # 2.1 模糊时间：8点多、8点半、8点左右、8点钟左右
    text = re.sub(r'\d+\s*[点时]\s*[多半来钟]', '', text)
    text = re.sub(r'\d+\s*[点时]\s*(?:左右|上下)', '', text)
    
    # 2.2 差几分几点：差5分8点、差一刻9点
    text = re.sub(r'差\s*\d+\s*分\s*\d+\s*[点时]', '', text)
    text = re.sub(r'差\s*一刻\s*\d+\s*[点时]', '', text)
    
    # 2.3 几点几刻：8点一刻、9点三刻
    text = re.sub(r'\d+\s*[点时]\s*[一二三]刻', '', text)
    
    # ===== 3. 时间段表达 =====
    
    # 3.1 时间段：早上、上午、中午、下午、晚上、夜里、凌晨、深夜
    text = re.sub(r'[早上中下晚夜凌深][上午里晨间夜]', '', text)
    
    # 3.2 带时间段的完整表达：上午8点、晚上9点半
    text = re.sub(r'[早上中下晚夜凌深][上午里晨间夜]\s*\d+\s*[点时]', '', text)
    
    # ===== 4. 模糊时间词 =====
    
    # 刚才、现在、马上、立刻、稍后、待会、一会儿
    fuzzy_time_words = [
        '刚才', '刚刚', '现在', '此刻', '当前', '目前',
        '马上', '立刻', '立即', '立马', '即刻',
        '稍后', '待会', '一会儿', '过会', '等会',
        '随后', '之后', '然后', '接着', '紧接着'
    ]
    for word in fuzzy_time_words:
        text = text.replace(word, '')
    
    return text


def remove_date_expressions(text):
    """
    去除文本中的日期表达式（系统化处理）
    
    支持的日期格式：
    - 标准日期：2024年11月10日、2024-11-10、11/10、11.10
    - 年月日：2024年、11月、10号、10日
    - 相对日期：昨天、今天、明天、前天、后天
    - 周期：第一天、第二周、上个月、去年
    
    Args:
        text: 待处理的文本
        
    Returns:
        去除日期表达后的文本
    """
    if not isinstance(text, str):
        text = str(text)
    
    # ===== 1. 标准日期格式 =====
    
    # 1.1 完整日期：2024年11月10日、2024-11-10、2024/11/10、2024.11.10
    text = re.sub(r'\d{4}[-/年.]\d{1,2}[-/月.]\d{1,2}[日号]?', '', text)
    
    # 1.2 月日格式：11月10日、11-10、11/10、11.10
    text = re.sub(r'\d{1,2}[-/月.]\d{1,2}[日号]?', '', text)
    
    # 1.3 单独的年月日号：2024年、11月、10号、10日
    text = re.sub(r'\d+\s*[年]', '', text)
    text = re.sub(r'\d+\s*[月]', '', text)
    text = re.sub(r'\d+\s*[日号]', '', text)
    
    # ===== 2. 中文数字日期 =====
    
    chinese_num_pattern = r'[零一二三四五六七八九十百千万亿壹贰叁肆伍陆柒捌玖拾佰仟萬億两]+'
    
    # 2.1 中文数字+年月日号
    text = re.sub(chinese_num_pattern + r'\s*[年]', '', text)
    text = re.sub(chinese_num_pattern + r'\s*[月]', '', text)
    text = re.sub(chinese_num_pattern + r'\s*[日号]', '', text)
    
    # ===== 3. 相对日期表达 =====
    
    # 3.1 昨天、今天、明天、前天、后天、大前天、大后天
    relative_days = ['昨天', '今天', '明天', '前天', '后天', '大前天', '大后天', '昨日', '今日', '明日']
    for day in relative_days:
        text = text.replace(day, '')
    
    # 3.2 上周、本周、下周、上月、本月、下月、去年、今年、明年
    relative_periods = [
        '上周', '本周', '下周', '这周', '上星期', '本星期', '下星期', '这星期',
        '上月', '本月', '下月', '这月', '上个月', '这个月', '下个月',
        '去年', '今年', '明年', '前年', '后年'
    ]
    for period in relative_periods:
        text = text.replace(period, '')
    
    # ===== 4. 周期表达 =====
    
    # 4.1 第X天/周/月/年
    text = re.sub(r'第' + chinese_num_pattern + r'[天周月年]', '', text)
    text = re.sub(r'第\d+[天周月年]', '', text)
    
    # 4.2 X天/周/月/年前/后
    text = re.sub(chinese_num_pattern + r'[天周月年][前后]', '', text)
    text = re.sub(r'\d+[天周月年][前后]', '', text)
    
    # 4.3 星期X、周X
    text = re.sub(r'[星期周][一二三四五六七日天]', '', text)
    text = re.sub(r'礼拜[一二三四五六七日天]', '', text)
    
    return text


def remove_dates(text):
    """
    去除文本中的日期和时间表达式（统一入口）
    
    这是一个兼容性函数，内部调用更细粒度的处理函数。
    建议直接使用 remove_date_expressions() 和 remove_time_expressions()。
    
    Args:
        text: 待处理的文本
        
    Returns:
        去除日期和时间后的文本
    """
    if not isinstance(text, str):
        text = str(text)
    
    # 先去除时间（因为时间可能包含在日期中，如"2024年11月10日8点"）
    text = remove_time_expressions(text)
    
    # 再去除日期
    text = remove_date_expressions(text)
    
    return text
