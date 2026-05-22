"""停用词与时间日期过滤"""

import os
import re
import logging


# 停用词缓存字典：{文件路径: 停用词集合}
_stopwords_cache = {}


# ==================== 通用中文数字模式 ====================
# 包含：零一二三四五六七八九十百千万亿（简体）
#       壹贰叁肆伍陆柒捌玖拾佰仟萬億（繁体/大写）
#       〇两（特殊数字）
CHINESE_NUM_PATTERN = r'[零〇一二三四五六七八九十百千万亿壹贰叁肆伍陆柒捌玖拾佰仟萬億两]+'
# 阿拉伯数字模式
ARABIC_NUM_PATTERN = r'\d+'
# 任意数字模式（中文或阿拉伯数字）
ANY_NUM_PATTERN = f'(?:{CHINESE_NUM_PATTERN}|{ARABIC_NUM_PATTERN})'


def load_stopwords(stopwords_file="stopwords.txt"):
    """
    从文件加载停用词列表（带缓存机制，避免重复加载）
    
    Args:
        stopwords_file: 停用词文件路径（默认为当前目录下的 stopwords.txt）
        
    Returns:
        停用词集合（set）
    """
    # 如果提供的是相对路径，则相对于当前脚本所在目录
    if not os.path.isabs(stopwords_file):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        stopwords_file = os.path.join(script_dir, stopwords_file)
    
    # 检查缓存中是否已经加载过该文件
    if stopwords_file in _stopwords_cache:
        return _stopwords_cache[stopwords_file]
    
    # 首次加载
    stopwords = set()
    
    # 检查文件是否存在
    if not os.path.exists(stopwords_file):
        logging.warning(f"停用词文件不存在: {stopwords_file}")
        _stopwords_cache[stopwords_file] = stopwords
        return stopwords
    
    try:
        with open(stopwords_file, 'r', encoding='utf-8') as f:
            for line in f:
                # 去除首尾空白字符
                word = line.strip()
                # 跳过空行和注释行
                if word and not word.startswith('#'):
                    stopwords.add(word)
        logging.info(f"成功加载 {len(stopwords)} 个停用词（来自: {os.path.basename(stopwords_file)}）")
        # 缓存结果
        _stopwords_cache[stopwords_file] = stopwords
    except Exception as e:
        logging.error(f"加载停用词文件失败: {e}")
        _stopwords_cache[stopwords_file] = stopwords
    
    return stopwords


def is_date_keyword(keyword):
    """
    判断关键词是否为日期相关词汇
    
    检测规则：
    - 年份：2024年、二零二四年、二千零二十四年
    - 月份：1月、一月、01月
    - 日期：1号、一号、01号、1日、一日
    - 完整日期：5月15、五月十五号、2024年11月
    - 星期：周一、星期一、礼拜一
    - 相对日期：今天、昨天、明天、前天、后天
    
    Args:
        keyword: 待检测的关键词
        
    Returns:
        True 如果是日期关键词，否则 False
    """
    if not isinstance(keyword, str):
        return False
    
    # ===== 1. 年份检测 =====
    
    # 1.1 数字年份：2024年、24年、任意数字+年
    if re.search(ARABIC_NUM_PATTERN + r'\s*年', keyword):
        return True
    
    # 1.2 中文年份：二零二四年、二〇二四年、二千零二十四年
    if re.search(CHINESE_NUM_PATTERN + r'\s*年', keyword):
        return True
    
    # ===== 2. 月份检测 =====
    
    # 2.1 数字月份：1月、01月、12月
    if re.search(ARABIC_NUM_PATTERN + r'\s*月', keyword):
        return True
    
    # 2.2 中文月份：一月、十二月、任意中文数字+月
    if re.search(CHINESE_NUM_PATTERN + r'\s*月', keyword):
        return True
    
    # ===== 3. 日期检测 =====
    
    # 3.1 数字日期：1号、01号、1日、01日
    if re.search(ARABIC_NUM_PATTERN + r'\s*[号日]', keyword):
        return True
    
    # 3.2 中文日期：一号、二十七号、三十一日
    if re.search(CHINESE_NUM_PATTERN + r'\s*[号日]', keyword):
        return True
    
    # 3.3 只有"号"字（在特定上下文中）
    if keyword == '号' or keyword == '日':
        return True
    
    # ===== 4. 完整日期格式 =====
    
    # 4.1 月日组合：5月15、11.15、11-15、11/15
    if re.search(ARABIC_NUM_PATTERN + r'\s*月\s*' + ARABIC_NUM_PATTERN, keyword):
        return True
    if re.search(r'\d{1,2}[\./-]\d{1,2}', keyword):
        return True
    
    # 4.2 中文月日组合：五月十五
    if re.search(CHINESE_NUM_PATTERN + r'\s*月\s*' + CHINESE_NUM_PATTERN, keyword):
        return True
    
    # 4.3 年月日组合：2024年11月15日
    if re.search(ANY_NUM_PATTERN + r'\s*年\s*' + ANY_NUM_PATTERN + r'\s*月\s*' + ANY_NUM_PATTERN + r'\s*[日号]?', keyword):
        return True
    
    # ===== 5. 星期检测 =====
    
    # 5.1 星期：周X、星期X、礼拜X（使用通用模式）
    if re.search(r'周[一二三四五六日天]', keyword):
        return True
    if re.search(r'星期[一二三四五六日天]', keyword):
        return True
    if re.search(r'礼拜[一二三四五六日天]', keyword):
        return True
    
    # 5.2 工作日、周末
    if '工作日' in keyword or '周末' in keyword or '双休' in keyword:
        return True
    
    # ===== 6. 相对日期 =====
    
    relative_dates = [
        '今天', '今日', '今儿',
        '昨天', '昨日', '昨儿',
        '明天', '明日', '明儿',
        '前天', '前日',
        '后天', '后日',
        '大前天', '大后天'
    ]
    for date in relative_dates:
        if date in keyword:
            return True
    
    # ===== 7. 其他日期表达 =====
    
    # 7.1 月初、月中、月末、月底
    if re.search(r'月[初中末底]', keyword):
        return True
    
    # 7.2 上旬、中旬、下旬
    if '旬' in keyword and any(x in keyword for x in ['上', '中', '下']):
        return True
    
    # 7.3 季度：任意数字+季度、Q+数字
    if re.search(r'第?' + ANY_NUM_PATTERN + r'\s*季度', keyword):
        return True
    if re.search(r'[Qq]' + ARABIC_NUM_PATTERN, keyword):
        return True
    
    return False


def is_time_keyword(keyword):
    """
    判断关键词是否为时间相关词汇
    
    使用与预处理阶段相同的检测规则，确保一致性。
    
    检测规则（与 preprocess_v2.py 中的 remove_time_expressions 保持一致）：
    - 标准时间格式：8:40、08:40:30、8.40、8点40分30秒
    - 口语化时间：8点多、8点半、8点左右、差5分8点
    - 时间段：早上、上午、中午、下午、晚上、凌晨、夜里
    - 模糊时间：刚才、现在、马上、立刻、稍后
    
    Args:
        keyword: 待检测的关键词
        
    Returns:
        True 如果是时间关键词，否则 False
    """
    if not isinstance(keyword, str):
        return False
    
    # ===== 1. 标准时间格式 =====
    
    # 1.1 冒号分隔的时间：8:40、08:40:30、23:59:59
    if re.search(r'\d{1,2}:\d{1,2}(:\d{1,2})?', keyword):
        return True
    
    # 1.2 点号分隔的时间：8.40、8.40.30（排除价格）
    if re.search(r'(?<!\d)\d{1,2}\.\d{1,2}(\.\d{1,2})?(?!\d)', keyword):
        return True
    
    # 1.3 中文完整时间：8点40分30秒、8点40分、8点40、八点四十分（统一使用通用数字模式）
    if re.search(ANY_NUM_PATTERN + r'\s*点\s*' + ANY_NUM_PATTERN + r'\s*分\s*' + ANY_NUM_PATTERN + r'\s*秒', keyword):
        return True
    if re.search(ANY_NUM_PATTERN + r'\s*点\s*' + ANY_NUM_PATTERN + r'\s*分', keyword):
        return True
    if re.search(ANY_NUM_PATTERN + r'\s*点\s*' + ANY_NUM_PATTERN + r'(?![分秒])', keyword):
        return True
    
    # 1.4 中文时间单位：8点、40分、30秒、8时、40分钟（数字或中文数字）
    if re.search(ANY_NUM_PATTERN + r'\s*[点时]\s*(?:钟)?', keyword):
        return True
    if re.search(ANY_NUM_PATTERN + r'\s*分\s*(?:钟)?', keyword):
        return True
    if re.search(ANY_NUM_PATTERN + r'\s*秒\s*(?:钟)?', keyword):
        return True
    
    # ===== 2. 口语化时间表达 =====
    
    # 2.1 模糊时间：8点多、8点半、8点左右、8点钟左右（数字或中文数字）
    if re.search(ANY_NUM_PATTERN + r'\s*[点时]\s*[多半来钟]', keyword):
        return True
    if re.search(ANY_NUM_PATTERN + r'\s*[点时]\s*(?:左右|上下)', keyword):
        return True
    
    # 2.2 差几分几点：差5分8点、差一刻9点（数字或中文数字）
    if re.search(r'差\s*' + ANY_NUM_PATTERN + r'\s*分\s*' + ANY_NUM_PATTERN + r'\s*[点时]', keyword):
        return True
    if re.search(r'差\s*' + CHINESE_NUM_PATTERN + r'\s*刻\s*' + ANY_NUM_PATTERN + r'\s*[点时]', keyword):
        return True
    
    # 2.3 几点几刻：8点一刻、9点三刻（数字或中文数字）
    if re.search(ANY_NUM_PATTERN + r'\s*[点时]\s*' + CHINESE_NUM_PATTERN + r'\s*刻', keyword):
        return True
    
    # ===== 3. 时间段表达 =====
    
    # 3.1 时间段：早上、上午、中午、下午、晚上、夜里、凌晨、深夜
    time_periods = ['早上', '上午', '中午', '下午', '晚上', '夜里', '凌晨', '深夜', '早晨', '傍晚', '黄昏']
    for period in time_periods:
        if period in keyword:
            return True
    
    # 3.2 带时间段的完整表达：上午8点、晚上9点半（数字或中文数字）
    if re.search(r'[早上中下晚夜凌深][上午里晨间夜]\s*' + ANY_NUM_PATTERN + r'\s*[点时]', keyword):
        return True
    
    # ===== 4. 模糊时间词 =====
    
    fuzzy_time_words = [
        '刚才', '刚刚', '现在', '此刻', '当前', '目前',
        '马上', '立刻', '立即', '立马', '即刻',
        '稍后', '待会', '一会儿', '过会', '等会',
        '随后', '之后', '然后', '接着', '紧接着'
    ]
    for word in fuzzy_time_words:
        if word in keyword:
            return True
    
    return False
