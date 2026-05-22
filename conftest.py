"""pytest 全局配置"""

import sys
from pathlib import Path

# 确保 extract_agent 包在 sys.path 中
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
