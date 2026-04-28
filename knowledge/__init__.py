"""
知识积累系统

管理评论者画像（个体结构体）和商品画像（客体结构体），
实现跨序列追踪、动态输出调整和分析报告生成。
"""

from .models import ReviewerProfile, ProductProfile
from .manager import KnowledgeManager

__all__ = ["ReviewerProfile", "ProductProfile", "KnowledgeManager"]
