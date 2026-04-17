"""
Extract Agent 使用示例

展示三种使用方式：
1. Agent 模式：由 ReAct Agent 自主规划工具调用
2. 快速模式：跳过 Agent 推理，按固定管线执行
3. 批量分析：对多条评论进行批量处理
"""

import json
import logging
import sys
import os
import importlib

# 将「包含本包目录的上一级」加入 sys.path，包内一律用相对导入；此处用目录名动态 import，便于整体搬迁或重命名包文件夹。
_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_ROOT = os.path.dirname(_EXAMPLES_DIR)
_REPO_ROOT = os.path.dirname(_PACKAGE_ROOT)
_PACKAGE_NAME = os.path.basename(os.path.normpath(_PACKAGE_ROOT))
sys.path.insert(0, _REPO_ROOT)

_cfg = importlib.import_module(f"{_PACKAGE_NAME}.config")
AgentConfig = _cfg.AgentConfig
_agent = importlib.import_module(f"{_PACKAGE_NAME}.agent.agent")
ReviewAnalysisAgent = _agent.ReviewAnalysisAgent


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def print_result(result: dict, title: str = "分析结果"):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"{'=' * 60}\n")


def demo_agent_mode():
    """示例 1：Agent 模式 —— Agent 自主决策工具调用顺序"""
    print("\n>>> 示例 1：Agent 模式（ReAct 自主规划）")

    config = AgentConfig()
    # 可通过修改 config 或环境变量自定义 LLM 地址：
    # config.AGENT_LLM_BASE_URL = "http://your-agent-llm:8001/v1"
    # config.TOOL_LLM_BASE_URL = "http://your-tool-llm:8002/v1"

    agent = ReviewAnalysisAgent(config)

    comment = "买了一段时间，都快吃完了才想起没评价。湾仔码头全线产品都很好，长期回购。日期新近，材料新鲜，味道自然，早餐宵夜都很好，云吞小小一个，和面条搭配，孩子最爱吃，配套的汤配料也很出彩。个人感觉京东的湾仔比起某团的冷冻效果好些，放了超多冰袋送来，直接到家，不用担心在提货点被老板误放普通的冷冻室而融掉。物流赞！"

    result = agent.run(comment, use_fast_path=False)
    print_result(result, "Agent 模式分析结果")


def demo_fast_mode():
    """示例 2：快速模式 —— 跳过 Agent 推理，按固定管线执行"""
    print("\n>>> 示例 2：快速模式（固定管线）")

    agent = ReviewAnalysisAgent()

    comment = "画质清楚，性价比高"

    result = agent.run(comment, use_fast_path=True)
    print_result(result, "快速模式分析结果")


def demo_batch_mode():
    """示例 3：批量分析"""
    print("\n>>> 示例 3：批量分析")

    agent = ReviewAnalysisAgent()

    comments = [
        "做工很好，面料柔软，穿着舒适",
        "垃圾产品，用了一天就坏了，退货！",
        "还行吧，一般般",
        "物流超快，第二天就到了，包装也很好，推荐购买！",
    ]

    results = agent.run_batch(comments, use_fast_path=True)

    for r in results:
        idx = r.get("batch_index", "?")
        print(f"\n[评论 {idx}] {r.get('original_text', '')[:40]}...")
        kws = r.get("keywords", [])
        kw_str = ", ".join(k.get("keyword", "") for k in kws) if kws else "(无)"
        sentiment = r.get("sentiment", {})
        print(f"  关键词: {kw_str}")
        print(f"  情感: {sentiment.get('label', '?')} (confidence={sentiment.get('confidence', 0)})")
        print(f"  耗时: {r.get('elapsed_ms', 0)} ms")


def demo_custom_config():
    """示例 4：自定义配置"""
    print("\n>>> 示例 4：自定义配置")

    config = AgentConfig()

    # 示例：使用不同的 Agent LLM 和 Tool LLM
    # config.AGENT_LLM_BASE_URL = "http://agent-server:8001/v1"
    # config.AGENT_LLM_MODEL = "Qwen2.5-72B-Instruct"
    # config.TOOL_LLM_BASE_URL = "http://tool-server:8002/v1"
    # config.TOOL_LLM_MODEL = "Qwen2.5-7B-Instruct"

    # 调整 Agent 行为参数
    config.AGENT_MAX_STEPS = 15
    config.AGENT_TIMEOUT = 180
    config.N = 10  # 最多返回 10 个关键词

    agent = ReviewAnalysisAgent(config)

    comment = "东西很好"
    # comment = "物美价廉，物流给力，值得回购是正品，而且价格不贵，用着也很大气，一直在用，很好�✨物流和产品都不错，性价比高，赞赞赞����收到立马打开看了一下，非常不错，质量超好，颜值又高�，推荐推荐��收到立马打开看了一下，非常不错，质量超好，颜值又高�，推荐推荐�宝贝已收到�，超级喜欢，颜值高又实用，超级喜欢！❥❥❥很精致，质量非常好"
    result = agent.run(comment)
    print_result(result, "自定义配置分析结果")


if __name__ == "__main__":
    setup_logging()

    print("=" * 60)
    print("  Extract Agent 使用示例")
    print("=" * 60)
    print("\n注意：运行前请确保 vLLM 服务已启动，")
    print("或通过环境变量设置 AGENT_LLM_BASE_URL / TOOL_LLM_BASE_URL。\n")

    # 取消注释要运行的示例：
    # demo_agent_mode()
    # demo_fast_mode()
    # demo_batch_mode()
    demo_custom_config()

    # print("请取消注释 run_agent.py 底部的 demo 函数来运行示例。")
