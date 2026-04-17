# Extract Agent — 中文电商评论分析 AI Agent

基于 ReAct（Thought → Action → Observation）范式的评论分析 Agent，具备**自主规划、迭代式工具调用、代码级反思、异常降级**能力。支持 **Python API**、**FastAPI HTTP 服务**和**交互式 CLI** 三种使用方式。

## 架构概览

```
用户请求 (Python API / FastAPI / CLI)
   │
   ▼
┌──────────────────────────────────────────────────────────────┐
│              ReviewAnalysisAgent (ReAct)                      │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐     │
│  │  Agent LLM (Qwen2.5-7B-Instruct 原始模型)            │     │
│  │  职责：推理规划 + 工具调度（不负责格式化输出）          │     │
│  │  Thought → Action → Observation → 审查结果质量         │     │
│  │  → 发现问题则回头修正 → 直到满意才推进下一步            │     │
│  └───────────────────────┬─────────────────────────────┘     │
│                          │                                    │
│                  选择并调用工具（可迭代调用）                    │
│                          │                                    │
│       ┌──────────────────┴──────────────────────┐            │
│       │          Tool Registry                   │            │
│       │  ├── text_preprocess      (代码)          │            │
│       │  ├── keyword_extract  ← Tool LLM (微调)   │            │
│       │  ├── jieba_extract        (代码)          │            │
│       │  ├── validate_keywords    (代码+去重)      │            │
│       │  └── sentiment_analyze ← Tool LLM (微调)  │            │
│       └─────────────────────────────────────────┘            │
│                          │                                    │
│                    代码层组装结果                               │
│                          │                                    │
│       ┌──────────────────┴──────────────────────┐            │
│       │       代码级反思 (Code-level Reflection)   │            │
│       │  1. Score 阈值过滤不合格关键词              │            │
│       │  2. 代码级数量达标检查                      │            │
│       │  3. LLM 辅助补充（需通过原文对齐校验）       │            │
│       │  4. 无有效新增 → 智能终止                   │            │
│       └─────────────────────────────────────────┘            │
│                          │                                    │
│                    代码层最终去重                               │
└──────────────────────────────────────────────────────────────┘
   │
   ▼
结构化分析报告 (JSON)  ──→  Redis (缓存/异步任务)
```

### 核心设计：三层职责解耦

| 层级 | 职责 | 实现 |
|------|------|------|
| Agent LLM | 推理规划 + 迭代式工具调度 | Qwen2.5-7B-Instruct 原始模型 |
| Tool LLM | 关键词提取 + 情感分析 | SFT+DPO+GRPO 微调模型 |
| 代码层 | 结果组装 + 去重 + 反思循环管理 | Python 确定性逻辑 |

### Agent ReAct 工作模式

Agent 不再按固定管线依次走一遍，而是采用**迭代循环**方式工作：

1. 每次调用工具后，审查结果质量
2. 如发现问题（重复、数量不足、质量不佳），回头重新调用相关工具修正
3. 只有当前步骤结果质量合格后，才推进到下一步
4. 代码层在 Agent 之后再做最终兜底去重

## 目录结构

```
extract_agent/
├── config.py                       # 全局配置（双 LLM endpoint + 反思器配置）
├── requirements.txt                # Python 依赖
├── run_vllm.sh                     # vLLM 部署脚本
├── README.md                       # 本文件
├── __init__.py                     # 包入口
├── __main__.py                     # python -m extract_agent 入口
│
├── agent/                          # Agent 核心
│   ├── agent.py                    # ReAct Agent 主循环 + 代码级反思 + 最终去重
│   ├── prompts.py                  # Agent LLM 系统 Prompt（迭代式 ReAct 引导）
│   ├── memory.py                   # 工作记忆管理
│   └── reflector.py                # LLM 反思器（辅助补充关键词）
│
├── core/                           # 底层能力模块
│   ├── preprocess.py               # 文本预处理
│   ├── post_process.py             # 关键词后处理
│   ├── fallback_extractor.py       # Jieba 兜底提取
│   ├── prompt_template_3.py        # 关键词提取 Prompt（微调训练时使用）
│   ├── prompt_template_4_shortComment.py  # 短评 Prompt
│   ├── sentiment_prompt.py         # 情感分析 Prompt（微调训练时使用）
│   ├── reflector_prompt.py         # 反思器 Prompt
│   └── stopwords.txt               # 停用词表
│
├── tools/                          # 标准化工具层
│   ├── base_tool.py                # BaseTool 抽象基类 + ToolResult
│   ├── preprocess_tool.py          # 文本预处理工具
│   ├── keyword_extract_tool.py     # LLM 关键词提取工具（含 thinking 捕获）
│   ├── jieba_extract_tool.py       # Jieba 兜底提取工具
│   ├── validate_tool.py            # 关键词校验 + 去重工具
│   └── sentiment_tool.py           # 情感分析工具
│
├── api/                            # FastAPI 接口层
│   ├── app.py                      # FastAPI 应用入口
│   ├── routes.py                   # 路由定义
│   ├── schemas.py                  # 请求/响应 Pydantic 模型
│   └── redis_client.py             # Redis 客户端封装
│
├── cli/                            # 命令行界面
│   ├── main.py                     # Typer 应用入口，注册所有子命令
│   ├── session.py                  # 会话管理（目录、元数据、结果持久化）
│   ├── formatter.py                # Rich 终端格式化输出与进度条
│   ├── file_reader.py              # 多格式文件读取（txt/csv/json）
│   ├── config_loader.py            # YAML 配置查找、解析与优先级合并
│   └── commands/                   # 子命令实现
│       ├── analyze.py              # analyze — 单次/批量分析
│       ├── interactive.py          # interactive — 交互式 REPL 模式
│       ├── serve.py                # serve — 启动 FastAPI HTTP 服务
│       ├── check.py                # check — 检测配置和组件连通性
│       └── config_cmd.py           # config — 查看/初始化配置文件
│
├── tests/                          # 测试
│   ├── test_cli_phase1.py          # CLI 骨架、help、analyze 参数测试
│   ├── test_cli_phase2.py          # CLI 第二阶段测试
│   ├── test_cli_phase3.py          # CLI 第三阶段测试
│   ├── test_cli_phase4.py          # CLI 第四阶段测试
│   ├── test_cli_phase5.py          # CLI 第五阶段测试
│   └── extra_test_api.py           # API 额外测试
│
├── examples/                       # 使用示例
│   ├── run_agent.py                # 运行示例脚本
│   ├── extra_test_tool_calling.py  # 工具调用测试脚本
│   ├── example_run_no_reflect.log  # 无反思输出示例
│   └── example_run_with_reflect.log # Agent 模式 + 反思输出示例
│
└── extract_agent_output/           # CLI 运行输出目录（自动生成，已 gitignore）
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动 vLLM 服务

需要启动两个 vLLM 实例，分别服务 Agent LLM 和 Tool LLM：

```bash
# Agent LLM（原始 Qwen2.5-7B-Instruct，需启用 Function Calling）
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --port 8001 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes

# Tool LLM（微调模型，无需 Function Calling）
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/your/finetuned-model \
  --port 8002
```

也可以使用项目提供的部署脚本 `run_vllm.sh`。

### 3. 配置

项目支持三种配置方式（优先级从高到低）：

1. **环境变量** — 直接设置系统环境变量
2. **YAML 配置文件** — 通过 `config.yaml` 或 `.extract-agent.yaml`
3. **代码默认值** — `config.py` 中的默认配置

#### 方式一：环境变量

```bash
# Agent LLM
export AGENT_LLM_BASE_URL=http://127.0.0.1:8001/v1
export AGENT_LLM_MODEL=Qwen2.5-7B-Instruct

# Tool LLM（微调模型）
export TOOL_LLM_BASE_URL=http://127.0.0.1:8002/v1
export TOOL_LLM_MODEL=/path/to/your/finetuned-model

# Redis（异步任务和缓存需要）
export REDIS_URL=redis://127.0.0.1:6379/0

# 反思器
export ENABLE_REFLECTION=true
export REFLECTION_MAX_ROUNDS=5
export REFLECTION_SCORE_THRESHOLD=0.7
```

#### 方式二：YAML 配置文件

使用 CLI 初始化配置文件：

```bash
python -m extract_agent config --init
```

配置文件查找顺序：
1. CLI `--config` 参数指定的路径
2. 当前目录下的 `.extract-agent.yaml`
3. 用户目录 `~/.extract-agent/config.yaml`

## 使用方式

### 方式一：CLI 命令行

项目提供完整的命令行工具，通过 `python -m extract_agent` 调用：

```bash
# 查看帮助
python -m extract_agent --help

# 单次分析
python -m extract_agent analyze "这件衣服质量很好，做工精致，物流也快"

# 指定模式分析
python -m extract_agent analyze "画质清楚，色彩好" --mode fast

# 批量分析文件
python -m extract_agent analyze --file comments.txt --mode auto

# 交互式 REPL 模式
python -m extract_agent interactive

# 检测配置和各组件连通性
python -m extract_agent check

# 启动 HTTP 服务
python -m extract_agent serve --port 8000

# 查看/初始化配置文件
python -m extract_agent config --show
python -m extract_agent config --init
```

#### CLI 子命令一览

| 子命令 | 说明 |
|--------|------|
| `analyze` | 单次/批量分析 — 分析评论文本并输出关键词与情感 |
| `interactive` | 交互式 REPL 模式 — 支持连续分析，内置 `/mode`、`/file`、`/resume` 等命令 |
| `serve` | 启动 FastAPI HTTP 服务 |
| `check` | 检测配置和各组件连通性（LLM、Redis、工具、FC 模式） |
| `config` | 查看当前配置或初始化 YAML 配置文件 |

#### 交互式模式内置命令

在 `interactive` 模式下，支持以下内置命令：

| 命令 | 说明 |
|------|------|
| `/mode [agent\|fast\|auto]` | 切换分析模式 |
| `/file <路径>` | 从文件加载评论进行分析 |
| `/resume [会话ID]` | 恢复历史会话 |
| `/help` | 显示帮助信息 |
| `/quit` | 退出交互模式 |

#### CLI 输出

分析结果以 JSON 格式保存到 `extract_agent_output/` 目录下，按日期和会话 ID 组织：

```
extract_agent_output/
└── 2026-04-17/
    └── <session_id>/
        ├── session_meta.json    # 会话元数据
        └── result.json          # 分析结果
```

### 方式二：FastAPI HTTP 服务

```bash
# 通过 CLI 启动
python -m extract_agent serve --host 0.0.0.0 --port 8000

# 或直接用 uvicorn
uvicorn extract_agent.api.app:app --host 0.0.0.0 --port 8000 --reload
```

启动后访问 http://localhost:8000/docs 查看交互式 API 文档（Swagger UI）。

### 方式三：Python API 直接调用

```python
from extract_agent.config import AgentConfig
from extract_agent.agent.agent import ReviewAnalysisAgent

agent = ReviewAnalysisAgent(AgentConfig())

# Agent 模式：自主规划工具调用 + 代码级反思
result = agent.run("这件衣服质量很好，做工精致，物流也快")

# 快速模式：固定管线，跳过 Agent 推理
result = agent.run("画质清楚", use_fast_path=True)
```

## 输出结构

Agent 模式的完整输出 JSON 包含以下字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `analysis_complete` | bool | 分析是否成功完成 |
| `original_text` | string | 原始评论文本 |
| `cleaned_text` | string | 预处理后的文本 |
| `keywords` | array | 关键词列表（含 keyword、reasoning、score，已去重） |
| `keyword_thinking` | string | Tool LLM 的完整关键词提取思考链 |
| `sentiment` | object | 情感分析结果（label、confidence、reasoning） |
| `agent_summary` | string | Agent LLM 的自然语言分析总结 |
| `reflection` | object | 反思记录（轮数、是否通过、历史详情） |
| `agent_trace` | array | Agent 推理轨迹（每步的 thought 和 action） |
| `elapsed_ms` | number | 总耗时（毫秒） |
| `steps` | number | ReAct 步数 |
| `mode` | string | 运行模式（agent-native / agent-prompt） |

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/analyze` | 单条评论分析 |
| POST | `/api/analyze/batch` | 批量评论分析（同步） |
| POST | `/api/task/submit` | 提交异步分析任务 |
| GET | `/api/task/{task_id}` | 查询异步任务状态 |
| GET | `/health` | 健康检查 |

### 单条分析示例

```bash
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "做工很好，面料柔软，穿着舒适", "mode": "fast"}'
```

### 批量分析示例

```bash
curl -X POST http://localhost:8000/api/analyze/batch \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["做工很好", "垃圾产品，退货！"],
    "mode": "fast"
  }'
```

## 双 LLM 架构

| 角色 | 用途 | 模型 | 配置项 |
|------|------|------|--------|
| Agent LLM | 推理规划、迭代式工具调度 | Qwen2.5-7B-Instruct（原始） | `AGENT_LLM_BASE_URL` / `AGENT_LLM_MODEL` |
| Tool LLM | 关键词提取、情感分析 | Qwen2.5-7B-Instruct（SFT+DPO+GRPO 微调） | `TOOL_LLM_BASE_URL` / `TOOL_LLM_MODEL` |

**Prompt 隔离**：两个 LLM 使用完全不同的 Prompt 体系：
- Agent LLM 使用 `agent/prompts.py` 中的迭代式 ReAct 系统 Prompt
- Tool LLM 使用 `core/prompt_template_3.py`（关键词提取）和 `core/sentiment_prompt.py`（情感分析），与微调训练时使用的 Prompt 保持一致

## 两种工具调用模式

| 模式 | 说明 | 配置 |
|------|------|------|
| Native Function Calling | vLLM 原生工具调用，需 `--enable-auto-tool-choice --tool-call-parser hermes` | `AGENT_TOOL_CALLING_MODE=native` |
| Prompt-based | 通过 `<tool_call>` 标签解析，不依赖服务端支持 | `AGENT_TOOL_CALLING_MODE=prompt` |

Native 模式下，Hermes 解析器会将 `<tool_call>` 标签前的文本保留到 `message.content`，即 Agent 的 Thought 部分。

## 三种运行模式

### Agent 模式 (`mode=agent`)

- Agent LLM 自主决策工具调用顺序
- 迭代式执行：发现结果有问题时会回头修正，而不是带着问题继续
- 支持异常自动降级（LLM 提取失败 → Jieba 兜底）
- 完成后进行代码级反思
- 适合复杂/长评论

### 快速模式 (`mode=fast`)

- 固定管线：预处理 → 关键词提取 → 校验 → 情感分析
- 无 Agent LLM 推理开销，延迟低
- 适合批量处理、短评论

### 自动模式 (`mode=auto`)

- 短评（< 30 字）走快速路径，长评走 Agent 模式

## 多层去重机制

为避免 7B Agent LLM 在构造工具参数时引入重复关键词，系统实现了多层去重防线：

| 层级 | 位置 | 说明 |
|------|------|------|
| Tool 层 | `keyword_extract_tool` | 提取结果的后处理去重 |
| 校验层 | `validate_tool` | 校验时检测并移除重复项 |
| 组装层 | `agent.py` `_assemble_result_from_memory` | 最终结果组装时的兜底去重 |

## 代码级反思机制

反思采用**代码级确定性逻辑**而非完全依赖 LLM 判断，避免 LLM 幻觉导致的死循环：

```
Tool LLM 提取关键词（含 score）
        │
        ▼
  ① Score 阈值过滤
  （score < REFLECTION_SCORE_THRESHOLD 的关键词被移除）
        │
        ▼
  ② 代码级数量检查
  （根据原文长度判断是否达标）
        │
    ┌───┴───┐
  达标     不足
    │       │
  通过   ③ 调用 LLM 反思器尝试补充
    │       │
    │    ④ 双重校验：
    │      · score >= 阈值？
    │      · 在原文中存在？（原文对齐）
    │       │
    │    ⑤ 有效新增？
    │    ┌──┴──┐
    │   有    无
    │    │   智能终止（接受当前结果）
    │    │
    │   合并后重新检查
    │    │
    ▼    ▼
  最终结果（按 score 降序排列）
```

### 关键词数量要求（按原文长度分级）

| 原文长度 | 最低关键词数 | 配置项 |
|----------|:----------:|--------|
| < 20 字 | 2 | `REFLECTION_MIN_KEYWORDS_SHORT` |
| 20 ~ 60 字 | 5 | `REFLECTION_MIN_KEYWORDS_MEDIUM` |
| 60 ~ 120 字 | 8 | `REFLECTION_MIN_KEYWORDS_LONG` |
| >= 120 字 | 10 | `REFLECTION_MIN_KEYWORDS_XLONG` |

## 工具一览

| 工具名 | 功能 | 调用 LLM | 说明 |
|--------|------|:--------:|------|
| `text_preprocess` | 文本清洗（去 URL/emoji/乱码等） | 否 | 代码实现 |
| `keyword_extract` | 关键词提取（含思考链捕获） | Tool LLM | 使用微调 Prompt，输出 thinking + JSON |
| `jieba_extract` | Jieba 分词兜底提取 | 否 | `keyword_extract` 失败时降级使用 |
| `validate_keywords` | 去重 + 质量校验 | 否 | 重复去除、停用词过滤、长度检查、原文对齐 |
| `sentiment_analyze` | 情感分析 | Tool LLM | 使用微调 Prompt，输出正面/负面/中性 |

## 全部配置项

| 分类 | 配置项 | 默认值 | 说明 |
|------|--------|--------|------|
| Agent LLM | `AGENT_LLM_BASE_URL` | `http://192.168.12.42:8001/v1` | Agent LLM 服务地址 |
| Agent LLM | `AGENT_LLM_MODEL` | Qwen2.5-7B-Instruct | Agent LLM 模型名称 |
| Agent LLM | `AGENT_LLM_TEMPERATURE` | `0` | 生成温度 |
| Agent LLM | `AGENT_LLM_MAX_TOKENS` | `4096` | 最大生成 token 数 |
| Tool LLM | `TOOL_LLM_BASE_URL` | `http://192.168.12.42:8002/v1` | Tool LLM 服务地址 |
| Tool LLM | `TOOL_LLM_MODEL` | 微调模型路径 | Tool LLM 模型名称 |
| Tool LLM | `TOOL_LLM_TEMPERATURE` | `0` | 生成温度 |
| Tool LLM | `TOOL_LLM_MAX_TOKENS` | `4096` | 最大生成 token 数 |
| Agent 控制 | `AGENT_TOOL_CALLING_MODE` | `native` | 工具调用模式（native/prompt/auto） |
| Agent 控制 | `AGENT_MAX_STEPS` | `10` | ReAct 最大步数 |
| Agent 控制 | `AGENT_TIMEOUT` | `120` | Agent 总超时（秒） |
| Agent 控制 | `TOOL_TIMEOUT` | `30` | 单工具超时（秒） |
| 反思器 | `ENABLE_REFLECTION` | `true` | 是否启用反思 |
| 反思器 | `REFLECTION_MAX_ROUNDS` | `5` | 反思最大轮数 |
| 反思器 | `REFLECTION_SCORE_THRESHOLD` | `0.7` | Score 合格阈值 |

## 测试

```bash
# CLI 单元测试（分阶段）
pytest extract_agent/tests/test_cli_phase1.py -v
pytest extract_agent/tests/test_cli_phase2.py -v
pytest extract_agent/tests/test_cli_phase3.py -v
pytest extract_agent/tests/test_cli_phase4.py -v
pytest extract_agent/tests/test_cli_phase5.py -v

# 运行全部测试
pytest extract_agent/tests/ -v

# 仅运行不需要 LLM 服务的单元测试
pytest extract_agent/tests/ -m "not integration" -v
```

## 运行示例

```bash
cd extract_agent/examples
python -B -X utf8 run_agent.py > ./example_run_with_reflect.log
```
