# Extract Agent — 中文电商评论分析 AI Agent

基于 ReAct（Thought → Action → Observation）范式的评论分析 Agent，具备**自主规划、迭代式工具调用、代码级反思、异常降级**能力。采用**四层架构**（Skill → Agent → LLM Service → Tool），支持**轨迹数据采集与 SFT 导出**、**知识积累**和**多种使用方式**（Python API / FastAPI HTTP / 交互式 CLI）。

## 架构概览

```
用户请求 (Python API / FastAPI / CLI)
   │
   ▼
┌──────────────────────────────────────────────────────────────┐
│  ① Skill 层 (skills/*.skill.md)                              │
│     纯 Prompt 工程产物，Markdown + YAML frontmatter          │
│     定义 LLM 在每个阶段的角色、规则和输出格式                   │
├──────────────────────────────────────────────────────────────┤
│  ② Agent 层 (agent/)                                         │
│     ReviewAnalysisAgent — ReAct 主循环                        │
│     Thought → Action → Observation → 审查 → 修正 / 推进       │
│     调度 LLM-powered 工具 & 纯计算工具                         │
│     集成知识积累更新                                           │
├──────────────────────────────────────────────────────────────┤
│  ③ LLM Service 层 (llm_service/)                             │
│     统一 LLM 交互：call_agent / call_tool                     │
│     SkillLoader 解析 SKILL.md → 结构化 Prompt                 │
│     自动检测 Native FC vs Prompt-based 模式                    │
│     轨迹采集 (TrajectoryRecorder)                             │
├──────────────────────────────────────────────────────────────┤
│  ④ Tool 层 (tools/)                                          │
│     纯计算/API 执行，不感知调用上下文                            │
│     text_preprocess / keyword_extract / jieba_extract          │
│     validate_keywords / sentiment_analyze                      │
└──────────────────────────────────────────────────────────────┘
   │
   ▼
结构化分析报告 (JSON)
   ├──→ CLI 输出 / Session 持久化
   ├──→ 轨迹 JSONL (SFT 训练数据)
   ├──→ 知识积累 (评论者画像 / 商品画像)
   └──→ Redis (缓存/异步任务)
```

### 四层职责解耦

| 层级 | 职责 | 关键特征 |
|------|------|----------|
| **Skill 层** | 定义 LLM 角色、规则、输出格式 | 纯 Markdown，不含 Python 代码；支持 `{{variable}}` 变量注入 |
| **Agent 层** | ReAct 推理规划 + 工具调度 | 不关心 Prompt 细节，不直接调用 LLM API |
| **LLM Service 层** | 统一 LLM 调用、模式检测、轨迹采集 | 集中管理 retry / token / 调用模式 |
| **Tool 层** | 纯计算执行 | 不知道被哪个阶段调用，不含 LLM 调用逻辑 |

### Agent ReAct 工作模式

Agent 采用**迭代循环**方式工作，而非固定管线：

1. 每次调用工具后，审查结果质量
2. 如发现问题（重复、数量不足、质量不佳），回头重新调用相关工具修正
3. 只有当前步骤结果质量合格后，才推进到下一步
4. 代码层在 Agent 之后再做最终兜底去重

## 目录结构

```
extract_agent/
├── config.py                       # 全局配置（双 LLM endpoint + 反思器 + 轨迹 + 知识积累）
├── requirements.txt                # Python 依赖
├── .env.example                    # 环境变量模板
├── README.md                       # 本文件
├── __init__.py                     # 包入口
├── __main__.py                     # python -m extract_agent 入口
│
├── skills/                         # ① Skill 层 — 纯 Prompt 定义
│   ├── agent_system.skill.md       # Agent 系统 Prompt
│   ├── agent_system_tools.skill.md # Agent 系统 Prompt（含工具描述）
│   ├── user_request.skill.md       # 用户请求模板
│   ├── keyword_extract_long.skill.md   # 长评论关键词提取 Prompt
│   ├── keyword_extract_short.skill.md  # 短评论关键词提取 Prompt
│   ├── sentiment_analyze.skill.md  # 情感分析 Prompt
│   └── reflector.skill.md          # 反思器 Prompt
│
├── agent/                          # ② Agent 层 — ReAct 主循环
│   ├── agent.py                    # ReviewAnalysisAgent + 知识积累集成
│   ├── prompts.py                  # 兼容垫片（已迁移至 SKILL.md）
│   ├── memory.py                   # 工作记忆管理
│   └── reflector.py                # LLM 反思器
│
├── llm_service/                    # ③ LLM Service 层
│   ├── models.py                   # SkillPrompt / LLMResponse 数据模型
│   ├── skill_loader.py             # SKILL.md 解析器 + 变量注入
│   ├── service.py                  # LLMService（call_agent / call_tool / 模式检测）
│   └── trajectory.py               # TrajectoryRecorder 轨迹采集
│
├── tools/                          # ④ Tool 层 — 标准化工具
│   ├── base_tool.py                # BaseTool 抽象基类 + ToolResult
│   ├── preprocess_tool.py          # 文本预处理工具
│   ├── keyword_extract_tool.py     # 关键词提取工具
│   ├── jieba_extract_tool.py       # Jieba 兜底提取工具
│   ├── validate_tool.py            # 关键词校验 + 去重工具
│   └── sentiment_tool.py           # 情感分析工具
│
├── core/                           # 底层能力模块
│   ├── preprocess.py               # 文本预处理
│   ├── post_process.py             # 关键词后处理
│   ├── fallback_extractor.py       # Jieba 兜底提取
│   ├── prompt_template_3.py        # 关键词提取 Prompt（已迁移至 SKILL.md）
│   ├── prompt_template_4_shortComment.py  # 短评 Prompt（已迁移至 SKILL.md）
│   ├── sentiment_prompt.py         # 情感分析 Prompt（已迁移至 SKILL.md）
│   ├── reflector_prompt.py         # 反思器 Prompt（已迁移至 SKILL.md）
│   └── stopwords.txt               # 停用词表
│
├── trajectory/                     # 轨迹数据导出
│   ├── exporter.py                 # TrajectoryExporter（加载 session → 导出 SFT）
│   └── formats.py                  # SFTFormatter（Agent SFT / Tool SFT / 监督三元组）
│
├── knowledge/                      # 知识积累系统
│   ├── models.py                   # ReviewerProfile / ProductProfile 数据模型
│   ├── manager.py                  # KnowledgeManager（CRUD + 动态详细度调整）
│   └── reporter.py                 # KnowledgeReporter（Rich 终端报告）
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
│       ├── config_cmd.py           # config — 查看/初始化配置文件
│       ├── export.py               # export — 导出 SFT 训练数据
│       └── report.py               # report — 知识积累分析报告
│
├── tests/                          # 测试
│   ├── test_cli_phase1.py ~ test_cli_phase5.py  # CLI 分阶段测试
│   └── extra_test_api.py           # API 额外测试
│
├── examples/                       # 使用示例
│   ├── run_agent.py                # 运行示例脚本
│   └── extra_test_tool_calling.py  # 工具调用测试脚本
│
├── imgs/                           # 文档配图
│   └── image.png                   # 交互模式终端示例截图
│
└── extract_agent_output/           # CLI 运行输出（自动生成，已 gitignore）
    ├── <date>/<session_id>/        # 分析结果
    ├── trajectory/                 # 轨迹 JSONL 文件
    └── knowledge_store/            # 知识积累 JSON 文件
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 LLM 后端

项目支持三种后端模式：

| 模式 | 说明 | 配置方式 |
|------|------|----------|
| `cloud_api` | 云服务商 API（OpenAI / DeepSeek / 通义千问等兼容接口） | 设置 `AGENT_LLM_BASE_URL` 为云端地址 |
| `local_model` | 本地部署模型（vLLM / Ollama 等） | 设置 `AGENT_LLM_BASE_URL` 为本地地址 |
| `offline` | 无 LLM，仅 Jieba 提取 | 不配置或设为空 |

#### 云服务 API 示例

```bash
# .env 或环境变量
AGENT_LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
AGENT_LLM_MODEL=qwen-plus
AGENT_LLM_API_KEY=sk-your-key

TOOL_LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
TOOL_LLM_MODEL=qwen-plus
TOOL_LLM_API_KEY=sk-your-key
```

#### 本地 vLLM 部署

```bash
# Agent LLM（需启用 Function Calling）
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --port 8001 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes

# Tool LLM（微调模型）
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/your/finetuned-model \
  --port 8002
```

### 3. 配置文件

项目支持三种配置方式（优先级从高到低）：

1. **环境变量** — 直接设置系统环境变量或 `.env` 文件
2. **YAML 配置文件** — 通过 `config.yaml` 或 `.extract-agent.yaml`
3. **代码默认值** — `config.py` 中的默认配置

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

# 导出 SFT 训练数据
python -m extract_agent export sft --session-dir <path> --format agent
python -m extract_agent export stats --session-dir <path>

# 知识积累分析报告
python -m extract_agent report reviewer <reviewer_id>
python -m extract_agent report product <product_id>
python -m extract_agent report summary
```

#### CLI 子命令一览

| 子命令 | 说明 |
|--------|------|
| `analyze` | 单次/批量分析 — 分析评论文本并输出关键词与情感 |
| `interactive` | 交互式 REPL 模式 — 支持连续分析，内置丰富命令 |
| `serve` | 启动 FastAPI HTTP 服务 |
| `check` | 检测配置和各组件连通性（LLM、Redis、工具、FC 模式） |
| `config` | 查看当前配置或初始化 YAML 配置文件 |
| `export` | 导出 SFT 训练数据（Agent SFT / Tool SFT / 监督三元组） |
| `report` | 知识积累分析报告（评论者画像 / 商品画像 / 汇总统计） |

#### 交互式模式

交互式模式提供完整的 REPL 体验，支持连续分析、会话恢复和知识查询：

![交互模式示例](imgs/image.png)

**内置命令：**

| 命令 | 说明 |
|------|------|
| `/mode fast\|agent` | 切换分析模式 |
| `/full on\|off` | 切换完整 JSON 输出 |
| `/file <path>` | 从文件加载评论批量分析 |
| `/history` | 查看当前 session 的分析历史 |
| `/session` | 显示当前 session 信息 |
| `/resume [id]` | 恢复之前保存的 session 继续对话 |
| `/reviewer <id>` | 查看评论者画像 |
| `/reviewer list` | 列出所有已追踪的评论者 ID |
| `/product <id>` | 查看商品画像 |
| `/product list` | 列出所有已追踪的商品 ID |
| `/help` | 显示帮助信息 |
| `/exit` | 退出交互模式 |

#### CLI 输出

分析结果以 JSON 格式保存到 `extract_agent_output/` 目录下，按日期和会话 ID 组织：

```
extract_agent_output/
├── 2026-04-24/
│   └── <session_id>/
│       ├── session_meta.json    # 会话元数据
│       └── result.json          # 分析结果
├── trajectory/                  # 轨迹 JSONL（启用时）
│   └── <session_id>.jsonl
└── knowledge_store/             # 知识积累
    ├── reviewer_<id>.json
    └── product_<id>.json
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
from extract_agent.llm_service import LLMService

config = AgentConfig()
llm_service = LLMService(config)
agent = ReviewAnalysisAgent(config, llm_service=llm_service)

# Agent 模式：自主规划工具调用 + 代码级反思
result = agent.run("这件衣服质量很好，做工精致，物流也快")

# 快速模式：固定管线，跳过 Agent 推理
result = agent.run("画质清楚", use_fast_path=True)

# 带知识积累上下文
result = agent.run(
    "做工很好，面料柔软",
    reviewer_id="user_123",
    product_id="prod_456",
    product_name="纯棉T恤",
)
```

## Skill 层

所有 Prompt 统一定义为 `skills/*.skill.md` 文件，格式如下：

```markdown
---
name: keyword_extract_long
description: 长评论关键词提取
target: tool_llm
variables:
  - name: comment
    required: true
---

## system

你是一个中文电商评论关键词提取专家...

## user

请提取以下评论的关键词：{{comment}}
```

- **YAML frontmatter**：声明技能名、描述、目标 LLM、所需变量
- **`## system`**：系统 Prompt（可选，`user_request` 等模板无需此段）
- **`## user`**：用户 Prompt
- **`{{variable}}`**：运行时由 `SkillLoader` 注入实际值

当前内置技能：

| 技能文件 | 用途 | 目标 LLM |
|----------|------|----------|
| `agent_system.skill.md` | Agent 系统 Prompt | Agent LLM |
| `agent_system_tools.skill.md` | Agent 系统 Prompt（含工具描述） | Agent LLM |
| `user_request.skill.md` | 用户请求模板 | Agent LLM |
| `keyword_extract_long.skill.md` | 长评论关键词提取 | Tool LLM |
| `keyword_extract_short.skill.md` | 短评论关键词提取 | Tool LLM |
| `sentiment_analyze.skill.md` | 情感分析 | Tool LLM |
| `reflector.skill.md` | 反思器 | Agent LLM |

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

## 轨迹数据采集与 SFT 导出

### 轨迹采集

启用 `ENABLE_TRAJECTORY=true` 后，系统自动记录每次 LLM 交互的完整序列：

- **Agent LLM 轨迹**：thinking + tool_use + tool_result 完整序列
- **Tool LLM 轨迹**：Prompt → 原始输出 → 解析结果

轨迹以 JSONL 格式存储在 `TRAJECTORY_OUTPUT_DIR` 目录下。

### SFT 数据导出

支持三种训练数据格式：

| 格式 | 说明 | CLI 命令 |
|------|------|----------|
| Agent SFT | OpenAI native `tool_calls` 格式，用于训练 Agent LLM | `export sft --format agent` |
| Tool SFT | 工具 LLM 的输入输出对，用于训练 Tool LLM | `export sft --format tool` |
| 工具调用监督 | `tool_name + input + result` 三元组 | `export sft --format supervision` |

```bash
# 导出 Agent SFT 数据
python -m extract_agent export sft --session-dir extract_agent_output/2026-04-24/abc123 --format agent

# 查看轨迹统计
python -m extract_agent export stats --session-dir extract_agent_output/2026-04-24/abc123
```

## 知识积累系统

通过 `ENABLE_KNOWLEDGE=true` 环境变量开启（默认关闭）。开启后，每个 session 自动生成对应的 `reviewer_id` 和 `product_id`，分析完成后自动积累知识数据，无需手动传参。

### 评论者画像 (ReviewerProfile)

跨会话追踪评论者行为模式：
- 累计分析次数
- 关键词频次统计
- 情感倾向分布
- 正向标签 / 负面短板

### 商品画像 (ProductProfile)

全局模式统计：
- 累计被分析次数
- 高频关键词 Top-N
- 情感正负比例
- 典型评价摘要

### 动态详细度调整

根据评论者的累计分析次数，Agent 自动调整输出详细程度：
- 首次分析（< 5 次）：完整详细输出
- 常规分析（5-20 次）：标准输出
- 高频分析（> 20 次）：仅输出增量变化

### 知识查询

```bash
# CLI 命令
python -m extract_agent report reviewer <reviewer_id>
python -m extract_agent report product <product_id>
python -m extract_agent report summary

# 交互模式 — 查看画像
>> /reviewer <reviewer_id>
>> /product <product_id>

# 交互模式 — 列出所有已追踪的 ID
>> /reviewer list
>> /product list
```

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
| Agent LLM | 推理规划、迭代式工具调度 | Qwen2.5-7B-Instruct（原始）或云端模型 | `AGENT_LLM_BASE_URL` / `AGENT_LLM_MODEL` |
| Tool LLM | 关键词提取、情感分析 | SFT+DPO+GRPO 微调模型或云端模型 | `TOOL_LLM_BASE_URL` / `TOOL_LLM_MODEL` |

**Prompt 隔离**：两个 LLM 使用完全不同的 Prompt 体系，均定义在 `skills/` 目录的 SKILL.md 文件中：
- Agent LLM 使用 `agent_system.skill.md` / `agent_system_tools.skill.md`
- Tool LLM 使用 `keyword_extract_long.skill.md`（关键词提取）和 `sentiment_analyze.skill.md`（情感分析）

## 两种工具调用模式

| 模式 | 说明 | 配置 |
|------|------|------|
| Native Function Calling | vLLM 原生工具调用，需 `--enable-auto-tool-choice --tool-call-parser hermes` | `AGENT_TOOL_CALLING_MODE=native` |
| Prompt-based | 通过 `<tool_call>` 标签解析，不依赖服务端支持 | `AGENT_TOOL_CALLING_MODE=prompt` |

`LLMService` 支持 `auto` 模式，启动时自动探测后端是否支持 Native Function Calling 并选择最优模式。

## 三种运行模式

### Agent 模式 (`mode=agent`)

- Agent LLM 自主决策工具调用顺序
- 迭代式执行：发现结果有问题时会回头修正
- 支持异常自动降级（LLM 提取失败 → Jieba 兜底）
- 完成后进行代码级反思
- 适合复杂/长评论

### 快速模式 (`mode=fast`)

- 固定管线：预处理 → 关键词提取 → 校验 → 情感分析
- 无 Agent LLM 推理开销，延迟低
- 适合批量处理、短评论

### 自动模式 (`mode=auto`)

- 短评（< 30 字）走快速路径，长评走 Agent 模式

## 代码级反思机制

反思采用**代码级确定性逻辑**而非完全依赖 LLM 判断：

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
| `text_preprocess` | 文本清洗（去 URL/emoji/乱码等） | 否 | 纯计算工具 |
| `keyword_extract` | 关键词提取（含思考链捕获） | Tool LLM | 通过 LLM Service 调用 |
| `jieba_extract` | Jieba 分词兜底提取 | 否 | `keyword_extract` 失败时降级使用 |
| `validate_keywords` | 去重 + 质量校验 | 否 | 重复去除、停用词过滤、长度检查、原文对齐 |
| `sentiment_analyze` | 情感分析 | Tool LLM | 通过 LLM Service 调用 |

## 全部配置项

| 分类 | 配置项 | 默认值 | 说明 |
|------|--------|--------|------|
| Agent LLM | `AGENT_LLM_BASE_URL` | `http://192.168.12.42:8001/v1` | Agent LLM 服务地址 |
| Agent LLM | `AGENT_LLM_MODEL` | Qwen2.5-7B-Instruct | Agent LLM 模型名称 |
| Agent LLM | `AGENT_LLM_API_KEY` | — | API 密钥（云端模式必填） |
| Agent LLM | `AGENT_LLM_TEMPERATURE` | `0` | 生成温度 |
| Agent LLM | `AGENT_LLM_MAX_TOKENS` | `4096` | 最大生成 token 数 |
| Tool LLM | `TOOL_LLM_BASE_URL` | `http://192.168.12.42:8002/v1` | Tool LLM 服务地址 |
| Tool LLM | `TOOL_LLM_MODEL` | 微调模型路径 | Tool LLM 模型名称 |
| Tool LLM | `TOOL_LLM_API_KEY` | — | API 密钥（云端模式必填） |
| Tool LLM | `TOOL_LLM_TEMPERATURE` | `0` | 生成温度 |
| Tool LLM | `TOOL_LLM_MAX_TOKENS` | `4096` | 最大生成 token 数 |
| Agent 控制 | `AGENT_TOOL_CALLING_MODE` | `native` | 工具调用模式（native/prompt/auto） |
| Agent 控制 | `AGENT_MAX_STEPS` | `10` | ReAct 最大步数 |
| Agent 控制 | `AGENT_TIMEOUT` | `120` | Agent 总超时（秒） |
| Agent 控制 | `TOOL_TIMEOUT` | `30` | 单工具超时（秒） |
| 反思器 | `ENABLE_REFLECTION` | `true` | 是否启用反思 |
| 反思器 | `REFLECTION_MAX_ROUNDS` | `5` | 反思最大轮数 |
| 反思器 | `REFLECTION_SCORE_THRESHOLD` | `0.7` | Score 合格阈值 |
| Skill 层 | `SKILLS_DIR` | `extract_agent/skills/` | SKILL.md 文件目录 |
| 轨迹采集 | `ENABLE_TRAJECTORY` | `false` | 是否启用轨迹采集 |
| 轨迹采集 | `TRAJECTORY_OUTPUT_DIR` | `extract_agent_output/trajectory` | 轨迹 JSONL 输出目录 |
| 轨迹采集 | `TRAJECTORY_INCLUDE_THINKING` | `true` | 轨迹是否包含 thinking 内容 |
| 知识积累 | `ENABLE_KNOWLEDGE` | `false` | 是否启用知识积累（开启后 session 自动生成 ID） |
| 知识积累 | `KNOWLEDGE_STORE_DIR` | `extract_agent_output/knowledge_store` | 知识 JSON 存储目录 |

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
