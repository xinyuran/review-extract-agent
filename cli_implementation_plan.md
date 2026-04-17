# Extract Agent CLI 封装实施方案

## 1. 概述

将 `extract_agent` 封装为 CLI 工具，支持**单次命令模式**和**交互式 REPL 模式**两种使用方式，统一通过 `python -m extract_agent` 入口运行。

### 1.1 目标

- 支持终端直接输入评论文本或从文件读取
- 默认输出简化的人类可读格式（关键词 + 情感），可通过参数切换为完整 JSON 并持久化
- 完整 JSON 输出按 `日期/session/结果文件` 的目录结构存储
- Session 级别的 Agent 隔离，每个 session 独立拥有 Agent 实例和 Memory
- 通过配置文件传递 LLM 地址等参数
- 集成现有 FastAPI 服务启动

### 1.2 技术选型

| 组件 | 选型 | 理由 |
|------|------|------|
| CLI 框架 | **Typer** | 基于 type hints 自动生成参数，与 Pydantic/FastAPI 生态一致 |
| 配置文件 | **YAML** (`~/.extract-agent/config.yaml`) | 可读性好，结构化，嵌套配置友好 |
| 配置解析 | **PyYAML** | 轻量、成熟 |
| REPL 交互 | **prompt_toolkit** | 支持自动补全、历史记录、多行编辑 |
| 表格输出 | **rich** | 终端美化输出、表格、进度条 |

### 1.3 新增依赖

```
typer[all]>=0.12.0
pyyaml>=6.0
prompt-toolkit>=3.0
rich>=13.0
```

## 2. 命令结构设计

```
python -m extract_agent <command> [options]
```

### 2.1 命令一览

| 命令 | 说明 | 示例 |
|------|------|------|
| `analyze` | 单次分析模式 | `python -m extract_agent analyze "评论文本"` |
| `interactive` | 交互式 REPL 模式 | `python -m extract_agent interactive` |
| `serve` | 启动 FastAPI 服务 | `python -m extract_agent serve --port 8000` |
| `check` | 检测配置和 LLM 连通性 | `python -m extract_agent check` |
| `config` | 查看/初始化配置文件 | `python -m extract_agent config --init` |

### 2.2 `analyze` 命令

```bash
# 直接传入文本
python -m extract_agent analyze "这件衣服质量很好，做工精致"

# 从文件读取（txt：每行一条；csv：需指定列名；json：需指定字段）
python -m extract_agent analyze -f comments.txt
python -m extract_agent analyze -f comments.csv --column review_text
python -m extract_agent analyze -f comments.json --field text

# 切换模式
python -m extract_agent analyze "评论" --mode fast       # 快速模式
python -m extract_agent analyze "评论" --mode agent      # Agent 模式（默认）

# 完整 JSON 输出（自动存储到 output/YYYY-MM-DD/session_id/ 目录）
python -m extract_agent analyze "评论" --full

# 禁用反思
python -m extract_agent analyze "评论" --no-reflect
```

### 2.3 `interactive` 命令

```bash
# 启动交互式 REPL
python -m extract_agent interactive

# 指定模式
python -m extract_agent interactive --mode fast

# 启用完整输出
python -m extract_agent interactive --full
```

REPL 内置命令：

| 命令 | 说明 |
|------|------|
| 直接输入文本 | 分析该评论 |
| `/mode fast\|agent\|auto` | 切换分析模式 |
| `/full on\|off` | 切换完整 JSON 输出 |
| `/file <path>` | 从文件加载评论批量分析 |
| `/history` | 查看当前 session 的分析历史 |
| `/session` | 显示当前 session 信息（ID、已分析条数、存储路径） |
| `/help` | 帮助信息 |
| `/exit` 或 `Ctrl+D` | 退出 |

### 2.4 `serve` 命令

```bash
python -m extract_agent serve --host 0.0.0.0 --port 8000 --reload
```

直接集成现有 FastAPI app，底层调用 `uvicorn.run()`。

### 2.5 `check` 命令

```bash
python -m extract_agent check
```

输出示例：

```
[配置文件]  ~/.extract-agent/config.yaml  ✓ 已加载
[Agent LLM] http://192.168.12.42:8001/v1  ✓ 连通 (模型: Qwen2.5-7B-Instruct)
[Tool LLM]  http://192.168.12.42:8002/v1  ✓ 连通 (模型: finetuned-model)
[Redis]     redis://127.0.0.1:6379/0      ✗ 不可用 (异步任务不可用，不影响 CLI)
[工具]      5 个工具已加载 ✓
[Function Calling] native 模式 ✓
```

### 2.6 `config` 命令

```bash
# 初始化默认配置文件
python -m extract_agent config --init

# 查看当前生效的配置
python -m extract_agent config --show
```

## 3. 配置文件设计

### 3.1 配置文件路径

按以下优先级查找：

1. 命令行参数 `--config <path>`
2. 当前工作目录下的 `.extract-agent.yaml`
3. 用户目录下的 `~/.extract-agent/config.yaml`
4. 如果都不存在，使用代码中的默认值

### 3.2 配置文件格式

```yaml
# ~/.extract-agent/config.yaml

# Agent LLM 配置
agent_llm:
  base_url: "http://192.168.12.42:8001/v1"
  api_key: "dummy"
  model: "Qwen2.5-7B-Instruct"
  temperature: 0
  max_tokens: 800

# Tool LLM 配置
tool_llm:
  base_url: "http://192.168.12.42:8002/v1"
  api_key: "dummy"
  model: "/path/to/finetuned-model"
  temperature: 0
  max_tokens: 4096

# Agent 控制
agent:
  tool_calling_mode: "native"   # native / prompt / auto
  max_steps: 10
  timeout: 120
  tool_timeout: 30

# 反思器
reflection:
  enabled: true
  max_rounds: 5
  score_threshold: 0.7

# CLI 特有配置
cli:
  # 完整输出的存储根目录（相对于当前工作目录，或绝对路径）
  output_dir: "./extract_agent_output"
  # 默认分析模式
  default_mode: "agent"

# Redis（仅 serve 命令使用）
redis:
  url: "redis://127.0.0.1:6379/0"
```

### 3.3 配置加载与 AgentConfig 映射

新建 `cli/config_loader.py`，负责：

1. 按优先级查找并加载 YAML 文件
2. 将 YAML 字段映射到现有 `AgentConfig` 类的属性
3. 不破坏现有的环境变量读取逻辑——YAML 配置作为"中间层"覆盖默认值，环境变量仍可进一步覆盖

```
最终优先级：环境变量 > YAML 配置文件 > AgentConfig 代码默认值
```

## 4. Session 管理与 Agent 隔离

这是 CLI 封装中最关键的设计点。

### 4.1 核心概念

| 概念 | 说明 |
|------|------|
| **Session** | 一次 CLI 调用的生命周期。单次命令 = 一个 session；REPL 从启动到退出 = 一个 session |
| **Session ID** | 格式为 `{8位随机hex}`（如 `a3f1b2c9`），用于目录命名和追踪 |
| **Agent 实例** | 每个 session 持有独立的 `ReviewAnalysisAgent` 实例 |
| **Session 存储** | 完整输出按 `output_dir/YYYY-MM-DD/session_id/` 组织 |

### 4.2 Session 生命周期

```
┌───────────────────────────────────────────────────────────┐
│                      Session 生命周期                       │
├───────────────────────────────────────────────────────────┤
│                                                           │
│  ① 创建 Session                                          │
│     ├── 生成 session_id (8位hex)                          │
│     ├── 创建 Agent 实例 (ReviewAnalysisAgent)              │
│     └── 如果 --full 模式，创建存储目录                      │
│              output_dir/                                   │
│                └── 2026-04-14/                             │
│                    └── a3f1b2c9/                           │
│                        ├── session_meta.json               │
│                        ├── result_001.json                 │
│                        ├── result_002.json                 │
│                        └── ...                             │
│                                                           │
│  ② 分析评论（可重复多次，仅 REPL 模式）                     │
│     ├── 调用 Agent.run() —— 每次调用 Memory 是独立的        │
│     ├── 终端输出简化格式                                    │
│     └── 如果 --full，写入 result_NNN.json                  │
│                                                           │
│  ③ 销毁 Session                                          │
│     ├── 如果 --full，更新 session_meta.json 统计信息        │
│     └── Agent 实例释放                                     │
│                                                           │
└───────────────────────────────────────────────────────────┘
```

### 4.3 Agent 隔离分析

当前 `ReviewAnalysisAgent.run()` 的实现中：
- **每次调用 `run()` 都会创建全新的 `AgentMemory`**（见 `agent.py:294`）
- Agent 实例本身是**无状态的**——它持有的 `config`、`tools`、`_agent_client` 都是可复用的配置和连接
- 不同评论之间没有共享的 Memory，天然隔离

因此，Agent 隔离策略为：

| 场景 | Agent 实例策略 | Memory 隔离 |
|------|---------------|-------------|
| 单次命令 `analyze` | 创建一个 Agent，处理完所有输入后销毁 | 每条评论独立 Memory（已有行为） |
| REPL `interactive` | 整个 session 共享一个 Agent 实例 | 每条评论独立 Memory（已有行为） |
| 批量文件 `analyze -f` | 创建一个 Agent，批量处理后销毁 | 每条评论独立 Memory（已有行为） |

**结论**：由于 `run()` 内部每次都创建新 Memory，**同一个 Agent 实例可以安全地在一个 session 内复用**。不需要为每条评论创建新的 Agent 实例（那样会重复初始化 OpenAI client 和 tools，浪费资源）。

### 4.4 存储目录结构

当用户使用 `--full` 参数时，完整 JSON 按以下结构存储：

```
extract_agent_output/                       # 可在配置文件中自定义
└── 2026-04-14/                             # 当天日期
    ├── a3f1b2c9/                           # session_id
    │   ├── session_meta.json               # session 元信息
    │   ├── result_001.json                 # 第 1 条分析结果
    │   ├── result_002.json                 # 第 2 条分析结果
    │   └── result_003.json                 # ...
    └── f7e2d4b1/                           # 另一个 session
        ├── session_meta.json
        └── result_001.json
```

`session_meta.json` 内容：

```json
{
  "session_id": "a3f1b2c9",
  "created_at": "2026-04-14T10:32:15",
  "finished_at": "2026-04-14T10:45:30",
  "mode": "agent",
  "total_analyzed": 12,
  "cli_command": "interactive --full",
  "config_source": "~/.extract-agent/config.yaml"
}
```

### 4.5 Session 对象设计

```python
class CLISession:
    """CLI Session 管理器"""
    
    session_id: str              # 8位hex
    created_at: datetime
    agent: ReviewAnalysisAgent   # 整个 session 共享
    config: AgentConfig
    mode: str                    # fast / agent / auto
    full_output: bool            # 是否输出完整 JSON
    output_dir: Path | None      # 完整输出存储路径（--full 时才创建）
    result_counter: int          # 结果计数器
    
    def analyze(self, text: str) -> dict:
        """分析单条评论，返回结果并处理输出/存储"""
        ...
    
    def analyze_batch(self, texts: list[str]) -> list[dict]:
        """批量分析"""
        ...
    
    def save_result(self, result: dict) -> Path:
        """将完整结果写入文件"""
        ...
    
    def close(self):
        """关闭 session，更新 meta"""
        ...
```

## 5. 输出格式设计

### 5.1 简化格式（默认，终端直接显示）

单条分析：

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📝 评论: 这件衣服质量很好，做工精致，物流也快
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔑 关键词:
   1. 质量很好 (score: 0.95)
   2. 做工精致 (score: 0.92)
   3. 物流也快 (score: 0.88)

💬 情感: positive (confidence: 0.96)

⏱  耗时: 2340ms | 模式: agent-native | 步数: 5
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

批量分析（文件输入）：

```
分析进度: ━━━━━━━━━━━━━━━━━━━━ 100% 4/4

┌───┬───────────────────────┬──────────────────────────┬──────────┬────────┐
│ # │ 评论 (截断)            │ 关键词                    │ 情感     │ 耗时   │
├───┼───────────────────────┼──────────────────────────┼──────────┼────────┤
│ 1 │ 做工很好，面料柔软...   │ 做工好, 面料柔软, 穿着舒适 │ positive │ 1.2s   │
│ 2 │ 垃圾产品，用了一天...   │ 垃圾产品, 一天就坏, 退货   │ negative │ 1.5s   │
│ 3 │ 还行吧，一般般         │ 一般般                    │ neutral  │ 0.8s   │
│ 4 │ 物流超快，第二天就...   │ 物流超快, 包装好, 推荐购买 │ positive │ 1.3s   │
└───┴───────────────────────┴──────────────────────────┴──────────┴────────┘

总计: 4 条 | 成功: 4 | 失败: 0 | 总耗时: 4.8s
```

### 5.2 完整格式（`--full` 参数）

终端仍然显示简化格式，但同时将完整 JSON 写入文件，并在终端提示存储路径：

```
🔑 关键词: 质量很好, 做工精致, 物流也快
💬 情感: positive (0.96)
📁 完整结果已保存: extract_agent_output/2026-04-14/a3f1b2c9/result_001.json
```

## 6. 目录结构变更

在现有项目中新增 `cli/` 子包：

```
extract_agent/
├── ...（现有文件不变）
├── cli/                           # 新增：CLI 模块
│   ├── __init__.py
│   ├── main.py                    # Typer 应用入口 + 命令注册
│   ├── commands/                   # 各子命令实现
│   │   ├── __init__.py
│   │   ├── analyze.py             # analyze 命令
│   │   ├── interactive.py         # interactive REPL 命令
│   │   ├── serve.py               # serve 命令
│   │   ├── check.py               # check 命令
│   │   └── config_cmd.py          # config 命令
│   ├── session.py                 # CLISession 管理器
│   ├── config_loader.py           # YAML 配置加载 + AgentConfig 映射
│   ├── formatter.py               # 输出格式化（简化格式 + rich 表格）
│   └── file_reader.py             # 文件输入解析（txt/csv/json）
└── __main__.py                    # python -m extract_agent 入口
```

## 7. 实施阶段

### 第一阶段：基础骨架（优先级最高）

**目标**：能跑通最简单的 `python -m extract_agent analyze "评论"` 命令。

| 任务 | 文件 | 说明 |
|------|------|------|
| 1.1 | `__main__.py` | `python -m extract_agent` 入口，调用 `cli.main` |
| 1.2 | `cli/main.py` | Typer app 定义 + 命令注册 |
| 1.3 | `cli/commands/analyze.py` | `analyze` 命令基本实现（文本输入 + 简化输出） |
| 1.4 | `cli/formatter.py` | 简化格式输出（使用 rich） |
| 1.5 | `requirements.txt` | 添加 `typer`, `rich` 依赖 |

**验收标准**：
```bash
python -m extract_agent analyze "这件衣服质量很好" --mode fast
# 终端输出简化的关键词和情感结果
```

### 第二阶段：配置文件 + Session

**目标**：配置文件生效，`--full` 输出可持久化。

| 任务 | 文件 | 说明 |
|------|------|------|
| 2.1 | `cli/config_loader.py` | YAML 加载 + AgentConfig 映射 |
| 2.2 | `cli/commands/config_cmd.py` | `config --init` / `config --show` |
| 2.3 | `cli/session.py` | Session 创建、ID 生成、目录管理、结果写入 |
| 2.4 | 更新 `analyze` 命令 | 接入 config_loader + session |
| 2.5 | `requirements.txt` | 添加 `pyyaml` 依赖 |

**验收标准**：
```bash
python -m extract_agent config --init      # 生成默认配置文件
python -m extract_agent config --show      # 显示当前配置
python -m extract_agent analyze "评论" --full  # 结果写入文件
```

### 第三阶段：文件输入 + 批量分析

**目标**：支持从文件读取评论并批量分析。

| 任务 | 文件 | 说明 |
|------|------|------|
| 3.1 | `cli/file_reader.py` | txt/csv/json 文件解析 |
| 3.2 | 更新 `analyze` 命令 | 接入 file_reader，支持 `-f` 参数 |
| 3.3 | `cli/formatter.py` | 添加批量结果的 rich 表格输出 + 进度条 |

**验收标准**：
```bash
python -m extract_agent analyze -f comments.txt          # txt 批量
python -m extract_agent analyze -f data.csv --column text # csv 批量
```

### 第四阶段：交互式 REPL

**目标**：`python -m extract_agent interactive` 可进入交互式会话。

| 任务 | 文件 | 说明 |
|------|------|------|
| 4.1 | `cli/commands/interactive.py` | REPL 主循环 + 内置命令解析 |
| 4.2 | 更新 session.py | 支持 REPL 场景的多次分析 + 历史记录 |
| 4.3 | `requirements.txt` | 添加 `prompt-toolkit` 依赖 |

**验收标准**：
```bash
python -m extract_agent interactive
# 进入交互模式，可连续输入评论分析
# 输入 /mode fast 可切换模式
# 输入 /exit 退出
```

### 第五阶段：serve + check

**目标**：集成 FastAPI 启动和连通性检查。

| 任务 | 文件 | 说明 |
|------|------|------|
| 5.1 | `cli/commands/serve.py` | 封装 `uvicorn.run()` |
| 5.2 | `cli/commands/check.py` | LLM 连通性检测 + 配置校验 |

**验收标准**：
```bash
python -m extract_agent serve --port 8000   # 启动 API 服务
python -m extract_agent check               # 显示各组件状态
```

## 8. 关键设计决策记录

### 8.1 为什么 Agent 实例在 session 内复用？

`ReviewAnalysisAgent.run()` 内部每次调用都会创建全新的 `AgentMemory`（`agent.py:294`），不同评论之间没有共享 Memory。Agent 实例本身是无状态的（只持有 config、tools、OpenAI client），因此在同一个 session 内复用是安全的，避免了重复初始化的开销。

### 8.2 为什么不用环境变量而用配置文件？

- CLI 场景下用户更倾向于编辑一个固定的配置文件，而不是每次设置环境变量
- YAML 格式支持结构化嵌套，比 `.env` 可读性更好
- 环境变量仍然保留作为最高优先级覆盖手段，方便 CI/CD 和 Docker 场景

### 8.3 为什么 REPL 和单次命令用同一套 Session？

统一 Session 抽象简化了代码：
- 单次命令 = 创建 session → 分析 → 关闭 session
- REPL = 创建 session → 分析 → 分析 → ... → 关闭 session
- 输出路径、Agent 生命周期、元信息记录的逻辑完全一致

### 8.4 `__main__.py` 的作用

Python 通过 `python -m extract_agent` 运行包时会执行 `__main__.py`。该文件只需一行：

```python
from extract_agent.cli.main import app
app()
```

这不影响现有的 `from extract_agent.agent.agent import ReviewAnalysisAgent` 导入方式。

## 9. 与现有代码的兼容性

| 现有组件 | 影响 | 说明 |
|----------|------|------|
| `config.py` (AgentConfig) | **不修改** | CLI 的 config_loader 创建 AgentConfig 实例后覆盖属性值 |
| `agent/agent.py` | **不修改** | CLI 通过标准的 `ReviewAnalysisAgent(config).run()` 调用 |
| `api/` | **不修改** | `serve` 命令直接复用 `api.app:app` |
| `examples/run_agent.py` | **不修改** | 仍可独立运行 |
| `requirements.txt` | **追加依赖** | 新增 typer, rich, pyyaml, prompt-toolkit |

## 10. 未来扩展点

| 方向 | 说明 | 当前阶段不实现的原因 |
|------|------|---------------------|
| `pip install` 支持 | 添加 `pyproject.toml` + entry_points，全局可用 `extract-agent` 命令 | 当前 `python -m` 够用，后续需要时加 |
| 多 session 并发 | 同时运行多个 REPL session | 当前单用户 CLI 无需并发 |
| Session 恢复 | 从 `session_meta.json` 恢复中断的 session | 复杂度高，收益不明确，后续按需加 |
| 插件机制 | 支持用户自定义 tool 注册到 Agent | 需设计插件接口规范 |
