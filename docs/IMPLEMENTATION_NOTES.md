# ReproAgent 实现说明

> 2026-08 P0 加固：真实命令现在默认经由 fail-closed Docker 后端；任务结果使用版本化、attempt 绑定的 envelope；下游仅消费已验证依赖；实验按 static/unit/smoke/reduced/full 分级；验证记录独立持久化；mock 结果不得形成真实复现结论；CLI 使用显式运行结果与非零失败退出码；PDF 输入由专用解析路径处理。

本文档总结 `repro_agent/` 的实现思路、对开源项目设计的复用来源，以及关键的工程取舍，重点覆盖本轮迭代新增的**工具权限分级与授权机制**。所有条款编号（如 §12、§15.2）均对应 `doc/reproagent_system_design.md`。

## 1. 总体结构

```text
repro_agent/
├── domain/          # 纯数据模型：Job/Task/DAG/Experiment/Reflection
├── storage/          # SQLite 持久化（任务状态唯一事实来源，§3 原则 15-16）
├── scheduler/        # DAG 调度、租约心跳、超时策略、优先级排序
├── sandbox/           # 任务级物理隔离（input/workspace/output/tmp）
├── tools/             # 工具风险分级 + 授权（本轮核心新增）
├── agents/            # 10 个子智能体实现
├── memory/            # 渐进式 Markdown 记忆（L0~L3）
├── evaluation/        # 分级实验门禁、容差策略、结果差距判定
├── evidence/           # SHA-256 证据链、反作弊扫描
├── observability/     # 最终报告、复现结论判定
├── orchestrator/      # 主智能体主循环、规划/校验/重规划/反思编排
├── providers/         # LLM 供应商抽象（含 Token 递减重试）
└── cli/               # 命令行入口
```

## 2. 工具权限分级与授权机制（本轮核心需求）

**需求原文**："文件查找、文件阅读、资源阅读等这些封装为工具，子 agent 只能使用主 agent 传给他的、低风险的工具。"

### 2.1 风险分级

`tools/base.py::ToolRiskLevel` 三级：

- `READ_ONLY`：`find_files`/`read_file`/`grep_files`/`list_directory`/资源探测类，无副作用；
- `RESTRICTED_WRITE`：`write_file`/`write_task_output`，仅限沙箱 `workspace/`、`output/`；
- `HIGH_RISK`：`execute_command`/`build_environment_image`，通过隔离后端执行命令或构建环境。旧的 `git_worktree_apply` 兼容入口已禁用。

### 2.2 纵深防御式授权（`tools/authorization.py`）

`ToolAuthorizer.authorize()` 做两层校验：

1. **显式白名单**：任务定义 `TaskDefinition.allowed_tools`；
2. **任务类型风险预算**：`TASK_TYPE_RISK_BUDGET` 声明每种任务类型允许的最高风险等级（例如 `paper_analysis`/`code_analysis`/`resource_check`/`verification`/`reflection` 固定为 `READ_ONLY`），即便任务定义误配置了高危工具名，也会在这一层被拒绝并记录审计日志。

例外：`write_task_output` 被列入 `_ALWAYS_ALLOWED_TOOLS` 豁免名单——因为 §15.2 要求**所有**子智能体（包括纯只读分析类）都必须能产出 `result.json`/`candidate_memory.md`，这是任务的"最小必需能力"而非"额外提权"，其写入范围严格限定在该任务自己的 `output/` 目录。

### 2.3 子智能体永远拿不到全局注册表

`ToolAuthorization` 对象只持有"已过滤好的 `ToolSpec` 子集 + 沙箱上下文"，`BaseSubAgent.call_tool()` 是子智能体调用工具的唯一入口，物理上不存在拿到 `ToolRegistry` 或裸沙箱对象的路径。每次调用都会被记录进 `invocation_log`，供事后审计（`AgentDispatcher._record_tool_invocations`）。

### 2.4 派发闭环（`orchestrator/dispatcher.py`）

`AgentDispatcher.dispatch_and_run()` 串联：创建/复用沙箱 → 工具授权 → 按 `task_type` 查表得到子智能体类 → 用 `ToolAuthorization` 构造实例 → 运行并捕获 `ToolPermissionError`/`ToolExecutionError` 转换为标准 `FailureReport`。

## 3. 修复的关键设计缺陷（本轮排查发现）

在做端到端冒烟测试（`cli/main.py run --mock`）时发现并修复：

1. **`TaskDAG.ready_tasks()` 状态窗口错误**：`unblock_or_block()` 把任务转为 `READY` 后，`ready_tasks()` 只筛选 `PENDING/BLOCKED`，导致已就绪任务永远拿不到手。修复为同时包含 `READY` 状态。
2. **沙箱路径越界**：`paper_path`/`repository_path` 等宿主机绝对路径未被拷入沙箱就直接传给只读工具。修复为 `SandboxManager._stage_declared_inputs()` 统一把声明的路径字段拷贝进 `input/` 并原地重写为沙箱内相对路径。
3. **相对路径多根匹配歧义**：`validate_within_roots()` 对相对路径按 `roots` 列表顺序"命中第一个就返回"，导致 `"output/x"` 被误解析到排在最前的 `workspace_dir`/`input_dir` 下。修复为优先选择"实际存在"的候选路径，并新增 `SandboxContext.resolve_output_path()` 供 `write_task_output` 显式锚定 `output_dir`，不依赖猜测。
4. **`write_task_output` 风险预算冲突**：见 2.2 节的豁免名单修复。

以上问题均通过 `cli/main.py run --mock`（`providers.mock.MockLLMProvider`，无需真实 API Key）端到端复现并验证修复。

## 4. 复用的开源设计（摘录）

| 模块 | 复用来源 | 要点 |
|---|---|---|
| `storage/database.py` | Pi 项目 SQLite 使用方式 | WAL + busy_timeout，任务状态唯一事实来源 |
| `scheduler/lease.py` | DeerFlow RunStore Lease+心跳 | 区分"卡死"与"运行缓慢"，支持多 Worker 归属 |
| `sandbox/paths.py` | DeepCode `validate_path` | resolve() 前缀比较防路径穿越 |
| `tools/authorization.py` | DeepCode 索引增强模式 + DeerFlow Skills `allowed-tools` | 运行时收窄工具面、不放宽已有限制 |
| `evidence/hashing.py` | 通用 SHA-256 证据链 | 产物-代码-配置绑定，防止"拟合论文"式作弊 |

## 5. 已知的后续扩展点

- `orchestrator/main_agent.py::_latest_full_experiment_comparisons` 目前返回 `None`（骨架），需要在接入真实 `verification` 任务输出后补充 `MetricComparison` 组装逻辑；
- `cli/main.py` 的 `ReportInputs` 目前只传 `job`，需要在真实项目中从各任务的 `result.json` 组装完整报告输入；
- `execution/` 目录（分级实验真实执行器）仍是占位，`evaluation/tier_gate.py` 已提供门禁判定逻辑，可直接接入。
