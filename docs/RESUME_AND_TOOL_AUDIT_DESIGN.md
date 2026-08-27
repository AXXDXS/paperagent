# 断点续跑与工具调用审计设计

## 目标

系统进程中断后，应当能够从同一 `work-dir` 恢复指定 Job，而不是重新创建
Job 或重跑已经完成的论文分析、代码分析等任务。恢复必须保持既有的安全
边界：未经过结果契约校验的输出不能被当作成功；有副作用的操作不能因为
恢复而被静默重复或假定已经完成。

本设计覆盖三个恢复层次：

1. **工作流级恢复**：从 SQLite 重建 Job、Task DAG、实验记录、验证记录和
   反思报告。已是 `SUCCEEDED` 的任务保持完成，PhaseCoordinator 从第一个未满足
   的门禁继续创建下游任务。
2. **任务级恢复**：进程中断时处于 `DISPATCHED` 或 `RUNNING` 的任务进入
   `RECOVERING`。如果该 attempt 已经留下可验证的 `output/result.json`，恢复器先
   独立校验并转为 `SUCCEEDED`；否则创建新的 attempt 并转回 `PENDING`。
3. **子 Agent 级恢复**：子 Agent 在安全步骤边界保存 JSON 检查点；新的 attempt
   可以复用已完成的只读工具结果和 LLM 响应，避免重新阅读论文、重新扫描代码
   或重复模型调用。执行命令、写文件、创建隔离仓库副本等有副作用的动作不从缓存
   返回，必须由任务实现明确地做幂等恢复或以新 attempt 重跑。

## 状态机变更

任务增加 `RECOVERING` 状态，恢复状态转换为：

```text
DISPATCHED / RUNNING
        |
        v (进程重启)
   RECOVERING
   ├─ 可验证的旧 attempt 输出 ──> SUCCEEDED
   └─ 没有可信输出 ─────────────> PENDING ──> READY ──> 新 attempt
```

`SUCCEEDED` 是唯一允许解锁 DAG 依赖的状态。恢复逻辑不会把一个仅写出部分
文件、但没有合法 result envelope 的任务标记成功。

## 持久化模型

新增两个 SQLite 表：

- `task_checkpoints`：以 `(task_id, checkpoint_key)` 为键，保存 JSON payload、
  产生 checkpoint 的 attempt、时间和版本。Checkpoint 是“最后一个可安全重用的
  逻辑步骤”，不是 Python 线程/调用栈快照。
- `tool_invocations`：逐次记录 `invocation_id`、Job/Task/attempt、序号、工具名、
  脱敏参数、结果摘要、可恢复结果、结果状态、是否来自恢复缓存及时间。记录在
  工具调用结束时立即写库，而不是等整个 Agent 结束，因而中途崩溃也保留已完成
  调用的审计轨迹。

参数和结果均进行敏感字段脱敏并限制大小；`write_task_output` 的正文不写入审计
表。工具结果只在 `READ_ONLY` 风险级工具上允许重放。

## 恢复算法

`RecoveryService.resume(job_id)` 执行以下步骤：

1. 从数据库读取 Job；终态 Job 拒绝恢复。
2. 构造 MainAgent 时由 TaskScheduler 重建任务 DAG；恢复器从持久化数据恢复
   ReflectionReport 和待审计集合。
3. 逐个处理 `DISPATCHED` / `RUNNING` 任务：查找该 task/attempt 对应沙箱的
   `output/result.json`，使用原 attempt ID、任务类型和 expected outputs 做
   独立校验。校验成功则写入 outputs 并标记 `SUCCEEDED`；否则记录恢复事件，
   释放过期租约、清空运行时归属并设为 `PENDING`。
4. 重新运行主循环。DAG 只会调度尚未成功或新创建的任务，因此已完成的论文/代码
   阅读不会重跑。

恢复器不尝试重连 Python 线程。Docker 的 `--rm` 容器以及当前执行后端也没有可
安全接管的持久运行句柄；若需要“正在训练的进程继续执行”，训练脚本必须自行
支持 checkpoint/`--resume`，这是后续执行后端接管功能的边界。

## CLI

保留现有 `run` 命令；新增：

```bash
python -m repro_agent.cli.main resume --work-dir ./repro_agent_workdir --job-id job_xxx
```

`resume` 从同一 work-dir 的 SQLite 读取 Job 和任务，而不是接受新的论文或仓库
参数。LLM、Docker/Mock 等运行配置沿用 CLI 参数；恢复过程输出与 `run` 相同的
最终 Markdown/JSON 报告。

## 安全与一致性原则

- 绝不把旧 attempt 的结果写入新 attempt；恢复成功仅使用旧 attempt 仍为 active
  attempt 的已验证 envelope。
- 不自动缓存或重放高风险/有副作用工具。
- 所有恢复决策、状态转换和工具调用均写入 task event 或专用审计表。
- 恢复测试必须覆盖：已完成上游任务不重跑；中断任务合法输出被接纳；无输出任务
  重试并获得新 attempt；只读工具与 LLM 检查点可复用；有副作用工具不复用。
