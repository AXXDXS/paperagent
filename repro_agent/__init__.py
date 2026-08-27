"""ReproAgent —— 论文实验自动复现多智能体系统。

系统定位（详见 /doc/reproagent_system_design.md）：
    主智能体 + 多个沙箱化子智能体 + 确定性任务调度器
    + 渐进式记忆 + 上下文管理 + 反思审计闭环

本包遵循设计文档第 21 节推荐的目录结构，并在实现中吸收了以下三个
开源项目的健壮性工程经验（在对应模块的 docstring 中会再次注明出处）：

- DeepCode（Paper2Code）：机械化完成判定、Token 递减式重试、
  计划评审门禁、CodeRAG 参考检索、"写完即清空"记忆压缩策略。
- DeerFlow：Lease + 心跳的任务/Worker 归属模型、Checkpoint
  full/delta 快照、Middleware Chain 关注点分离、Fail-Closed 语义。
- paper-replication-paper：基于 SHA-256 的证据链、单活跃目标状态机、
  按科学含义分类的验收模式、反作弊禁用词扫描。

详细的复用来源与改动说明见仓库根目录的
``paper_agent/CHANGES_AND_DESIGN_NOTES.md``。
"""

__version__ = "0.1.0"
