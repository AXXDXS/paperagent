# ReproAgent 小型评测集

这是一个面向当前项目的内部回归数据集，不追求成为通用 benchmark。它主要回答三个问题：

1. 智能体能否从论文和代码中找对实验、参数与入口；
2. 遇到小故障时，能否修复后真实执行，而不是伪造结果；
3. 当资源缺失或实验确实复现失败时，能否给出正确终态并保留证据。

## 数据集组成

数据集共 12 条：

| 类别 | 数量 | 用途 |
| --- | ---: | --- |
| `direct` | 4 | 两条工作区真实实验（SINDy、PIFT）和两条轻量合成成功案例 |
| `repair` | 3 | 入口错误、配置类型错误、vendored 模块路径错误 |
| `gap` | 2 | 两个确定性的数值/趋势差距 |
| `blocked` | 2 | 数据集或 checkpoint 明确缺失 |
| `invalid_input` | 1 | 扩展名为 PDF、内容却不是 PDF 的输入 |

`dataset.json` 是总索引；每条用例位于 `cases/<case_id>/`，包含：

- `task.json`：交给智能体的输入和执行约束；
- `gold.json`：仅供评分器使用的预期终态、指标、参数与证据要求。

合成用例的论文和仓库放在 `fixtures/`。真实用例不复制大文件，而是引用工作区中的 `paper-replication-paper-main/case_studies`。

## 路径与执行约定

所有 `paper_path`、`repository_path` 和资源路径都相对于工作区根目录解析。`dataset.json` 中的 `path_base` 指明了从本目录回到工作区根目录的相对路径。

`run.command` 不是强制智能体逐字照抄的唯一命令，而是一个可复核的参考入口。执行适配器应替换：

- `{repository}`：仓库的**可写暂存副本**根目录；
- `{output}`：本次运行的输出目录。

真实仓库中的脚本会写入仓库内的 `artifacts/`，因此不应直接在只读挂载上执行。

实际发给智能体时，只传 `paper_path`、`repository_path`、`target_experiment`、`inputs`、`resources` 和 `budget`。`category`、`source_kind`、`known_challenge`、参考 `run` 以及整个 `gold.json` 都属于评测器侧元数据，不应进入智能体上下文，否则会泄露题型或答案。

## 推荐使用方式

先跑开发集，调通报告适配与评分，再只跑一次评测集：

```bash
cd /path/to/paper_agent
python eval_dataset/scripts/validate_dataset.py
```

开发集与评测集定义在 `splits.json`。为避免调参污染结果，开发阶段只看 `development`，方案稳定后再跑 `evaluation`。

对某个 ReproAgent `final_report.json` 评分：

```bash
python eval_dataset/scripts/score_report.py \
  eval_dataset/cases/case_001_sindy_linear \
  /absolute/path/to/final_report.json
```

评分满分 100：终态 25、执行真实性 25、观测指标 25、关键参数 10、证据完整性 10、预算 5。对于 `gap`、`blocked`、`invalid_input`，如果智能体错误宣布 `FULLY_REPRODUCED` 或 `PARTIALLY_REPRODUCED`，会触发 0 分硬规则。

## 重要说明

- `gold.json` 里的 `expected_observed_metrics` 是复核智能体是否读到/跑出正确结果；`paper_metrics` 是论文声称的目标，两者不能混用。
- gap 用例的程序都能正常退出，但结果不满足论文容差，因此正确答案是 `VERIFIED_REPRODUCTION_GAP`，不是把容差放宽后宣告成功。
- blocked 用例中的缺失路径是故意的，校验器会确认它们确实不存在。
- 真实深度学习用例可能比较慢；`budget.max_runtime_seconds` 是单条用例的宽松上限，不要求每次开发都全部重跑。

## 扩展数据集

新增用例时：

1. 复制一份现有 `task.json` 和 `gold.json`；
2. 使用新的连续 `case_id`；
3. 将用例登记进 `dataset.json` 和 `splits.json`；
4. 运行 `validate_dataset.py`；
5. 不要用论文目标值充当实际观测值，也不要把暂存运行产生的文件提交回原始真实仓库。
