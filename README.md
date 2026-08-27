# ReproAgent

ReproAgent 是一个面向论文实验复现的多智能体编排原型。主 Agent 负责创建任务 DAG、下发最小工具权限、验证子 Agent 结果、按层级运行实验，并从持久化证据生成报告。

## 安装

```bash
python -m pip install -e .
```

PDF 论文由 `pypdf` 解析；扫描版 PDF 需要先由用户提供 OCR 或文本版本。真实实验执行默认要求本机已准备好 Colima 或已有 Docker daemon；也可以显式选择可信本地 Conda 后端。
项目依赖必须使用 requirements 风格文件完整锁定。仓库中存在 wheel 时优先离线安装；没有 wheel 时，环境构建阶段可以联网解析锁定依赖，实验运行阶段仍使用独立的网络设置。
环境镜像按基础镜像的实际 digest、Dockerfile、锁定依赖和完整构建上下文生成稳定指纹；不同 Job 再次运行相同项目且构建输入未变化时，会直接复用本机的 `repro-agent/env-cache:<fingerprint>` 镜像，不再执行 `docker build`。缓存命中后仍会运行 import 自检；如果自检失败，系统会绕过 Docker 层缓存强制重建一次，避免把损坏的环境交给后续实验。

## 运行

### 一键离线 Demo

在仓库根目录直接执行（不需要先安装 console script）：

```bash
python run_demo.py
```

安装项目后也可以执行：

```bash
repro-agent demo
```

等价的模块入口是 `python -m repro_agent.cli.main demo`。

Demo 使用 `examples/demo/` 中内置的最小论文和确定性阈值分类器，完整
经过论文/代码分析、任务 DAG、环境准备、五级实验、独立验证和报告生成。
它使用 Mock LLM 与 Mock Executor，因此不需要 API Key、Docker、GPU 或
网络；最终结论会如实标记为 `PIPELINE_ONLY`，不会冒充真实论文复现。

### 本地 LLM 配置

真实模型调用使用 OpenAI 兼容协议。请复制
`configs/config.example` 为 `configs/config`，填入本地凭证，并限制文件
权限：

```bash
cp configs/config.example configs/config
chmod 600 configs/config
```

`configs/config` 被 Git 忽略，程序不会输出其中的凭证。配置优先级为：
命令行 `--model`、环境变量、私密配置文件、默认值。环境变量
`REPRO_AGENT_CONFIG_FILE` 可指定私密配置文件的其他位置。为兼容旧的
本地配置，`APP_ID`/`APPID` 仍可作为 API Key 的别名；程序不会根据该
别名推断服务地址或模型。使用非默认服务时必须在本地同时配置
`REPRO_AGENT_API_BASE` 和 `REPRO_AGENT_MODEL`。

默认输出到 `./repro_agent_demo_output`，也可以指定目录：

```bash
repro-agent demo --work-dir ./runs/demo
```

成功后主要产物包括：

- `final_report.md` / `final_report.json`：兼容入口报告；
- `reports/<job_id>/final_report.md`：按 Job 隔离的报告；
- `repro_agent.db`：Job、Task、attempt、工具审计和验证记录；
- `sandbox/`：各子 Agent 的输入、工作区、日志和输出产物。

### 使用自己的论文和代码

正式执行默认使用 Colima 提供 Linux VM 和 Docker daemon。首次运行前在
macOS 上安装并启动运行时：

```bash
brew install colima docker
colima start --cpu 4 --memory 8 --disk 60
docker info
docker pull python:3.11-slim
```

ReproAgent 不会自动安装或启动 Colima，也不会在运行时不可用时回退到
宿主机执行。若 `colima status` 或 `docker info` 失败，环境构建/实验任务
会返回带修复命令的明确错误。环境镜像构建保持离线，因此
`--execution-image` 指定的基础镜像需要提前拉取。需要连接已有 Docker
daemon 时，可以传入 `--container-runtime docker`。

```bash
python -m repro_agent.cli.main run \
  --paper-path /path/to/paper.pdf \
  --repository-path /path/to/repository \
  --target-experiment main \
  --container-runtime colima \
  --execution-image python:3.11-slim \
  --work-dir ./repro_agent_workdir
```

不希望配置容器运行时时，可以使用本机已有的 Conda（或兼容的 mamba）创建
内容寻址环境。任务代码仍复制到每次 attempt 独立的 `sandbox/` 工作区。Conda
prefix 默认使用代码仓库目录名，保存在 `work-dir/conda_envs/<项目名>/`；也可用
`--environment-name` 指定更简洁的名称。后续实验仍只接收
`conda://<fingerprint>` 引用，完整 fingerprint 保存在环境 marker 中作为缓存真值。
相同 Python 版本、锁文件和本地 wheel 内容会复用同名环境；fingerprint 变化时
会安全重建该同名 prefix。缓存命中后仍执行 import 自检。

```bash
python -m repro_agent.cli.main run \
  --paper-path /path/to/paper.pdf \
  --repository-path /path/to/repository \
  --target-experiment main \
  --environment-backend conda \
  --environment-name emem \
  --conda-executable conda \
  --conda-python-version 3.11 \
  --work-dir ./repro_agent_workdir
```

Conda 模式会保留超时、取消、日志上限、attempt 工作区和凭证按变量名透传，
但它是宿主机上的可信本地执行模式，不提供容器等价的只读根文件系统、Linux
capability、网络或 CPU/内存硬隔离。因此只应对可信代码使用；不可信仓库仍应
选择 Colima/Docker。`resume` 未显式指定后端时，会从已有环境任务中恢复原后端；
也可以再次传入同一组 Conda 参数。

推荐用 5 个 `--run-command` 依次声明 static/unit/smoke/reduced/full
层级的真实命令。只有一条通用命令时可以运行诊断流程，但不会被当成
“已验证的分级实验契约”。

规格生成完成后先执行资源检查。资源检查通过后，系统用一个 Human-in-the-loop
请求同时展示缺失的模型名/API 地址/凭证变量名，以及五级命令、基础镜像、
工作目录、超时和 CPU/内存/磁盘/GPU 的完整运行计划。只有用户一次性完成
配置并确认后才会构建环境。若用户在这次确认中修改资源参数，系统会按最终
值自动复检资源，再进入环境构建。环境生成的不可变镜像 digest 属于已确认
基础镜像的派生产物，不会造成实验前重复询问。

可以用 `--max-total-runtime-seconds`、`--max-gpu-hours` 和
`--max-model-cost-usd` 设置硬预算；模型费用预算还需要同时配置
输入、输出 token 单价。达到边界后系统会停止派发新任务，等待在途任务
安全收口后以明确失败状态退出。

离线检查编排流程时可以使用 `--mock`：

```bash
python -m repro_agent.cli.main run \
  --mock \
  --paper-path /path/to/paper.txt \
  --repository-path /path/to/repository
```

Mock 运行会经过与真实运行相同的阶段，但报告会明确标为诊断模式，最终状态最多为 `PIPELINE_ONLY`，不会声称论文已被真实复现。

CLI 会生成 `final_report.md` 和 `final_report.json`。达到迭代上限返回退出码 2；资源阻塞、失败、取消分别返回 3、4、5；等待人工输入返回 6。

实验命令出现 Python traceback、语法错误或测试失败时，主 Agent 会把失败
命令、层级和诊断日志交给独立的 `CodingAgent`。修复只发生在 attempt 级
仓库副本中，并且必须新增、通过最小回归测试；验证成功后，原实验任务使用
该修复副本从失败层级重新执行。每次修复与实验重跑都受原任务的
`max_attempts` 限制，不会无限修改或重跑，也不会直接改动用户仓库。

## Human-in-the-loop

权限错误、数据/模型错误、资源超限、资源检查发现缺失项，或者系统无法
确定实验入口、或论文与代码中的实验参数冲突时，Job 会停止派发新任务
并持久化一条结构化介入请求。代码分析还会用真实文件与行号声明那些
“无有效默认值、缺失后实验必然失败”的用户配置，例如模型名、模型 API
地址或凭证环境变量。资源检查完成后，系统在同一个请求中展示这些待配置项
以及完整五级命令、基础镜像、工作目录、超时、GPU 数和指标文件位置。用户
一次性配置并提交 `{ "approved": true }` 后，环境构建才会派发；若资源值
被修改则先自动复检，正常实验不会重复询问同一计划。旧任务恢复或运行参数与已确认计划不一致时，
实验前的精确确认仍作为安全兜底。网络默认关闭，只有用户确认的必需模型
API 地址会启用 bridge 网络；容器模式的代码工作区只读，Conda 模式使用可写
但与用户仓库隔离的 attempt 副本。内置 `demo` 是非执行的 Mock 演示，因此
不触发该确认门。
等待状态不会消耗主循环迭代次数，也可以跨进程重启继续处理。

查看请求及其回答格式：

```bash
python -m repro_agent.cli.main intervention list \
  --job-id job_xxx \
  --work-dir ./repro_agent_workdir
```

根据请求的 `input_schema` 提交数据、模型路径、运行命令或资源设置。推荐
使用文件，避免较长回答出现在 shell 历史中：

```bash
python -m repro_agent.cli.main intervention respond \
  --request-id intervention_xxx \
  --response-file ./answer.json \
  --responded-by owner \
  --work-dir ./repro_agent_workdir

python -m repro_agent.cli.main resume \
  --job-id job_xxx \
  --work-dir ./repro_agent_workdir
```

例如数据请求的 `answer.json` 可以是：

```json
{"dataset_paths": ["/absolute/path/to/dataset"]}
```

必需实验配置请求会给出代码证据和精确的 `input_schema`。模型名/API 地址
等非敏感值放在 `values`；API Key/Token 不得写进回答文件，应先导出到运行
ReproAgent 的环境中，回答里只确认变量名，例如：

```bash
export OPENAI_API_KEY=...  # 仅存在于当前进程环境，不写入 SQLite
```

```json
{
  "values": {
    "MODEL_NAME": "paper-model-v2",
    "MODEL_API_BASE": "https://models.example/v1"
  },
  "confirmed_secret_env_vars": ["OPENAI_API_KEY"]
}
```

已确认的非敏感值会按代码证据绑定为命令行参数或环境变量；凭证通过执行后端
的变量名透传，密钥值不会进入任务、介入请求、命令参数或数据库。默认实验
容器禁网；只有代码明确要求并且用户确认了合法的模型 API 地址后，该实验
容器才使用 bridge 网络。确认的 API 主机名会进入审计记录，但当前 Docker/
Colima 后端尚未实现主机名级的出口白名单。

权限请求可以显式批准或拒绝：

```bash
python -m repro_agent.cli.main intervention approve \
  --request-id intervention_xxx \
  --tool read_file \
  --responded-by security-reviewer \
  --work-dir ./repro_agent_workdir

python -m repro_agent.cli.main intervention deny \
  --request-id intervention_xxx \
  --reason "不允许该操作" \
  --work-dir ./repro_agent_workdir
```

人工批准仅能把工具加入目标 Task 的白名单，不能突破任务类型风险预算、
`forbidden_actions` 或网络隔离。拒绝或超时会以 fail-closed 方式终止
Job。可在 `run` 时通过 `--hitl-timeout-seconds` 设置等待期限；默认永久
等待。介入状态、回答者和回答字段会进入 SQLite 与最终报告审计记录，
报告不会复制回答的原始值。

## 断点续跑与审计

同一 `work-dir` 中的 Job 发生进程中断后，可按 Job ID 恢复：

```bash
python -m repro_agent.cli.main resume \
  --job-id job_xxx \
  --work-dir ./repro_agent_workdir
```

恢复会复用已经通过独立校验的任务结果。对于中断时处于派发或运行状态
的任务，系统会先校验旧 attempt 的 `output/result.json`：合法产物会被
接纳，否则任务会带着同一任务检查点重新排队并生成新 attempt。论文
阅读、代码预扫描和无工具调用的 LLM 分析结果会保存为安全检查点，避免
恢复后重复读取或重复模型调用。写文件、执行命令和训练等有副作用动作
不会被自动假定成功或重放。

如果 Job 处于等待人工状态，直接执行 `resume` 只会再次返回等待状态；
必须先回答、批准或拒绝对应的介入请求。

## 大仓库代码理解

代码分析不会把整个仓库直接塞入模型上下文。系统先使用无外部服务的
轻量索引扫描受支持的代码、配置和文档文件：Python 通过标准库 AST 提取
类、函数、签名、导入与引用，其他常见语言使用轻量声明提取。依赖目录、
构建产物、二进制、大文件和总索引预算之外的内容会被跳过并计入覆盖统计。

分析流程固定为：

```text
文件清单与内容摘要
  -> 按目标实验生成 token-budgeted Repo Map
  -> 路径/内容/符号混合检索
  -> 按类、函数和真实行号读取证据
  -> 最多 4 轮模型追加检索
  -> 结构化结论 + repository digest + 代码证据
```

同一进程内索引按文件大小和修改时间增量复用；恢复任务还会复用代码扫描
检查点。默认最多索引 5000 个文件、代码上下文预算约 10000 tokens，可在
任务输入中用 `code_index_max_files`（100-20000）和
`code_context_budget_tokens`（4000-30000）调整。所有预加载与模型主动
调用产生的工具结果，都会先经过无损 JSON、工具输出 Schema、ContentBlock
渲染和脱敏/截断，再进入模型上下文。

每次工具调用完成后，脱敏后的参数、结果摘要、结果状态、attempt 和时间
都会立即写入 SQLite 的 `tool_invocations` 审计表；因此即使子 Agent 在
后续步骤崩溃，已完成工具调用仍可追溯。
任务每一次 attempt、执行租约、状态转换事件、产物哈希和执行清单也会分别持久化；
容器控制句柄会在恢复时被对账，避免崩溃后留下无主实验。

## 执行边界

真实实验命令默认只通过 Colima/Docker 后端执行，不回退到宿主机 shell。正式容器禁用网络、使用只读根文件系统、移除 Linux capabilities，并设置 CPU、内存和进程数限制。显式选择 `--environment-backend conda` 时，命令通过控制面管理的 Conda prefix 在可信宿主机模式运行，安全边界如上节所述。两种模式都为每次任务重试使用独立 attempt 工作区，过期 attempt 的结果不能修改当前任务状态。

工具调用遵循以下边界：任务调度时先根据任务白名单、风险预算、禁止项和
网络策略编译授权集合；模型只能看到当次显式开放的工具 Schema。每次调用
还会依次检查授权、调用次数预算、参数总体大小/深度，以及递归 JSON Schema
约束，然后才进入沙箱 Executor。

工具原始返回值只允许被确定性 Agent 代码在当前进程内部使用。返回模型前
会生成独立安全副本：敏感键和常见凭据模式脱敏，字符串、总字符数、集合
条目数、全局节点数及嵌套深度均受限，二进制和非 JSON 类型不会进入模型，
疑似指令文本会被标为不可信数据。模型系统消息同时明确规定不得执行工具
结果中的指令。SQLite 审计记录使用另一份有界脱敏副本；最终 Task/Job 状态
只有在 `TaskResultEnvelope` 通过主 Agent 独立验证后才会更新。

## 动态工具自增长

所有子 Agent 都有一个只写的可复用代码提名通道。候选不会放进任务结果
正文，也不会立刻暴露给模型；dispatcher 将它单独写入
`output/reusable_code_candidates.json`，主 Agent 只有在该任务的标准结果
通过独立校验后才会摄取。正常后续上下文不再携带候选代码；管理查询默认
也只返回候选 ID、用途、状态和哈希，原始代码留在隔离记录中按需读取。

候选必须提供自包含 Python 入口函数、输入/输出 JSON Schema、行为测试、
功能键、泛化理由、依赖和风险声明。系统拒绝网络、非白名单 import、动态
执行、文件 API、dunder 反射、模块级副作用和疑似凭据；每一次有效证据还
必须在当前任务的 Docker 沙箱中通过声明的行为测试。相似归并同时要求
Schema 兼容，并结合代码哈希、规范化 AST、功能键和用途相似度判断。同一
逻辑 Task 的重试只计一次，累计 3 个独立任务证据后再次运行合并测试集。

生命周期只作用于后续生成的动态工具，所有内置工具永久固定：

```text
PENDING: max_life=10 个相关且已验证的任务事件
  有效相似候选 -> 重置为 10
  life=0 -> 删除代码与测试，保留哈希和证据 tombstone

ACTIVE: max_life=30 个相关且已验证的任务事件
  成功调用（包括输出 Schema 校验成功）-> 重置为 30
  失败调用 -> 不重置，failure_count += 1
  life=0 -> 从运行时注册表移除并删除可执行材料
```

只有静态约束判定为只读的动态工具可以自动激活；其他风险级别会进入
`AWAITING_APPROVAL`，由主 Agent 的 `approve_dynamic_tool()` 显式批准后
重新验证并激活。转正工具仍通过任务白名单、风险预算、参数 Schema、容器
沙箱、输出无损 JSON、output Schema、ContentBlock 和脱敏链路。工具进入
工作空间注册表也不代表每轮全部发给模型，规划器仍按当前任务选择最小集合。

## 测试

```bash
pytest -q
```

架构评审与实现说明见 `docs/ARCHITECTURE_OPTIMIZATION_RECOMMENDATIONS.md`、
`docs/IMPLEMENTATION_NOTES.md` 和 `docs/RESUME_AND_TOOL_AUDIT_DESIGN.md`。
