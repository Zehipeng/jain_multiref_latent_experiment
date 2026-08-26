**# 新项目 GitHub 与 AutoDL 同步工作流**

> 本文件用于记录新项目的固定协作方式。Codex 在处理本项目时，应优先读取本文件，并以这里的仓库地址和同步流程为准，不要误用旧项目 `code_fix.git` 的配置。

## 1. 固定项目参数

- GitHub 仓库：`https://github.com/Zehipeng/jain_multiref_latent_experiment`
- Git 远端地址：`https://github.com/Zehipeng/jain_multiref_latent_experiment.git`
- 默认分支：`main`（若实际工作分支不是 `main`，以当前分支为准）
- 工作模式：Codex 在本地修改和检查代码，提交并推送到 GitHub；AutoDL 从 GitHub 拉取完全相同的 commit 后运行 GPU 实验。
- GitHub 只保存源码、配置和必要说明；模型权重、数据集、实验图片、日志、结果文件、压缩包、Token、密码和 SSH 私钥不得提交。

## 2. Codex 必须遵守的流程

每次修改前，Codex 先在项目源码目录执行只读检查：

```bash
git branch --show-current
git log -1 --oneline
git status --short
git remote -v
```

要求：

1. 保留用户已有改动，不覆盖或提交与当前任务无关的文件。
2. 检查 `origin` 是否指向本文件记录的新仓库。
3. 在本地完成代码修改和与改动直接相关的静态检查或单元测试。
4. 只暂存本次修改涉及的文件，创建含义清楚的独立 commit。
5. 推送当前分支到 GitHub。
6. 推送完成后报告：分支名、完整 commit SHA、远端仓库地址，以及可直接复制到 AutoDL 的拉取命令。
7. AutoDL 必须拉取并运行该同一 commit；实验记录中必须保存 commit、配置、模型路径、随机种子和实际运行命令。
8. 用户要求修改本项目代码、配置、测试或运行脚本时，Codex 在完成相关检查后，默认自动将**本次任务涉及的文件**创建独立 commit 并推送到当前 GitHub 分支，然后生成带 AutoDL 网络加速、HTTP/1.1、`fetch && merge --ff-only` 和完整 commit 核验的可复制命令。除非用户明确要求暂不提交或暂不推送，否则不应停留在“仅本地修改”状态。

上述自动同步规则不扩大暂存范围：仍然只能显式暂存当前任务文件，不得使用 `git add .`，不得包含用户已有的无关修改、数据集、模型、日志或实验结果。若检查失败、GitHub 认证失败、远端发生分歧或需要破坏性处理，Codex 应停止自动同步并报告具体阻塞，不得强推或覆盖历史。

### 2.1 修改实验代码后的强制回复结构

每次 Codex 修改并推送实验代码后，在同一回复中必须按照实际提交、脚本参数、配置文件和输出目录，提供以下六组**可直接复制**的 AutoDL 指令，不得只给出其中一部分：

1. **代码拉取指令**：加载 `/etc/network_turbo`、设置 HTTP/1.1、执行 `fetch && merge --ff-only`，并核对完整 commit SHA。
2. **单元测试指令**：运行与本次修改直接相关的测试；必要时同时运行静态检查或完整测试集。
3. **实验运行指令**：使用唯一 `run_id`，明确配置、样本数量、迭代次数、随机种子相关配置、日志路径及恢复方式。
4. **检查结果指令**：检查退出码、日志末尾、manifest、预期文件数量、关键图片／CSV／JSON和异常信息。
5. **评价实验指令**：执行与当前实验匹配的评价脚本，明确是否启用 LPIPS、错误密钥检测等可选项，并给出汇总结果查看命令。
6. **打包实验结果指令**：记录 commit、分支、工作区状态、环境、实际命令和退出码；仅打包本次实验的明确输出、日志、配置和元数据；生成 `.tar.gz`、SHA-256与内容清单并检查文件大小。

指令中的分支、完整 SHA、`run_id`、配置文件、输出目录和日志文件必须针对当次实验填写，不得沿用无关示例占位符。若某一步必须等待前一步成功，Codex 应明确要求用户暂停并回传结果，或使用安全的条件连接避免失败后继续执行。

## 3. 本地电脑：首次连接 GitHub（只做一次）

### 情况 A：本地目录还不是 Git 仓库

在新项目源码目录执行：

```bash
git init
git branch -M main
git remote add origin https://github.com/Zehipeng/jain_multiref_latent_experiment.git
```

完成首次提交后推送：

```bash
git push -u origin main
```

### 情况 B：本地目录已经是 Git 仓库

先检查远端：

```bash
git remote -v
```

如果已有 `origin` 但地址错误：

```bash
git remote set-url origin https://github.com/Zehipeng/jain_multiref_latent_experiment.git
```

如果没有 `origin`：

```bash
git remote add origin https://github.com/Zehipeng/jain_multiref_latent_experiment.git
```

然后推送当前分支并建立跟踪关系：

```bash
git push -u origin HEAD
```

Windows 上使用 HTTPS 推送时，首次认证应通过 Git Credential Manager／浏览器完成。凭据会由系统凭据管理器保存，之后正常 `git push` 不应要求重新连接。不得把 Token 写入远端 URL、脚本、Markdown、代码或 Git 历史。若认证失败，Codex应停止推送并提示用户完成 GitHub 授权，不得索要或记录 Token。

## 4. 本地电脑：日常修改与推送

远端和上游分支建立后，Codex 每次完成代码、配置、测试或运行脚本修改并通过相应检查后，应自动执行以下流程；用户无需再次单独提醒“同步到 GitHub”：

```bash
git status --short
git add <本次修改的文件>
git commit -m "<清楚描述本次修改>"
git push
```

推送前还应运行与本次改动直接相关的检查。不要使用 `git add .` 把无关文件一并提交；不要使用强制推送或覆盖远端历史，除非用户明确授权。推送完成后，Codex 必须立即报告完整 commit SHA，并按第6节生成 AutoDL 拉取和核验指令。

## 5. AutoDL：首次获取项目（每台实例只做一次）

### 5.1 访问 GitHub 前必须先启用 AutoDL 网络加速

AutoDL 容器直连 GitHub 时可能出现以下错误：

```text
GnuTLS recv error (-110): The TLS connection was non-properly terminated
SSL connection timeout
```

因此，在 AutoDL 中执行 `git clone`、`git fetch`、`git pull` 或其他需要访问 GitHub 的命令前，必须先在**当前终端会话**中检测并加载 AutoDL 网络加速脚本：

```bash
if [ -f /etc/network_turbo ]; then
    source /etc/network_turbo
    echo "AutoDL network turbo enabled"
else
    echo "/etc/network_turbo does not exist"
fi
```

`source /etc/network_turbo` 设置的网络环境通常只对当前终端会话有效。新建终端、重新登录容器或重启实例后，在第一次访问 GitHub 前应重新执行。不得通过关闭 SSL 验证规避网络问题，例如不得执行 `git config http.sslVerify false`。

随后建议在项目仓库中固定使用 HTTP/1.1，以降低部分 AutoDL 网络环境中 HTTP/2/TLS 连接异常的概率：

```bash
git config --local http.version HTTP/1.1
```

如果尚未进入 Git 仓库（例如准备首次 `clone`），可在单次命令中指定 HTTP/1.1：

```bash
git -c http.version=HTTP/1.1 clone https://github.com/Zehipeng/jain_multiref_latent_experiment.git
```

此仓库为公开仓库时，推荐用 HTTPS clone，拉取源码不需要 GitHub 登录或 SSH 密钥：

```bash
cd /root/autodl-tmp/project
if [ -f /etc/network_turbo ]; then
    source /etc/network_turbo
fi
git -c http.version=HTTP/1.1 clone https://github.com/Zehipeng/jain_multiref_latent_experiment.git
cd /root/autodl-tmp/project/jain_multiref_latent_experiment
git config --local http.version HTTP/1.1
git remote -v
git log -1 --oneline
git status --short
```

如果仓库以后改为私有仓库，HTTPS 匿名拉取将失效。届时应单独配置 AutoDL 的 GitHub SSH deploy key 或授权凭据，但不得把私钥或 Token 放进仓库。本文件不保存任何凭据。

## 6. AutoDL：以后每次同步代码

本地 Codex 推送完成后，在 AutoDL 执行。这里将远端抓取和本地快进合并用 `&&` 连接；只有 `fetch` 成功时才执行 `merge`，避免抓取失败后仍使用陈旧的本地 `origin/main` 并误判为“Already up to date”：

```bash
cd /root/autodl-tmp/project/jain_multiref_latent_experiment

if [ -f /etc/network_turbo ]; then
    source /etc/network_turbo
    echo "AutoDL network turbo enabled"
else
    echo "/etc/network_turbo does not exist"
fi

git config --local http.version HTTP/1.1
git switch <本地推送的分支名>

git -c http.version=HTTP/1.1 fetch --no-tags origin <本地推送的分支名> \
&& git merge --ff-only origin/<本地推送的分支名> \
&& git rev-parse HEAD \
&& git status --short
```

对于本项目当前的 `main` 分支，可直接复制：

```bash
cd /root/autodl-tmp/project/jain_multiref_latent_experiment

if [ -f /etc/network_turbo ]; then
    source /etc/network_turbo
    echo "AutoDL network turbo enabled"
else
    echo "/etc/network_turbo does not exist"
fi

git config --local http.version HTTP/1.1
git switch main

git -c http.version=HTTP/1.1 fetch --no-tags origin main \
&& git merge --ff-only origin/main \
&& git rev-parse HEAD \
&& git status --short
```

如果 `fetch` 报错，后续 `merge` 不会执行。此时不得单独运行 `git merge origin/main`，因为本地的 `origin/main` 可能仍是旧缓存；应先解决网络连接并确保 `fetch` 成功。

若远端有一个尚未在 AutoDL 建立的工作分支，首次拉取该分支使用：

```bash
cd /root/autodl-tmp/project/jain_multiref_latent_experiment
if [ -f /etc/network_turbo ]; then
    source /etc/network_turbo
fi
git -c http.version=HTTP/1.1 fetch --no-tags origin <分支名> \
&& git switch -c <分支名> --track origin/<分支名> \
&& git rev-parse HEAD \
&& git status --short
```

核对规则：

- `git log -1` 显示的 commit 必须与 Codex 推送后报告的 commit 一致。
- `git status --short` 不应显示已跟踪源码的本地修改。
- AutoDL 有未提交的源码改动时，不直接 pull，也不擅自删除；先保存或确认这些改动的用途。
- 只运行已提交且已推送的版本，不直接在 AutoDL 临时修改源码后作为正式实验版本。

## 7. 实验运行与结果回传

固定顺序：

```text
确认实验设置
→ Codex 本地修改与验证
→ 创建独立 commit
→ 推送 GitHub
→ AutoDL 拉取并核对同一 commit
→ 运行单元测试
→ 运行实验
→ 检查输出完整性
→ 运行评价实验
→ 记录复现元数据
→ 打包本次输出、日志、配置和结果
→ 生成并校验 SHA-256 与内容清单
→ 下载到本地
→ Codex 校验、解压、分析并给出结论
```

任何 smoke、小规模核心实验、消融实验或正式实验都必须走完上述闭环。不得因为实验规模较小而省略结果检查、评价或打包指令。若实验失败，也应保存和打包当次日志、配置、manifest／部分输出及退出码，用于本地诊断，但必须标记为失败或不完整实验，不能当作正式结果。

正式实验优先在 `tmux` 中运行。实验至少记录：

- 完整 commit SHA
- 分支名和实际命令
- 配置文件及模型路径
- 数据或 manifest 版本
- 所有随机种子
- 环境信息、完整日志和退出码
- 核心 CSV／JSON、指标、生成图像和检查点

实验结果不要推送到 GitHub。应在 AutoDL 打包为独立的 `tar.gz`，再通过 AutoDL 文件管理器下载到本地指定的实验结果目录，由 Codex 读取和分析。

Codex 在提供打包命令时必须使用该次实验的准确路径，默认采用以下三个同名交付文件：

```text
<run_id>.tar.gz
<run_id>.tar.gz.sha256
<run_id>.contents.txt
```

用户将这三个文件下载到本地后，只需向 Codex提供压缩包的绝对路径。Codex 随后负责校验 SHA-256、解压、核对复现信息和文件完整性，区分已验证结果与后续假设，并分析攻击成功率、质量指标、成本、失败样本和不同方法间的差异，形成当前实验结论与下一步建议。

### 7.1 打包前记录可复现信息

只有实验进程正常结束、退出码和输出文件已检查后才打包。以下示例中的 `run_id` 每次实验都要改成唯一名称；项目实际位于 `/root/autodl-tmp/project/jain_multiref_latent_experiment`：

```bash
cd /root/autodl-tmp/project/jain_multiref_latent_experiment

run_id="tree_ring_stage1_20260824_run01"
project_dir="/root/autodl-tmp/project/jain_multiref_latent_experiment"
export_root="/root/autodl-tmp/experiment_exports"
metadata_dir="$project_dir/logs/run_metadata_$run_id"
archive="$export_root/${run_id}.tar.gz"

mkdir -p "$export_root" "$metadata_dir"

git rev-parse HEAD > "$metadata_dir/commit_sha.txt"
git branch --show-current > "$metadata_dir/branch.txt"
git status --short > "$metadata_dir/git_status.txt"
python --version > "$metadata_dir/environment.txt" 2>&1
python -m pip freeze >> "$metadata_dir/environment.txt"
nvidia-smi >> "$metadata_dir/environment.txt" 2>&1
```

还应把正式实验的原始命令、配置文件、模型标识、数据或 manifest 版本、随机种子、开始与结束时间、退出码写入日志或元数据文件。若这些信息尚未记录，先补齐，不要仅凭文件名猜测。

### 7.2 在 AutoDL 打包并校验

下面的命令打包本阶段输出、日志、配置和依赖说明，不包含 Hugging Face 缓存、模型权重或数据集。执行前确认 `run_id` 与上一节完全一致：

```bash
cd /root/autodl-tmp/project/jain_multiref_latent_experiment

run_id="tree_ring_stage1_20260824_run01"
project_dir="/root/autodl-tmp/project/jain_multiref_latent_experiment"
export_root="/root/autodl-tmp/experiment_exports"
archive="$export_root/${run_id}.tar.gz"

mkdir -p "$export_root"

tar -czf "$archive" \
  -C "$project_dir" \
  outputs/tree_ring_stage1 \
  logs \
  configs/tree_ring_stage1.yaml \
  requirements.txt

cd "$export_root"
sha256sum "${run_id}.tar.gz" > "${run_id}.tar.gz.sha256"
tar -tzf "${run_id}.tar.gz" > "${run_id}.contents.txt"
ls -lh "${run_id}.tar.gz" "${run_id}.tar.gz.sha256" "${run_id}.contents.txt"
```

若某次实验使用不同的输出目录或配置文件，必须相应修改 `tar` 的显式路径。不要直接打包整个项目、模型缓存或数据集。打包后先确认压缩包大小合理、`tar -tzf` 成功且 SHA-256 文件已生成。

### 7.3 下载到本地并交给 Codex 分析

通过 AutoDL 文件管理器或 FileZilla 下载以下三个文件：

- `/root/autodl-tmp/experiment_exports/<run_id>.tar.gz`
- `/root/autodl-tmp/experiment_exports/<run_id>.tar.gz.sha256`
- `/root/autodl-tmp/experiment_exports/<run_id>.contents.txt`

本地统一保存到：

```text
C:\Users\dell\Desktop\codex学习文档\实验结果\<run_id>\
```

不要修改压缩包内部文件。下载完成后，把本地压缩包的绝对路径告诉 Codex，并明确要求：

```text
请先校验 SHA-256，再解压到同名目录；核对 commit、配置、环境、日志、退出码和文件清单，
然后分析 baseline 与 full 的核心指标、逐样本结果、攻击成功率、图像质量、失败样本、
指标间权衡和异常；输出表格、必要图形、结论边界，以及下一轮最小实验建议。
```

Codex 分析时遵循以下顺序：

1. 校验压缩包 SHA-256；不一致时停止分析并重新下载。
2. 核对完整 commit SHA、分支、配置、随机种子、环境、命令和退出码。
3. 检查预期 CSV／JSON、图像、日志和文件数量是否齐全，区分完整结果与部分结果。
4. 先报告原始指标和样本量，再比较 baseline 与 full；同时报告绝对差、相对差和质量—攻击权衡。
5. 检查失败样本、离群值、缺失值和统计不确定性；小样本筛选结果不得表述成稳定结论。
6. 生成可直接用于后续实验讨论的 Markdown 分析报告、汇总表和必要图形，并列出缺失信息。

只有 SHA-256 校验通过、Codex 能正常解压读取、关键结果文件完整后，才允许清理 AutoDL 压缩包。

### 7.4 本地确认后清理 AutoDL 压缩文件

远端清理属于不可恢复操作。先由用户确认本地压缩包、校验文件和 Codex 分析目录均可正常读取，然后只删除该次导出的压缩包及其校验／清单文件；默认保留 AutoDL 中的原始 `outputs/`、`logs/` 和代码。

```bash
run_id="tree_ring_stage1_20260824_run01"
export_root="/root/autodl-tmp/experiment_exports"
archive="$export_root/${run_id}.tar.gz"
checksum="$export_root/${run_id}.tar.gz.sha256"
contents="$export_root/${run_id}.contents.txt"

resolved_archive="$(realpath -m "$archive")"
expected_archive="/root/autodl-tmp/experiment_exports/${run_id}.tar.gz"

echo "$resolved_archive"

if [ "$resolved_archive" = "$expected_archive" ] \
  && [ -f "$archive" ] \
  && [ -f "$checksum" ] \
  && [ -f "$contents" ]; then
  rm -- "$archive" "$checksum" "$contents"
  echo "已清理本次导出文件；原始实验结果仍保留在 AutoDL。"
else
  echo "路径校验失败或导出文件不完整，未执行删除。"
fi
```

不得使用通配符删除 `experiment_exports`，不得删除整个项目目录，也不得自动清理原始实验结果。如需删除原始 `outputs/` 或 `logs/`，必须另行确认准确目录和备份状态。

## 8. 常见问题处理

### `origin already exists`

不要重复添加，改为：

```bash
git remote set-url origin https://github.com/Zehipeng/jain_multiref_latent_experiment.git
```

### AutoDL 的 `git pull --ff-only` 失败

如果错误包含 `GnuTLS recv error (-110)` 或 `SSL connection timeout`，先重新加载当前终端会话的 AutoDL 网络加速，再使用 HTTP/1.1 重新抓取：

```bash
if [ -f /etc/network_turbo ]; then
    source /etc/network_turbo
fi
git config --local http.version HTTP/1.1
git -c http.version=HTTP/1.1 fetch --no-tags origin main
```

只有上述 `fetch` 成功后，才执行：

```bash
git merge --ff-only origin/main
git rev-parse HEAD
git status --short
```

若仍然失败，执行以下只读检查并保留输出：

```bash
curl --http1.1 --connect-timeout 20 -I https://github.com
env | grep -i proxy
git remote -v
git status --short
git branch -vv
git log --oneline --decorate -5
git config --show-origin --get-regexp 'http\..*|https\..*|.*proxy.*' || true
```

不要直接 `git reset --hard`，不要删除本地文件，不要关闭 SSL 验证，也不要在失败的 `fetch` 之后依据“Already up to date”判断同步成功；应把输出交给 Codex 判断。

### 本地推送要求重新认证

确认 Windows Git Credential Manager 可用，并通过浏览器重新授权。Codex 不保存、不展示、不提交密码或 Token。

### GitHub 上已有文件导致首次推送被拒绝

先检查远端历史，再决定是否 `git pull --rebase` 或采用其他合并方式。未经确认不得强推覆盖 GitHub 历史。

## 9. 给 Codex 的长期提示

处理 `jain_multiref_latent_experiment` 项目时：

- 先读取本文件，再执行 Git 操作。
- 不要把旧项目仓库 `https://github.com/Zehipeng/code_fix.git` 设置为本项目的 `origin`。
- 日常同步不重新执行 `git init`、`git remote add` 或 `git clone`；这些只在首次配置时使用。
- 本地已配置好远端后使用 `git push`；AutoDL 已 clone 后按本文第6节使用 `fetch && merge --ff-only` 同步。
- AutoDL 每个新终端会话第一次访问 GitHub 前，先检测并执行 `source /etc/network_turbo`，再使用 HTTP/1.1 执行 Git 网络命令。
- AutoDL 日常同步优先使用“`fetch && merge --ff-only`”串联流程；`fetch` 失败时不得继续合并陈旧的远端跟踪分支。
- 用户要求修改本项目代码、配置、测试或运行脚本时，完成检查后默认自动创建独立 commit 并推送；除非用户明确要求不提交或不推送，无需再次询问是否同步。
- 自动推送后必须在同一回复中提供包含网络加速、HTTP/1.1、快进同步和完整 SHA 核验的 AutoDL 命令。
- 每次修改实验代码后的回复必须完整提供：代码拉取、单元测试、实验运行、结果检查、实验评价和结果打包六组指令。
- 每次实验（包括 smoke）都必须生成独立 `run_id`，记录复现元数据，并打包为同名 `.tar.gz`、`.sha256` 和内容清单供用户下载。
- 用户下载结果后，Codex 必须先校验和核对完整性，再分析数据并给出有证据边界的实验结论。
- 每次推送后都给出 AutoDL 可复制命令，并要求核对完整 commit SHA。
- 认证或远端异常、需要覆盖历史、发现未说明的本地改动时停止并报告，不自行采取破坏性操作。
