# 新项目 GitHub 与 AutoDL 同步工作流

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

远端和上游分支建立后，每次只需：

```bash
git status --short
git add <本次修改的文件>
git commit -m "<清楚描述本次修改>"
git push
```

推送前还应运行与本次改动直接相关的检查。不要使用 `git add .` 把无关文件一并提交；不要使用强制推送或覆盖远端历史，除非用户明确授权。

## 5. AutoDL：首次获取项目（每台实例只做一次）

此仓库为公开仓库时，推荐用 HTTPS clone，拉取源码不需要 GitHub 登录或 SSH 密钥：

```bash
cd /root/autodl-tmp
git clone https://github.com/Zehipeng/jain_multiref_latent_experiment.git
cd /root/autodl-tmp/jain_multiref_latent_experiment
git remote -v
git log -1 --oneline
git status --short
```

如果仓库以后改为私有仓库，HTTPS 匿名拉取将失效。届时应单独配置 AutoDL 的 GitHub SSH deploy key 或授权凭据，但不得把私钥或 Token 放进仓库。本文件不保存任何凭据。

## 6. AutoDL：以后每次同步代码

本地 Codex 推送完成后，在 AutoDL 执行：

```bash
cd /root/autodl-tmp/jain_multiref_latent_experiment
git switch <本地推送的分支名>
git pull --ff-only
git log -1 --format=fuller
git status --short
```

若远端有一个尚未在 AutoDL 建立的工作分支，首次拉取该分支使用：

```bash
cd /root/autodl-tmp/jain_multiref_latent_experiment
git fetch origin
git switch -c <分支名> --track origin/<分支名>
git log -1 --format=fuller
git status --short
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
→ 运行必要检查和一次小规模 smoke
→ 运行正式实验
→ 打包日志、配置和结果
→ 下载回本地分析
```

正式实验优先在 `tmux` 中运行。实验至少记录：

- 完整 commit SHA
- 分支名和实际命令
- 配置文件及模型路径
- 数据或 manifest 版本
- 所有随机种子
- 环境信息、完整日志和退出码
- 核心 CSV／JSON、指标、生成图像和检查点

实验结果不要推送到 GitHub。应在 AutoDL 打包为独立的 `tar.gz`，再通过 AutoDL 文件管理器下载到本地指定的实验结果目录，由 Codex读取和分析。

## 8. 常见问题处理

### `origin already exists`

不要重复添加，改为：

```bash
git remote set-url origin https://github.com/Zehipeng/jain_multiref_latent_experiment.git
```

### AutoDL 的 `git pull --ff-only` 失败

先执行以下只读检查并保留输出：

```bash
git status --short
git branch -vv
git log --oneline --decorate -5
```

不要直接 `git reset --hard`，也不要删除本地文件；应把输出交给 Codex 判断。

### 本地推送要求重新认证

确认 Windows Git Credential Manager 可用，并通过浏览器重新授权。Codex 不保存、不展示、不提交密码或 Token。

### GitHub 上已有文件导致首次推送被拒绝

先检查远端历史，再决定是否 `git pull --rebase` 或采用其他合并方式。未经确认不得强推覆盖 GitHub 历史。

## 9. 给 Codex 的长期提示

处理 `jain_multiref_latent_experiment` 项目时：

- 先读取本文件，再执行 Git 操作。
- 不要把旧项目仓库 `https://github.com/Zehipeng/code_fix.git` 设置为本项目的 `origin`。
- 日常同步不重新执行 `git init`、`git remote add` 或 `git clone`；这些只在首次配置时使用。
- 本地已配置好远端后使用 `git push`；AutoDL 已 clone 后使用 `git pull --ff-only`。
- 每次推送后都给出 AutoDL 可复制命令，并要求核对完整 commit SHA。
- 认证或远端异常、需要覆盖历史、发现未说明的本地改动时停止并报告，不自行采取破坏性操作。
