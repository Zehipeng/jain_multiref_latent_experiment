# AutoDL 使用说明

## 一、把本地代码同步到 GitHub

实验代码目录为：

```text
jain_multiref_latent_experiment/
```

该目录已经配置 `.gitignore`，不会把 `data/`、`outputs/`、模型权重和
Python 缓存提交到 GitHub。首次同步和后续更新由本地 Codex 在你指定的
GitHub 仓库中完成。

## 二、在 AutoDL 拉取代码

```bash
git clone <你的GitHub仓库地址>
cd <仓库目录>/jain_multiref_latent_experiment
```

后续本地代码更新后，在 AutoDL 中执行：

```bash
git pull
```

## 三、安装依赖

优先选择已经安装 PyTorch、CUDA 和 torchvision 的 AutoDL 镜像，不要在
确认前覆盖镜像自带的 PyTorch。

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -m pip install -U pip
python -m pip install -r requirements.txt
```

本项目针对 AutoDL 的 PyTorch 2.1.2 + CUDA 12.1 镜像锁定了核心依赖：
Diffusers 0.30.3、Transformers 4.41.2、Accelerate 0.31.0。不要单独升级
Transformers；较新的 Transformers 版本不兼容 PyTorch 2.1.2 的 pytree API。

检查环境：

```bash
python -m compileall rmlp prepare_references.py run_forgery.py evaluate.py
python -m pytest -q
```

如 Hugging Face 模型需要授权：

```bash
huggingface-cli login
```

## 四、准备 cover 图像

完整 MS-COCO 数据集放在：

```text
data/MS-COCO/
```

当前 AutoDL 数据的图像文件直接位于 `data/MS-COCO/`，没有额外的
`val2017/` 子目录。因此两个跨模型配置均固定使用：

```bash
--cover-dir data/MS-COCO
```

程序会递归扫描该目录，忽略标注 JSON 等非图像文件，并按“目录名 + 文件名”
确定性读取。设置 `--limit 2` 时找到前 2 张图即停止扫描，不会先遍历完整
数据集。图像统一做 512×512 resize 和 center crop。每次运行的 manifest
会记录实际选中图像的相对路径和 SHA-256。论文中的 split 名称应在统计
上传文件数量并核对数据来源后再填写，不能仅由目录名推断。

## 五、生成同密钥参考库

```bash
python prepare_references.py \
  --config configs/tree_ring_stage1.yaml \
  --verify
```

确认终端中 5 张参考图的 p-value。若有参考图的 `p > 0.05`，程序会保留
记录并发出警告，不会自动挑图或重新生成。

## 六、运行最低成本 smoke test

```bash
python run_forgery.py \
  --config configs/tree_ring_stage1.yaml \
  --mode both \
  --limit 2 \
  --iterations 200 \
  --run-name smoke_2x200
```

```bash
python evaluate.py \
  --config configs/tree_ring_stage1.yaml \
  --run-dir outputs/tree_ring_stage1/attacks/smoke_2x200 \
  --no-lpips
```

smoke test 只检查程序能否运行、loss 是否下降、图片和评价文件能否生成，
不用于判断方法效果。

## 六-A、运行跨模型 smoke test

该配置以 SD2-base 作为目标水印生成与检测模型，以 SD1.4 VAE 作为攻击
代理。参考生成器只收录检测 `p <= 0.05` 的候选图，直到得到 5 张共享
`w_seed=0` 的有效参考图。

原 `stabilityai/stable-diffusion-2-base` Hub 仓库已经废弃，因此配置使用
其公开归档镜像 `sd2-community/stable-diffusion-2-base`，并固定 revision
`f5bc1bd97485577aa0b946fa8a9004e2ec147402`。模型身份会写入参考 metadata
和攻击 manifest。

```bash
python prepare_references.py \
  --config configs/tree_ring_cross_model_smoke.yaml \
  --verify \
  --overwrite
```

确认 metadata 中 `accepted_count=5` 后运行：

```bash
python run_forgery.py \
  --config configs/tree_ring_cross_model_smoke.yaml \
  --mode all \
  --limit 2 \
  --iterations 200 \
  --run-name cross_model_smoke_2x200
```

```bash
python evaluate.py \
  --config configs/tree_ring_cross_model_smoke.yaml \
  --run-dir outputs/tree_ring_cross_model_smoke/attacks/cross_model_smoke_2x200 \
  --no-lpips
```

修正后的 smoke 必须同时确认：`prototype_diagnostics.json` 中所有参考
latent 的 `finite=true`，且所有 `distances` 都是有限数值，不能出现
`Infinity` 或 `NaN`。

## 六-B、运行 10×3000 跨模型正式预实验

正式预实验固定使用上传到 `data/MS-COCO/` 的前 10 张图，比较 Jain 单参考、
五参考直接平均和稳健 5→4 聚合三种方法：

```bash
python run_forgery.py \
  --config configs/tree_ring_cross_model_formal.yaml \
  --mode all \
  --limit 10 \
  --iterations 3000 \
  --run-name cross_model_pretest_10x3000
```

```bash
python evaluate.py \
  --config configs/tree_ring_cross_model_formal.yaml \
  --run-dir outputs/tree_ring_cross_model_formal/attacks/cross_model_pretest_10x3000
```

该实验计算 LPIPS，并只为前 5 张 cover 保存 1000、2000、3000 步快照。
运行目录还会保存配置快照、Git commit、目标/代理模型 revision、参考库
checksum、cover 文件 SHA-256 清单和评价结果 checksum。

## 七、运行第一阶段核心实验

```bash
python run_forgery.py \
  --config configs/tree_ring_stage1.yaml \
  --mode both \
  --limit 10 \
  --iterations 3000 \
  --run-name core_10x3000
```

```bash
python evaluate.py \
  --config configs/tree_ring_stage1.yaml \
  --run-dir outputs/tree_ring_stage1/attacks/core_10x3000
```

如果攻击进程中断，使用完全相同的参数并增加：

```bash
--skip-existing
```

例如：

```bash
python run_forgery.py \
  --config configs/tree_ring_stage1.yaml \
  --mode both \
  --limit 10 \
  --iterations 3000 \
  --run-name core_10x3000 \
  --skip-existing
```

## 八、需要下载回本地的结果

核心结果目录：

```text
outputs/tree_ring_stage1/attacks/core_10x3000/
```

至少下载：

```text
manifest.json
prototype_diagnostics.json
metrics.csv
summary.json
logs/
baseline/
full/
covers/
```

其中：

- `summary.json`：baseline/full 的 ASR、eligible ASR、p-value 和质量均值；
- `metrics.csv`：每张 cover 的配对结果；
- `prototype_diagnostics.json`：5 个参考距离、保留编号和剔除编号；
- `logs/`：每张图的 latent loss、pixel loss 和 total loss 曲线；
- `manifest.json`：运行参数、输入输出、耗时和显存。

## 九、现阶段禁止同时修改的参数

第一次核心实验保持配置不变：

```text
w_seed=0
w_channel=0
w_radius=16
N=5
K=4
lambda_pixel=2
alpha=5/255
iterations=3000
```

先完成 baseline/full 公平比较，再根据结果决定是否调整步数、参考数量或
加入两阶段质量约束。

## 十、Simple Average：λ=10⁴ 与每 100 步最早成功停止

本实验直接平均全部 5 张同 key 参考图，不执行稳健参考筛选。3000 步是最大
预算；每 100 步调用一次 SD2-base 目标检测器。首次 `p_value <= 0.05` 时立即
保存当前图像并停止该样本，终端会持续显示 `progress` 与 `detect` 行。

```bash
run_name="simple_average_lambda1e4_smoke_2x3000"
python -u run_forgery.py \
  --config configs/tree_ring_simple_average_lambda1e4_smoke.yaml \
  --mode simple_average \
  --limit 2 \
  --iterations 3000 \
  --lambda-pixel 10000 \
  --detection-every 100 \
  --early-stop-on-success \
  --run-name "$run_name" \
  2>&1 | tee "logs/${run_name}_attack.log"
```

周期 p 值位于 `detections/simple_average/*.csv`；`manifest.json` 记录
`executed_iterations`、`first_success_step` 和 `first_success_p_value`。
