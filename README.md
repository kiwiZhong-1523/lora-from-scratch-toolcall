# lora-toolcall

用纯 PyTorch 手写 LoRA（不依赖 `peft`），在 Qwen2.5-0.5B-Instruct 上做函数调用（function calling）微调，并围绕 rank、目标模块、学习率、随机种子和全参数微调做了一组对照实验。

A hand-written LoRA in plain PyTorch (no `peft`), used to fine-tune Qwen2.5-0.5B-Instruct for function calling, with controlled ablations over rank, target modules, learning rate, random seed, and a full fine-tuning reference.

## 目的 / Purpose

1. **理解 LoRA 的实现细节**：自己写出注入、前向、merge / unmerge、保存和加载，而不是把 `peft` 当黑盒用；再用 `peft` 在完全相同的初始化下交叉验证。
2. **搞清楚各个超参数实际影响多大**：在同一份数据、同一套评测下，逐个改变一个因素，看效果和成本如何变化。
3. **建立可信的实验流程**：固定数据划分、记录每个 run 的配置和资源消耗，并用多个 seed 估计噪声，避免把随机波动当成结论。
4. **Understand LoRA at the implementation level**: write injection, forward, merge / unmerge, save and load myself instead of treating `peft` as a black box, then cross-check against `peft` from an identical initialization.
5. **Measure how much each hyperparameter actually matters**: on one dataset and one evaluation, vary one factor at a time and observe both quality and cost.
6. **Build a trustworthy experiment workflow**: fixed data splits, per-run config and resource logs, and multiple seeds to estimate noise so random variation is not mistaken for a finding.

## 任务与数据 / Task and data

- **任务**：给定可用工具列表和用户请求，模型只输出一个 JSON 对象 `{"name": ..., "arguments": {...}}`。
- **数据**：[xLAM-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k)。只保留单次调用样本，去掉超过 1,536 token 的样本，剩 28,412 条；固定划分为训练 20,000 / 验证 500 / 测试 2,000，样本 id 记录在 [data/splits/](data/splits/)。
- **主要指标**：`full_exact`，即工具名和全部参数都完全正确。另外记录工具名正确率、参数正确率、JSON 格式合法性等诊断指标。
- **训练设置**：1 epoch，有效 batch 32，只在答案 token 上计算 loss，alpha = 2r，默认学习率 2e-4；单卡 A10（24 GB）。
- **Task**: given a list of available tools and a user request, the model must output only one JSON object `{"name": ..., "arguments": {...}}`.
- **Data**: [xLAM-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k). Single-call examples only, those longer than 1,536 tokens dropped, leaving 28,412; fixed split of 20,000 train / 500 val / 2,000 test, with example ids stored in [data/splits/](data/splits/).
- **Primary metric**: `full_exact` — tool name and every argument exactly correct. Tool-name accuracy, argument accuracy and JSON validity are tracked as diagnostics.
- **Training setup**: 1 epoch, effective batch 32, loss on answer tokens only, alpha = 2r, default learning rate 2e-4; a single A10 (24 GB).

## 实验内容与结论 / Experiments and findings

完整的数据表和图见 [results/experiment-results-bilingual.md](results/experiment-results-bilingual.md)。

Full tables and figures are in [results/experiment-results-bilingual.md](results/experiment-results-bilingual.md).

**1. 零样本基线 / Zero-shot baselines** — 不微调时，1.5B 模型的 `full_exact` 比 0.5B 高约 9 个百分点。两者工具名大多能选对，主要差距在参数上。

Without fine-tuning, the 1.5B model scores about 9 points higher than 0.5B in `full_exact`. Both mostly pick the right tool; the gap is mainly in the arguments.

**2. LoRA rank** — 在 attention 上把 rank 从 1 增到 64，准确率持续上升、验证 loss 持续下降，在测试范围内没有看到饱和；adapter 大小则随 rank 线性增长（r=8 约 4 MB，r=64 约 33 MB）。

Increasing the attention-LoRA rank from 1 to 64 steadily raises accuracy and lowers validation loss, with no saturation in the tested range; adapter size grows linearly with rank (about 4 MB at r=8, 33 MB at r=64).

**3. 随机种子 / Seed sensitivity** — 同一配置换三个 seed，`full_exact` 的样本标准差约 0.5 个百分点。因此 1 个百分点以内的差异不足以说明哪个配置更好。

Across three seeds of the same configuration, the sample standard deviation of `full_exact` is about 0.5 points, so differences under about one point are not evidence that one configuration is better.

**4. 目标模块 / Target modules** — 在相同 r=8 下，把 LoRA 放进 MLP 比只放 attention 高约 2.8 个百分点；attention + MLP 只比单独 MLP 再高 0.4 个百分点，落在种子噪声范围内，但参数量和训练时间都更多。

At the same r=8, putting LoRA in the MLP beats attention-only by about 2.8 points; attention + MLP adds only 0.4 points over MLP alone, within seed noise, at higher parameter count and training time.

**5. 全参数微调参照 / Full fine-tuning reference** — 全参数微调比 attention r=8 LoRA 高约 4.4 个百分点，但训练的参数多约 450 倍。它的峰值显存反而更低，原因是开了梯度检查点、micro-batch 也更小，所以这是一个参考对比，不是严格的单变量对照。

Full fine-tuning is about 4.4 points above attention r=8 LoRA while training roughly 450× more parameters. Its peak memory was actually lower because it used gradient checkpointing and a smaller micro-batch, so this is a reference comparison rather than a strict single-variable control.

**6. 学习率 / Learning rate** — 默认的 2e-4 偏低：提到 5e-4 或 1e-3 后，`full_exact` 提升约 2.5 个百分点。1e-3 的测试分数略高，但验证 loss 已开始回升。

The default 2e-4 is too low: raising it to 5e-4 or 1e-3 improves `full_exact` by about 2.5 points. 1e-3 has a marginally higher test score, but validation loss is already starting to rise.

**7. 手写 LoRA 与 PEFT / Hand-written LoRA vs PEFT** — 两者从逐位相同的初始 adapter 出发，`full_exact` 只差 0.1 个百分点，验证 loss 和 adapter 大小几乎一致，说明手写实现是正确的。

Starting from bit-identical adapters, the two differ by only 0.1 points in `full_exact`, with nearly identical validation loss and adapter size, which confirms the hand-written implementation is correct.

## 我学到了什么 / What I learned

**LoRA 本身 / LoRA itself**

- LoRA 的初始化有讲究：B 初始化为 0，保证训练开始时模型输出不变；A 必须随机初始化，否则 B 的梯度为 0，训练根本无法启动。缩放系数 alpha / r 决定了更新量的大小，固定 alpha = 2r 能让不同 rank 之间的有效学习率大致可比。
- merge 就是把 `(alpha / r) · B @ A` 加回原权重，推理时没有额外开销；unmerge 能把它再减回去，所以一个基座可以切换多个 adapter。
- Initialization matters: B starts at zero so the model's output is unchanged at step 0, while A must be random — otherwise B's gradient is zero and training never starts. The scale alpha / r sets the size of the update; fixing alpha = 2r keeps the effective learning rate roughly comparable across ranks.
- Merging just adds `(alpha / r) · B @ A` back into the base weight, so inference costs nothing extra; unmerging subtracts it again, which lets one base model switch between adapters.

**超参数 / Hyperparameters**

- 调学习率的收益可能比加容量还大：r=8 只把学习率从 2e-4 提到 5e-4，效果就接近 r=32。先把学习率调好，再比较 rank 和模块，结论才可靠。
- LoRA 放在哪里和 rank 设多大同样重要。在这个任务上，MLP 比 attention 更有效。
- 微调后的 0.5B 模型（最好的几组约 79%–81%）已经追平了未微调的 1.5B（80.8%）。对窄任务来说，小模型加 LoRA 是很划算的选择。
- Tuning the learning rate can pay off more than adding capacity: r=8 with the learning rate raised from 2e-4 to 5e-4 comes close to r=32. Get the learning rate right before comparing ranks or modules, or the conclusions are unreliable.
- Where LoRA goes matters as much as its rank. On this task, MLP is more effective than attention.
- The fine-tuned 0.5B model (about 79–81% in the best runs) catches up with the un-tuned 1.5B model (80.8%). For a narrow task, a small model plus LoRA is a very cost-effective choice.

## 项目结构 / Layout


| 路径 / Path                                                                                    | 说明 / Description                                                                                           |
| ---------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| [src/lora.py](src/lora.py)                                                                     | 手写 LoRA：注入 / merge / unmerge / 保存 / 加载 · hand-written LoRA: inject / merge / unmerge / save / load |
| [src/toolcall.py](src/toolcall.py)                                                             | prompt 格式、输出解析、指标 · prompt format, output parsing, metrics                                        |
| [src/data.py](src/data.py), [src/evaluation.py](src/evaluation.py)                             | 数据编码与共用评测 · data encoding and shared evaluation                                                    |
| [scripts/download.py](scripts/download.py), [scripts/convert_xlam.py](scripts/convert_xlam.py) | 下载模型与数据，生成固定划分 · download model and data, build fixed splits                                  |
| [scripts/train.py](scripts/train.py)                                                           | 训练（手写 LoRA / peft / 全参数）并评测 · train (hand LoRA / peft / full) and evaluate                      |
| [scripts/eval.py](scripts/eval.py)                                                             | 评测基座或 LoRA 模型 · evaluate a base or LoRA model                                                        |
| [scripts/run_all.sh](scripts/run_all.sh)                                                       | 按顺序运行全部实验，可断点续跑 · run all experiments in order, resumable                                    |
| [scripts/collect.py](scripts/collect.py)                                                       | 汇总结果到`results/summary.csv` · collect results into `results/summary.csv`                                |
| [tests/](tests/)                                                                               | 单元测试 · unit tests (`python -m pytest -q`)                                                               |

## 复现 / Reproduce

```bash
pip install -r requirements.txt
python scripts/download.py        # 模型与数据 / models and dataset
python scripts/convert_xlam.py    # 按 data/splits 重建划分 / rebuild splits from data/splits
nohup bash scripts/run_all.sh > run_all.log 2>&1 &
```
