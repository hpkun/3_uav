# MADDPG V1：pre-tanh regularization 受控实验

## 假设与边界

本轮仅验证：将 Actor regularizer 从 bounded action 移到 pre-tanh output，
是否能抑制 logit explosion 和 tanh saturation。这里不预先声称 V1 有效，
不以 smoke 胜率或 500k 胜率评价最终算法性能。

背景来自用户提供的既有 V0 1.2M / 1.8M / 2.0M 审计结论（本轮未重新复现该长训审计）：

- acceleration saturation 约 100%，turn saturation 约 99.7%–100%；
- 实际速度约 96%–97% 时间贴在 50 m/s 下限；
- pre-tanh z 达数百至数千；
- ||dQ/da|| 约 0.08–0.12，但 ||dQ/dz|| 约 1e-5–1e-4；
- tanh gradient retention 约 0.03%–0.09%。

这说明 bounded regularizer 本身也可能失去将 z 拉回的梯度。

## 唯一算法变量

始终有 `z = actor.pre_tanh(obs)`，`action = tanh(z)`，Critic 和环境始终看到
bounded action。当前 agent 替换自身 Replay action，其余四个保持 Replay action 并 detach。

| 版本 | profile | Actor loss |
| --- | --- | --- |
| V0（默认） | configs/maddpg.yaml | -mean(Q) + 0.001 mean(tanh(z)^2) |
| V1 | configs/maddpg_pretanh_reg.yaml | -mean(Q) + 0.001 mean(z^2) |

`actor_regularization_mode` 只允许 `bounded_action` / `pre_tanh`，默认前者。
两个 YAML 的所有数值完全相同，唯一值差异是该 mode。
没有 clip/detach z，没有额外动作缩放。

[OpenAI 原始 MADDPG trainer](https://github.com/openai/maddpg/blob/master/maddpg/trainer/maddpg.py)
的 `p_reg` 作用于 policy distribution 的 `flatparam()`，并乘 1e-3 加入 policy loss。
V1 应被描述为 **closer to the regularization intent of the original implementation under the bounded tanh adaptation**，
不是 OpenAI 官方实现的逐行等价版本；本项目的 bounded tanh adaptation 仍然存在。

全部冻结：环境、运动学及速度/加速度/转率上下限、Radar/Weapon/ATA/aspect、OOB、collision、
Reward、终止、observation/global state、Greedy Blue，以及 MAPPO 网络/训练/评估。
MADDPG 的网络层、hidden64、Xavier 初始化、参数名/量、独立五 Actor/五 Critic、Adam、
lr .01/.01、gamma .95、tau .01、lambda .001、逐参数 gradient clipping .5、
OU(0,.15,.20)、Replay/采样/掩码、更新顺序、target update 和调度均不变。
不增加 TD3/MATD3/PER/RNN/attention/parameter sharing/curriculum/BC 或 reward shaping。

## Checkpoint 兼容

load 仅在 checkpoint 的 `maddpg_config` 缺少 mode 时，在内存补 `bounded_action`，
然后严格比较整个算法配置。其余差异仍拒绝，不会将旧 V0 当 V1；也不允许 V1 误加载为 V0。
返回 payload 包含规范化 mode，但不改写磁盘文件。评估入口通过 dataclass 默认值支持旧格式。
新保存 checkpoint 显式包含 mode。网络 state_dict 名称/形状不变。

## 只读 Replay-batch diagnostics

每个 Actor **optimizer.step 前**，仅使用本次 joint Replay batch 中该 agent 的 active rows：

| 每 agent 字段 | 定义 | 聚合字段 |
| --- | --- | --- |
| actor_reg | 实际已乘 lambda 的 reg | mean_actor_reg |
| pre_tanh_abs_mean | mean(abs(z)) | mean_pre_tanh_abs |
| pre_tanh_abs_max | max(abs(z)) | max_pre_tanh_abs |
| tanh_derivative_mean | mean(1-action²) | mean_tanh_derivative |
| accel_saturation | mean(abs(action[:,0]) >= .95) | mean_accel_saturation |
| turn_saturation | mean(abs(action[:,1]) >= .95) | mean_turn_saturation |

mean 聚合是实际更新 agent 的等权平均（不是按 active row 数加权）；max 为跨 agent 最大值。
这些数值不参与 loss、不消耗 RNG，不能当作 rollout 速度/动作统计。
没有 active rows 的 agent 沿用旧约定：`agent{i}_updated=0`、诊断占位 0，
聚合时排除；全空 batch 聚合占位 0，需联读 `updated_agents`。
warmup 根本不产生 update CSV 行，终端仍 `warmup (loss=N/A)`，不假装诊断为 0。

updates.csv 保留旧字段并追加上表六个聚合字段及 `agent0_...` 到 `agent4_...`。
TensorBoard 增加 `maddpg/actor_reg_mean`、`pre_tanh_abs_mean`、`pre_tanh_abs_max`、
`tanh_derivative_mean`、`accel_saturation_mean`、`turn_saturation_mean` 及全部 per-agent 字段。
终端保留 actor/critic/Q/y/TD/grad(A/C)，追加
`Replay-batch[reg=... |z|=... zmax=... tanh'=... sat(A/W)=.../...]`。
其中 zmax 是最大值，其余是 agent 平均值。

## 本轮验证（2026-10-17）

完整 `pytest -q`：165 项全部通过（原有 150 + 新增 15）。专项覆盖：

- V0 固定 seed/batch 与冻结旧公式连续三次更新：loss、全部网络参数、Adam 状态逐 tensor 完全一致；RNG 不变。
- V0/V1 公式、Critic 收到 tanh(z) 而非 z、active mask、诊断手算值及只读性全部通过。
- 明确构造 float64 z=[10,-10]，bounded reg 梯度约 [8.2446e-12,-8.2446e-12]，
  pre-tanh reg 梯度 [0.01,-0.01]，幅度比约 1.213e9；不依赖随机网络初始化。
- 同权重/同 observation 的 V0/V1 inference 与旧 forward 逐 tensor 完全相同。
- 无 mode 的旧格式 checkpoint 按 V0 加载，动作一致、文件 hash 不变，V1 拒绝加载。
- V1 五 Actor、五 Critic、两组 targets、十个 optimizer、global_step/update_count、mode 完整精确往返；V0 拒绝加载 V1。
- CSV、TensorBoard、warmup 文案及 Replay-batch 日志验证通过。

真实 smoke：WSL Ubuntu / conda uav / CUDA / 16 独立环境进程 / seed7，2,560 individual
transitions、24 次 joint update（120 Actor / 120 Critic 更新）。只缩小 batch32、buffer4096、
learning_starts128；环境、网络及其它算法超参数完全正式值。
训练期间对每次实际 Actor update 做只读预计算，120 次 reg/loss 公式核对最大误差 0。

下表是 24 次 update 中**跨 agent 聚合指标**的 min/max，不是逐元素范围，也不是性能结论：

| 指标 | 最小 | 最大 |
| --- | ---: | ---: |
| mean_actor_loss | -0.269378 | 0.493278 |
| mean_critic_loss | 0.0064632 | 26.47294 |
| mean_pre_tanh_abs | 0.339294 | 2.261238 |
| max_pre_tanh_abs | 1.083445 | 6.699984 |
| mean_tanh_derivative | 0.176403 | 0.870224 |
| mean_accel_saturation | 0% | 60.9397% |
| mean_turn_saturation | 0% | 56.6320% |
| mean_actor_reg | 0.00015927 | 0.00673183 |

Q/y/TD/梯度及所有 diagnostics 均有限；训练 action 最大绝对值 1.0。
四组各五个网络均发生真实参数变化，checkpoint 精确重载、独立四回合 deterministic
evaluation 正常；Reward accounting 最大误差 **0**。不解读 smoke 胜率。
还对本机现存旧 V0 smoke final.pt 运行两回合独立评估成功；本机未发现用户提到的 MADDPG 2M
结果，因此不声称本轮实际加载了该 2M 文件，旧格式由专项夹具及现存旧模型双重验证。

结果：`runs/v2_4_maddpg_v1_pretanh_smoke_seed7/`，主要查看 `smoke_validation.json`、
`training_summary.json`、`updates.csv`、`legacy_v0_evaluation.json`。
`frozen_scope.json` 记录 11 个冻结代码/配置文件和 12 个既有 MADDPG 结果文件的修改前后 SHA256，
全部相同。既有 checkpoint/CSV/结果均未删除、覆盖或改名。

本轮没有运行 500k 或 2M 正式训练。

## 500k 实验预注册（尚未执行）

正式使用 base.yaml + maddpg_pretanh_reg.yaml，seed7，CUDA，16 个环境；
buffer1M、batch1024、learning_starts=null 解析为 1024×300=307200，interval100。
总步数 500000 是全部并行环境累计 individual transitions，不是每个环境 500000。
真正学习区间为 307200→500000，共约 1929 次 joint update（当前调度包含 307200 首次 update）。
关注 320k 日志、400k checkpoint、500k final，不以胜率作首要结论。

预注册判断顺序：

1. |z| 是否再次从个位数爆到数百/数千；
2. acceleration/turn saturation 是否迅速逼近 100%；
3. tanh derivative 是否趋近 0；
4. Actor gradient 是否快速消失；现有 grad(A/C) 是总 loss 的裁剪前梯度，
   不等价于纯 policy-gradient 或 dQ/dz；若需解释该链路，使用同方法离线梯度审计；
5. 实际 rollout 速度是否仍长期贴 50m/s，attack window 是否至少没有恶化。

Replay diagnostics 不直接包含实际速度占下限时间比例，不能拿 Replay action 的 saturation
替代该量。速度与 dQ/da、dQ/dz/retention 仍需对 400k/500k checkpoint 进行一致的离线轨迹/梯度审计；
attack window 用既有 episodes.csv / deterministic evaluation 指标。与 V0 使用同 seed 集和同统计定义。
若 V1 明显抑制 z explosion，再讨论延长到 2M；若仍快速饱和，停止扩大训练，下一轮单独研究 Actor lr，
不要在本轮顺便改 lr 或其它变量。本轮不额外创造未约定的成功阈值。

服务器进入项目根目录且 `conda activate uav` 后运行（仅提供，未执行）：

```bash
python -m uav_mappo_2d.train_maddpg \
  --config configs/base.yaml \
  --algo-config configs/maddpg_pretanh_reg.yaml \
  --steps 500000 --seed 7 --device cuda \
  --output runs/v2_4_maddpg_v1_pretanh_500k_seed7
```

输出目录必须全新，trainer 会拒绝覆盖非空目录。V0 使用原 maddpg.yaml，原默认语义不变。

仅复跑本轮 smoke（也需全新输出目录）：

```bash
python scripts/smoke_maddpg.py --algo-config configs/maddpg_pretanh_reg.yaml \
  --output runs/v2_4_maddpg_v1_pretanh_smoke_repeat_seed7 --device cuda
```
