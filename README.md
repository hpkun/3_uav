# 固定 5v5 二维无人机空战与 MAPPO

面向科研实验的二维定高（2D constant altitude）基础环境：红方 5 架同质 UAV 使用共享 MLP MAPPO，蓝方 5 架 UAV 使用遵守雷达限制的确定性规则策略。固定 5v5，不提供任意 NvN 框架。

## 环境

| 项目 | 默认值与语义 |
|---|---|
| 地图 | 10 km × 10 km，坐标 ±5000 m |
| 动力学 | 连续 acceleration / turn_rate，dt=1 s |
| 速度 / 加速度 / 转率 | 150–250 m/s；±20 m/s²；±15 deg/s |
| 回合长度 | 最多 150 steps；任一队全灭也终止 |
| 初始化 | x=±3500；y=[-3000,-1500,0,1500,3000]；位置 ±200 m、航向 ±5°、速度 190–210 m/s 独立采样 |
| 雷达 | 4 km deterministic circular radar，存活队友探测结果立即完美共享（perfect team information sharing），无记忆 |
| Actor | 68D partial observation：自身 5D + 4 个友机与 5 个敌机的 7D 固定 slots |
| Critic | 60D full global state：R0–R4、B0–B4，每架 6D |
| 动作 / 环境观测 / reward | (2,5,2) / (2,5,68) / (2,5) |
| 自动武器 | 1.2 km，前向攻击半角 ±15°，尾后薄弱半角 ±50° |
| 碰撞 | 所有阵营组合，150 m，使用一步 swept trajectory 连续最小距离 |
| 接近辅助惩罚 | 450 m 内仍在接近的存活 pair，scale=0.05 |
| 蓝方 | 雷达约束目标分配、800 m 尾后追踪、速度控制、500 m 边界恢复 |

初始化为 independent but distributionally symmetric initialization：两边边际分布严格对称，单局不逐机镜像，同 seed 完全可复现。

未知和死亡敌机的 Actor slots 全零；友机始终已知；死亡 Actor 全零。Critic 可以读取完整真实状态。weapon / OOB / collision 为互斥死亡原因，combat kill 只统计武器击毁。

每队共享 reward：

```text
R_team = weapon_reward + oob_reward + collision_reward
       + near_collision_reward + terminal_reward + shaping_reward
weapon_reward = 1 * weapon_kills - 1 * weapon_losses
oob_reward = -1 * own_oob
collision_reward = -2 * own_collision_losses
near_collision_reward = 每架接近飞机 -0.05 * clip((d_prev-d_curr)/450,0,1)
terminal_reward = win:+5 / loss:-5 / draw:0
shape_red = 0.1 * (0.99*Phi_next - Phi_current)
shape_blue = -shape_red
```

敌方自身 OOB/collision 不给本方即时正奖励，因此红蓝总 reward 不强制零和。势函数只使用各队当前探测到的敌机；详细数学定义和事件顺序见 [环境规范](docs/environment_spec.md)。

## 代码

`env.py` 实现动力学、雷达、攻击、碰撞、reward 和观测；`opponents.py` 仅保留正式规则策略。`parallel_env.py` 是每环境一个常驻进程的真实并行采样，terminal worker 自动 reset。`mappo.py` 保持共享 Actor、centralized critic、team GAE 和 dead-agent mask。

Actor 为 `68–128–128–2`，Critic 为 `60–128–128–1`，只调整输入维度，网络层次和 PPO 算法不变。无导弹轨迹、弹药、开火动作、锁定、探测概率、通信网络仿真或自博弈。仓库根目录 PDF 为研究资料，`对抗环境图.png` 为渲染素材。

## WSL 环境和验证

```bash
wsl -d Ubuntu
source /home/hpk/anaconda3/etc/profile.d/conda.sh
conda activate uav
cd /mnt/c/Users/HPK/Desktop/3_uav
python -m pip install -e . --no-deps
pytest -q
uav2d-train --config configs/smoke.yaml
uav2d-evaluate --config configs/smoke.yaml \
  --checkpoint runs/smoke_fixed_5v5/checkpoints/final.pt --episodes 20
python scripts/sanity_check.py --mode all --episodes 100
```

Smoke 是短 CPU 链路验证，4 次 PPO update；不用于衡量策略性能。Sanity 的 random、passive、nearest pure pursuit 仅在脚本内部实现，pursuit 同样遵守队伍雷达信息。JSON 包含胜负、存活、武器击毁、OOB、碰撞、长度、return。blue_combat_share 分母包括蓝方击毁、红方 OOB 与红方碰撞损失。

## 正式训练

```bash
uav2d-train --config configs/base.yaml --output runs/fixed_5v5_seed7
```

默认 CUDA、16 并行环境、rollout=128×16，总预算 2,000,000 environment steps。每个向量 step 增加 16 steps；rollout 可包含多个回合。GAE 在 terminal 停止，非 terminal rollout 尾部由 V(s) bootstrap。PPO 参数保持 clip=0.2、gamma=0.99、lambda=0.95、10 epochs、minibatch=512。

默认输出目录为 `runs/fixed_5v5`，上述 seed7 命令使用独立目录。终端每约 20k steps 输出最近 100 个完整回合的 return、W/D/L、alive、weapon kill、OOB、collision、六项 reward 分量、长度及 PPO 指标。相同数据写入 `episodes.csv` 和 TensorBoard。

```bash
tensorboard --logdir runs/fixed_5v5_seed7/tensorboard
uav2d-evaluate --config configs/base.yaml \
  --checkpoint runs/fixed_5v5_seed7/checkpoints/best.pt \
  --episodes 500 --seed-start 2000000
```

周期评估使用固定 seed 集合；最终研究报告应另取独立 seeds。`--gif path/to/replay.gif` 可保存首个评估回合。

检查点有周期 `step_XXXXXXXXX.pt`、按评估胜率再按存活差选取的 `best.pt`、训练结束的 `final.pt`。继续同版本训练时使用 `--resume path/to/checkpoint.pt`。

旧 3v3 checkpoint 的 Actor/Critic 输入维度不同，与新 5v5 模型不兼容；不提供迁移层。历史训练文件保留在原目录，新实验应使用独立目录。
