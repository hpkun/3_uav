# HISTORICAL / DEPRECATED — 不适用于 V2

下文仅为旧continuous 5v5历史记录，不代表当前V2，也不是V2训练证据；文内“当前Xu”也属于历史表述。当前规范见environment_spec.md，当前验收见validation_v2_5v5.md。

# 固定 5v5 升级验收

> 历史记录：本文件仅对应此前 continuous 5v5 环境，参数、测试和结果均不适用于当前 Xu 离散基线。旧 continuous 500k 结果同样不作为新基线的有效训练证据。当前定义见 environment_spec.md，当前验收见 validation_xu_5v5.md。

本轮完成固定5v5、确定性雷达及队内共享、partial Actor observation、自动武器、swept collision、reward分解和雷达约束规则蓝方。正式2M训练未运行。完整定义见 [environment_spec.md](environment_spec.md)。历史3v3训练结果保留，新模型输入维度不兼容旧checkpoint。

## 文件

修改：README.md、docs/environment_spec.md、pyproject.toml、configs/base.yaml、configs/smoke.yaml、scripts/sanity_check.py；src/uav_mappo_2d下config.py、env.py、opponents.py、models.py、mappo.py、train.py、evaluate.py、rendering.py；tests下test_environment.py、test_mappo.py、test_opponents.py、test_parallel_env.py。

新增：tests/test_radar_collision_reward.py、本报告。parallel_env.py原本已通过UAVCombatEnv常量确定shape，检查后复用，未改进程/自动reset实现。

## 参数与语义

| 项目 | 最终值 |
|---|---|
| 固定规模 / 地图 | 5v5 / 10km×10km，二维定高 |
| dt / 回合 | 1s / 150steps |
| 速度 / 加速度 / 转率 | 150–250m/s / ±20m/s² / ±15°/s |
| 出生 | x=±3500m，y=[-3000,-1500,0,1500,3000]m |
| 扰动 | 位置±200m、航向±5°、速度190–210m/s，双方各机独立采样 |
| radar | 4000m，存活队友探测结果完美即时共享，无记忆 |
| weapon | 距离≤1200m，abs(alpha)≤15°，abs(beta)≤50° |
| collision / warning | swept最小距离≤150m / 当前距离<450m且接近 |
| Actor / Critic | 68D partial / 60D full state |
| action / observation / rewards | (2,5,2) / (2,5,68) / (2,5) |
| parallel采样 | obs(E,5,68)，state(E,60)，mask(E,5)，action(E,5,2) |
| 正式MAPPO | MLP hidden128、16环境、128rollout、2Msteps、CUDA；PPO超参数不变 |

step顺序：保存旧位置→动力学→OOB→swept collision→同步weapon→最终存活pair的near penalty→计步→terminal/winner→potential/reward→obs/info。死亡原因互斥，同一collision victim和weapon victim分别去重。

collision使用相对运动r0、dr及tau=clip(-dot(r0,dr)/dot(dr,dr),0,1)，比较norm(r0+tau*dr)；相对运动近零取tau=0。

每队共享R=weapon+oob+collision+near+terminal+shape：weapon击杀+1/被击杀−1，自身OOB−1，碰撞损失每架−2，接近pair中每架−0.05×clip((d_prev−d_curr)/450,0,1)，胜负±5/平0。红方shape=0.1×(0.99Phi_next−Phi_current)，蓝方取相反数；真正terminal的Phi_next=0。Q_team目标只含当前detected敌机。敌方自身OOB/collision无本方即时正奖励，总reward不再强制零和。

Blue保留deterministic assignment、800m tail pursuit、速度控制和500m边界保护。无contact保持航向、speed_max接敌；出现contact、目标死亡或丢失时立即重分配，另每5步周期重分配。没有新增复杂避碰或战术FSM。

## 实际验证

均使用WSL Ubuntu，source conda.sh后conda activate uav。

```bash
pytest -q
uav2d-train --config configs/smoke.yaml
uav2d-evaluate --config configs/smoke.yaml \
  --checkpoint runs/smoke_fixed_5v5/checkpoints/final.pt \
  --episodes 20 --seed-start 300000
python scripts/sanity_check.py --mode all --episodes 100 --seed-start 200000
```

pytest：50项全部通过。覆盖5v5初始化/复现、双方雷达/共享/丢失/死亡/全局状态、武器角度和距离边界、最近目标/ID tie、多攻击者去重、同步双方击毁、endpoint和穿透碰撞、同队和跨队碰撞、多pair去重、死亡优先级、奖励分量核对、近距接近/远离、potential雷达约束、规则目标分配/丢失/死亡/周期、PPO finite和dead mask、多进程terminal reset及阵营交换对称性。

Smoke：128env steps、2个独立worker（PID32192/32193）、CPU、4次PPO update，2个完整训练episode，checkpoint正常保存。最终最近回合return=−12.192、W/D/L=0/0/100%、长度52.5；最终actor_loss=−0.0001、critic_loss=4.8485、entropy=1.8395、KL≈0、clip≈0、explained_variance=0.156，全部finite。

20回合确定性评估：0胜0平20负，return=−12.149208，平均长度47.95；红蓝alive=0/2.8，weapon kills=0/0.1，OOB=2.9/0，collision=2.0/2.2。该checkpoint只有128步训练，仅验证链路，不代表学习能力。

## Sanity实际结果

每组100回合，seeds=200000–200099。W/D/L为红胜/平/蓝胜；所有数量均为每回合均值。

| 模式 | W/D/L | 红蓝alive | 红蓝weapon kills | 红蓝OOB | 红蓝collision |
|---|---|---|---|---|---|
| rule-vs-rule | 0/100/0 | 0.21/0.21 | 0/0 | 0/0 | 4.79/4.79 |
| random-vs-rule | 4/0/96 | 0.09/2.77 | 0.07/0.71 | 2.54/0.21 | 1.66/1.95 |
| passive-vs-rule | 0/14/86 | 0/1.71 | 0/0.05 | 1.84/0 | 3.11/3.29 |
| pursuit-vs-rule | 2/95/3 | 0.59/0.61 | 0.01/0 | 0/0.01 | 4.41/4.37 |

| 模式 | 红胜/平/蓝胜率 | 平均长度 | 红方return | diagnostics |
|---|---|---|---|---|
| rule-vs-rule | 0%/100%/0% | 41.12 | −9.791067 | symmetry_gap=0 |
| random-vs-rule | 4%/0%/96% | 48.03 | −11.232669 | blue_combat_share=0.144603 |
| passive-vs-rule | 0%/14%/86% | 39.89 | −12.580778 | blue_combat_share=0.01 |
| pursuit-vs-rule | 2%/95%/3% | 77.75 | −9.059184 | blue_combat_share=0 |

symmetry_gap=abs(red_win_rate−blue_win_rate)。blue_combat_share=mean_blue_combat_kills/max(mean_blue_combat_kills+mean_red_oob+mean_red_collision,1e-12)，新分母包含collision。

## 已知局限和后续判断

规则对战已经被碰撞主导。对seed200000额外追踪：第14步R2/B2相撞（连续最小距离0.037m），第15步R0/B0、R1/B1、R3/B3相撞，第16步R4/B4相撞；双方各5架collision loss、0weapon kill、draw。这是双向迎头接近触发真实碰撞，不能据100%平局证明规则具有有效战术或证明没有阵营偏置。

随机/被动红方对战中蓝方虽胜率高，其胜利主要来自对方OOB/碰撞，不代表强武器作战能力。尾后追踪没有避免迎头相撞的机制；500m边界恢复也不保证在当前速度/转率下始终安全。这些结果按指定定义保留，没有为了提高胜率调参数或增加避碰shield。

实现及短训练链路通过验收，但尚未运行新环境长训练，也未验证新5v5 MAPPO的收敛性。历史3v3高胜率不适用于本环境。后续正式实验应关注collision是否持续主导、weapon kill是否能随训练增长。
