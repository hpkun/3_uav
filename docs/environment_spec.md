# 固定 5v5 环境定义

## 状态、动力学与初始化

二维定高（2D constant altitude）、双方各 5 架同质 UAV；地图为 10 km × 10 km，x,y∈[-5000,5000] m。实体物理状态为 `[x,y,v,psi,alive]`，连续动作 `[acceleration,turn_rate]∈[-1,1]^2` 分别映射到 ±20 m/s² 和 ±15 deg/s。

```text
dt = 1 s
v' = clip(v + a*dt, 150, 250)
psi' = wrap(psi + omega*dt)
p' = p + v'*[cos(psi'),sin(psi')]*dt
```

红蓝 nominal x 为 -3500/+3500，nominal heading 为 0/pi。nominal y 由 `(arange(N_AGENTS)-(N_AGENTS-1)/2)*1500` 得到 [-3000,-1500,0,1500,3000] m。位置各坐标 uniform(-200,200)，航向偏移 uniform(-5°,5°)，速度 uniform(190,210)。所有 UAV 和两队的随机变量分别独立，边际分布严格对称；相同 seed reset 可完全复现，单局不强制镜像。

固定常量为 N_AGENTS=5、OBS_DIM=68、STATE_DIM=60、ACTION_DIM=2；不提供任意 NvN 泛化。

## 雷达和队内共享

`team_detected_enemy_mask(team)` 返回长度 5 的 bool 数组。某敌机存活且距本队至少一架存活 UAV ≤4000 m 时 detected=True。圆形确定性距离探测，无 FOV、噪声、概率或扫描周期。探测结果立即、无损、无延迟地共享给所有存活队友（perfect team information sharing）。

探测没有记忆：敌机一旦离开所有队友的雷达范围，立即变为未知；死亡敌机不被探测。红蓝使用同一个 helper。雷达同时约束 Actor、规则目标候选和势函数目标集合；Critic 不受限制。

## Actor 和 Critic

环境 action shape=(2,5,2)、observation shape=(2,5,68)、reward shape=(2,5)。每架 Actor 的自身 5D：

```text
[x/5000,y/5000,(v-150)/100,sin(psi),cos(psi)]
```

其后按固定 ID 排列其余 4 架友机，再按 ID 排列 5 架敌机，每个 7D：

```text
[d/(sqrt(2)*10000),sin(alpha),cos(alpha),sin(beta),cos(beta),(v_j-v_i)/100,alive]
```

68D partial observation 不增加额外 radar flag。存活友机始终可见，敌机只有 detected 时填写；未知或死亡实体 slots 全零；自身死亡时整个 observation 全零。

60D centralized full global state 按 R0,R1,R2,R3,R4,B0,B1,B2,B3,B4 排列，每架 `[x/5000,y/5000,(v-150)/100,sin(psi),cos(psi),alive]`；死亡 slot 清零。训练一个红方 team value V(s)，Critic 使用雷达外实体的真实状态。

## 武器几何

```text
LOS = atan2(y_target-y_attacker,x_target-x_attacker)
alpha = wrap(LOS-heading_attacker)
beta = wrap(LOS-heading_target)
valid = distance <= 1200 m and abs(alpha) <= 15° and abs(beta) <= 50°
```

1.2 km automatic weapon 前向攻击半角为 ±15°（总30°），目标尾后薄弱半角为 ±50°（总100°）。同向尾追 alpha≈0,beta≈0 可攻击；迎头 beta≈pi 不可攻击。角度比较只加 1e-12 rad 浮点容差。

每名攻击者选择当前合法目标中最近者，同距离按目标 ID。根据同一个 alive snapshot 收集双方命中目标集合，然后同步应用死亡，允许双方当步互相击毁；多个攻击者命中同一个目标只记一次 weapon kill。combat_kills 严格只统计武器击毁。默认雷达4000 m大于攻击1200 m，合法武器目标必然在雷达内，攻击不额外查询雷达。

## 碰撞与事件顺序

碰撞适用于红红、蓝蓝和红蓝，阈值 ≤150 m，双方都损毁。每步保存运动前 p_old，以运动后 p_new 构成一步线性轨迹。对所有存活无序 pair：

```text
r0 = pA_old-pB_old
dr = (pA_new-pA_old)-(pB_new-pB_old)
tau = clip(-dot(r0,dr)/dot(dr,dr),0,1)
d_min = norm(r0+tau*dr)
```

相对位移平方 ≤1e-12 时取 tau=0。d_min≤150 时标记 pair 两架飞机；先收集所有 victims，再同步死亡，同一 UAV 在多 pair 中只计一次 collision loss。该 swept trajectory collision check 可以检测起点和终点均在阈值外的高速穿透。

每次 step 严格按以下顺序：

1. 保存 previous_positions。
2. 更新动力学。
3. OOB：任一坐标绝对值>5000 即死亡。
4. 在剩余 alive 中执行 swept collisions。
5. 在剩余 alive snapshot 中执行同步自动武器攻击。
6. 只对最终仍存活 UAV 计算 near-collision penalty。
7. step_count += 1。
8. 判 terminal 和 winner。
9. 计算 potential 和各队 reward。
10. 返回 observation / info。

OOB 后不能碰撞或攻击，collision 后不能攻击或被武器计杀；三类死亡原因互斥。任一队全灭或第150步即 terminal，按存活数判断胜负，相同为draw。150步是任务终止，truncated=False，GAE 不 bootstrap。

## Reward

每队所有5架飞机共享完全相同的 team reward，红蓝对称各自计算，不强制 blue_reward=-red_reward。

| 分量 | 默认公式 |
|---|---|
| weapon_reward | +1×本队武器击毁数 −1×本队被武器击毁数 |
| oob_reward | −1×本队OOB数 |
| collision_reward | −2×本队碰撞损失数 |
| near_collision_reward | 每个危险接近pair的每架飞机，向所属队记 −0.05×approach_fraction |
| terminal_reward | win +5 / lose −5 / draw 0 |
| shaping_reward | 红方 0.1×(0.99×Phi_next−Phi_current)，蓝方相反 |

最终 R_team 为以上六项之和。敌方OOB/collision不产生本方即时正奖励，只通过存活数和终局结果间接影响胜负。红红相撞红方−4；红蓝相撞双方各−2，均不计weapon kill。

Near-collision 仅对最终存活 pair 计算当前距离：当 d_curr<450 m 且 d_curr<d_prev，approach_fraction=clip((d_prev−d_curr)/450,0,1)。红蓝pair双方各记一次；同队pair同队记两次。远离不奖励也不惩罚，没有collision/weapon阶段死亡实体参与。

势函数保持：

```text
q_distance = exp(-d/3000)
q_alpha = (1+cos(alpha))/2
q_beta = (1+cos(beta))/2
q = 0.30*q_distance + 0.35*q_alpha + 0.35*q_beta
Q_team = mean_target(max_alive_attacker(q))
Phi = Q_red-Q_blue
```

target只能是本队当前detected的存活敌机；无探测目标时Q_team=0。攻击者为本队所有存活飞机，包括通过共享而间接获得contact者。真正terminal时Phi_next=0。

info包含红方当前六项reward、total_reward、potential和双方team_rewards，并累计combat_kills、oob、collision。episode_summary包含red/blue_alive、combat_kill、oob、collision、红方六项累计reward、return和长度。event_reward明确定义为weapon+oob+collision+near_collision之和；六项累计分量之和等于累计return。

## 雷达约束规则蓝方

目标候选只使用team_detected_enemy_mask。无contact时targets全−1；出现contact、当前目标死亡或脱离雷达时立即重分配，另每5步周期重分配。双方可各用独立RuleBasedOpponent实例。

friend数≤contact数时，穷举一对一匹配，最小化总距离；friend数更多时，先给每个contact匹配一名最佳追击者，剩余friend选最近contact。所有tie按ID处理，最多5!=120种permutations，无SciPy依赖。

有目标时追踪目标航向后方800 m的tail point。距离>2400 m时用speed_max，否则target_speed+20并clip。无目标时保持当前heading，期望速度speed_max，向前接敌。距边界500 m且继续向外时优先朝中心恢复，无contact也适用。没有复杂避碰、预测控制或attack/escape FSM。

## MAPPO 与记录

共享Actor为68–128–128–2，centralized Critic为60–128–128–1，仅输入维度改变。保持现有MLP、高斯tanh动作、PPO loss、value clipping、team GAE、dead-agent mask。RolloutBuffer必须显式传num_agents、observation_dim、state_dim，不再有旧维度默认值。

正式rollout为128×16，每环境独立常驻进程，worker terminal自动reset。PPO以T×E个transition shuffle，每项对应5个共享Actor样本和1个Critic样本；Actor只统计active UAV。gamma=.99、lambda=.95、clip=.2、10 epochs、minibatch512、总2M steps、默认CUDA均不变。explained_variance使用每次PPO更新完成后的critic重新forward计算。

终端、CSV、TensorBoard记录W/D/L、return、alive、weapon kill、OOB、collision、六项reward及现有PPO指标。旧3v3输入维度不兼容5v5 checkpoint，不提供迁移层。
