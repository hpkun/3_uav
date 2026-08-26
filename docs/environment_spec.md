# 固定 3v3 环境定义

## 1. 状态、动力学和终止

战场坐标为 `x,y ∈ [-5000,5000] m`。每架 UAV 状态为

```text
[x, y, v, psi, alive]
```

每个 `dt=1 s` 更新：

```text
v'   = clip(v + a dt, 150, 250)
psi' = wrap(psi + omega dt)
x'   = x + v' cos(psi') dt
y'   = y + v' sin(psi') dt
```

动作第一维映射到 `±20 m/s²`，第二维映射到 `±15 deg/s`。越出正方形边界的 UAV 在本步直接死亡，且不再参与本步攻击。

任一方全灭或达到第 150 步时任务终止；第 150 步比较存活数，数量相同为平局。150 步是任务终止，不是需要 bootstrap 的外部 truncation。

## 2. 对称初始化

红方基准位置为 `x=-3500`，蓝方为 `x=3500`；双方 ID 对应的 y 均为 `[-1500,0,1500]`。红方向右、蓝方向左。

每个 seed 生成一组位置、航向和速度扰动，然后以 y 轴镜像方式用于对应红蓝 UAV：x 扰动取镜像，y 扰动与速度相同，蓝方航向是红方航向的镜像。这样每一方边际分布满足规范，并消除随机数调用顺序造成的阵营偏置。

## 3. alpha、beta 与攻击

攻击者 i 指向目标 j 的 LOS 方位角为

```text
theta_ij = atan2(y_j-y_i, x_j-x_i)
```

定义：

```text
alpha_ij = wrap(theta_ij - psi_i)
beta_ij  = wrap(theta_ij - psi_j)
```

因此，攻击者在目标正后方且双方同向时 `alpha≈0, beta≈0`；迎头相遇时可以有 `alpha≈0, |beta|≈pi`。

攻击有效条件：

```text
d_ij <= 1200 m
|alpha_ij| <= 30 deg
|beta_ij| <= 45 deg
```

每个攻击者自动选择距离最近的有效目标。整步先根据同一状态收集双方攻击结果，再同步应用死亡；多个攻击者命中同一目标只记一次战斗击毁。

## 4. Actor observation

每架存活 UAV 的自身 5D 为：

```text
[x/5000, y/5000, (v-150)/100, sin(psi), cos(psi)]
```

其余 UAV 每个使用：

```text
[d/(sqrt(2)*10000), sin(alpha), cos(alpha),
 sin(beta), cos(beta), (v_j-v_i)/100, alive]
```

slot 顺序为自身、其余两架同队 UAV（ID 顺序）、三架敌方 UAV（ID 顺序），总维度 `5+5×7=40`。死亡实体 slot 清零；自身死亡时整个 observation 清零。

红蓝使用完全相同的计算规则。

## 5. Centralized critic state

固定顺序为 `R1,R2,R3,B1,B2,B3`。每架存活 UAV 使用：

```text
[x/5000, y/5000, (v-150)/100, sin(psi), cos(psi), alive]
```

死亡 UAV slot 清零，总维度 `6×6=36`。训练只有一个红方 team value `V(s)`。

## 6. Reward

每步红方三架 UAV 获得完全相同的团队 reward：

```text
r = r_event + r_terminal + r_shape
r_event = 本步蓝方损失数 - 本步红方损失数
r_terminal = win:+3, loss:-3, draw:0
```

战斗死亡和越界都计入 event loss。

势函数中：

```text
q_d     = exp(-d/D0), D0=3000 m
q_alpha = (1+cos(alpha))/2
q_beta  = (1+cos(beta))/2
q_ij    = 0.30 q_d + 0.35 q_alpha + 0.35 q_beta
```

对每个存活目标取所有存活攻击者中的最大 q，再对目标平均，得到 `Q_R` 和 `Q_B`：

```text
Phi = Q_R - Q_B
r_shape = 0.1 * (0.99 Phi_next - Phi_current)
```

真正 terminal 的 `Phi_next=0`。info 分别提供当前步的 `event_reward`、`terminal_reward`、`shaping_reward`、`total_reward` 和 `potential`；terminal summary 提供整个回合的累计分量。

## 7. 蓝方规则策略

规则策略每 5 步或目标死亡时重新分配目标。3v3 匹配使用 `itertools.permutations` 穷举最多 6 种唯一分配并最小化总距离。多余 UAV 选择最近存活目标，所有 tie 按 ID 确定性处理。

正常情况下，蓝机追踪目标航向后方 800 m 的尾后点。距离大于两倍攻击距离时使用最大速度，否则期望速度为目标速度加 20 m/s。接近边界且继续向外飞时，临时朝地图中心转向。

## 8. MAPPO rollout

正式训练 buffer 为 `[T,E]=[128,16]`。terminal 环境在 worker 内立即 reset，因此一个 rollout 可以包含多个 episode。团队级 GAE 在 terminal mask 处停止；若 rollout 在非 terminal 状态结束，则由 `V(s_T)` bootstrap。

PPO 以 `T×E` 个 environment transitions 为索引 shuffle，每个 transition 对应 3 个共享 Actor samples和一个 team Critic sample。只有 active UAV 参与 Actor loss。
