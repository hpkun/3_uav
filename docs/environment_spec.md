# Xu-thesis-aligned baseline环境规范

本基线以徐金辉《面向多无人机对抗的多智能体强化学习信用分配方法》第3章和第4章课程为唯一主要环境来源，不混入裴谕论文策略。已核对原文3.1.2、3.1.3、3.2、3.2.2、3.3及4.4.1，包含公式(3.1)–(3.5)、图3.2、图3.5和图4.7。纸面页码23–31、44–48，对应PDF页31–39、52–56。原文位置：`C:/Users/HPK/Downloads/面向多无人机对抗的多智能体强化学习信用分配方法_徐金辉.pdf`。

这是一套可验证的对齐基线，不声称恢复未公开源码或完全复现论文全部实验。

## 论文对齐表

Status A=论文明确规定；B=project convention / implementation choice；C=shared-team adaptation。A/B表示参数有原文依据，但具体程序判定或编码存在约定。

| Item | Xu thesis | Implementation | Status | Notes |
|---|---|---|---|---|
| 5v5 | 双方各5架同质UAV | 固定5v5 | A | 不做任意NvN |
| map | 10km×10km | 名义[-5000,5000]²m | A/B | 中心坐标表达为项目约定 |
| speed | 初速100，最小50，最大150m/s | 相同 | A | 无初速扰动 |
| turn | 每步最大5°，正值左转 | [-5,-2,0,2,5]° | A | 每步增量，不是turn-rate |
| action | 5速度动作+5转向动作 | 10类categorical，索引0–9 | A/B | 索引为约定，两个0动作均保留 |
| radar | 2km局部观测、队内通信 | deterministic圆形探测并即时全队共享 | A/B | 边界≤、无记忆/噪声是清晰实现选择 |
| weapon | 攻击与薄弱半径0.6km | ≤600m自动武器 | A/B | 最近合法目标、ID tie和同步结算沿用最小实现 |
| attack/rear angles | 总30°/100° | 半角15°/50° | A | 使用等价尾后alpha/beta语义 |
| episode length | 最大300步 | 300步terminal | A | 不按外部truncation处理 |
| collision | dc=.15km，友敌碰撞均损毁 | 两机最小距离≤150m | A/B | 依任务要求按中心距离阈值解释 |
| avoidance buffer | de=.45km，优先避碰、只考虑最近机 | 最近中心距离≤450m收缩为3个steering动作 | A/B | 原文“缓冲区域重合”与中心距离判定并非无歧义等价 |
| avoidance direction | 增大自身攻击角 | signed alpha>0右转，<0左转，近0固定右转 | B | 遵循任务明确符号；原文左右措辞有不一致 |
| collision auxiliary reward | 最近机<.6km时距离差 | 同一当前最近机的(d_curr-d_prev)/1000，team mean | A/B/C | 单位转换和聚合显式记录 |
| base task reward | 100/80/20及对应负值 | 消灭/超时/击杀/自身损失七分量相加 | A/B/C | 原文分情况，叠加优先级由任务要求明确 |
| distance/angle auxiliary | 最近敌机距离差、最小攻击角差 | 当前可探测集合，km/rad，team mean | A/B/C | 前后contact有效性和存活均值为约定 |
| Strategy1 | 初始方向直飞 | 无风险时恢复初始heading和100m/s | A/B | 避碰后回正、离散转向量化为约定 |
| Strategy2 | 最近敌机方向 | 最近detected enemy的当前位置 | A/B | 无contact回初始方向，速度恢复100 |
| Strategy3 | 最近优先、初始容量2、敌死亡容量+1、无目标追最远友机 | deterministic逐轮提议/拒绝分配 | A/B | 非全局最小代价；tie/每步重分配为约定 |
| curriculum | S1 2M→S2 2M→S3 6M，加载前一策略 | 三配置，--init-checkpoint加载Actor/Critic权重 | A/B | 同时加载Critic为本项目选择 |
| observation encoding | 队友全信息+探测敌机，隐藏零向量；Critic全局 | Actor68D / Critic60D固定slots | B | 论文未给出该精确numerical layout/dimension |
| dt | 公式有Δt，未给数值 | 1s | B | 不能宣称与原实验时间尺度相同 |
| spawn distribution | 左右随机初始化、相反初始heading | x=±3500，居中y间隔1500，位置±200m | B | 精确分布非论文给出 |
| swept collision | 未公开离散实现 | 一步线性相对轨迹最小距离 | B | 数值防穿透，不是论文创新 |
| OOB semantics | 描述地图，未明确OOB death | 出界只诊断，不死亡/奖励/改变胜负 | B | 无反射、teleport、物理clamp |
| individual→team | 个体辅助奖励、被击落者−20 | 存活agent辅助均值；每架己方损失团队−20 | C | 保留vanilla team critic/GAE |
| PPO/networks | 论文完整方法含其它网络与表4.2超参数 | 现有MLP MAPPO超参数，Gaussian改Categorical | B | 未复现GRU/attention/信用分配；不将项目超参数称为论文参数 |

## 动力学、初始化和动作

状态为[x,y,v,heading,alive]，二维同高度。每步严格先用旧速度、旧heading移动，再应用一个离散增量：

```text
p_(t+1) = p_t + v_t * [cos(heading_t),sin(heading_t)] * dt
v_(t+1) = clip(v_t + delta_v[action],50,150)
heading_(t+1) = wrap(heading_t + delta_heading[action])
```

| action | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| delta_v(m/s) | -1 | -.5 | 0 | .5 | 1 | 0 | 0 | 0 | 0 | 0 |
| delta_heading(deg) | 0 | 0 | 0 | 0 | 0 | -5 | -2 | 0 | 2 | 5 |

动作2与7物理等价，均保留。MultiDiscrete action shape=(2,5)，每个元素为整数0–9。动作不乘dt：它们是每步增量。环境检查shape、整数、取值范围与当前可行动作，违反掩码会抛错，绝不在采样后静默替换动作。

初始速度全部100，heading为红0/蓝pi。nominal y=(arange(5)−2)*1500。红蓝位置各坐标扰动独立uniform(−200,200)，同seed可复现，单局不镜像。spawn_x、spawn_y_spacing、jitter、dt均为项目约定。

名义边界不影响运动、alive、reward或terminal。保留outside_map（当前存活且在名义范围外的数量）和ever_outside（曾在存活时出界的不同UAV数）诊断，均不参与训练目标。渲染仅对像素裁剪。observation_space允许归一化x/y和distance超过1，不裁剪真实数值。

## 信息、武器和事件顺序

队伍探测mask长度5：敌机存活，且距离任一存活队友≤2000m即为True，全队立即共享；敌机离开全队覆盖立即隐藏，死亡平台不提供探测。

Actor68D：自身5D [x/5000,y/5000,(v−50)/100,sin(heading),cos(heading)]，随后按ID排列其余4友机和5敌机的7D [d/(sqrt(2)*10000),sin(alpha),cos(alpha),sin(beta),cos(beta),delta_v/100,alive]。友机存活即已知；隐藏和死亡实体slot为0；自身死亡则全0。

Critic60D：R0–R4、B0–B4，每架[x/5000,y/5000,(v−50)/100,sin(heading),cos(heading),alive]，死亡slot清零，隐藏敌机真实状态保留。

alpha=wrap(LOS−attacker_heading)，beta=wrap(LOS−target_heading)。当d≤600、abs(alpha)≤15°、abs(beta)≤50°时自动命中；尾追可攻击，迎头不可。每个攻击者只选择最近合法目标，同距按ID；所有命中先收集再同步结算，允许当步双方击毁，多人命中同一目标只计一次weapon kill。角度边界有1e-12rad浮点容差。

step顺序：

1. 校验当前双方action masks，保存旧位置及旧可观测敌机metrics。
2. 用旧v/heading移动，再更新速度/heading。
3. 更新outside diagnostic，不产生死亡。
4. 在所有alive无序pair上同步结算swept collision。
5. 剩余alive snapshot上同步结算自动武器。
6. 计步并确定terminal/winner。
7. 计算任务奖励和存活agent辅助奖励。
8. 输出新obs、global state、active mask、action mask和info。

collision与weapon死亡互斥；collision victim不能当步攻击。任一队全灭或达到300步即terminal；仅一队全灭则另一队胜，双方同时全灭平局，超时按存活数判胜负。truncated=False。

## 碰撞与action constraint

对两架存活UAV，r0=pA_old−pB_old，dr=(pA_new−pA_old)−(pB_new−pB_old)。tau=clip(−dot(r0,dr)/dot(dr,dr),0,1)，相对位移平方≤1e-12时取tau=0。若norm(r0+tau*dr)≤150m则双方损毁。先收集所有collision victims再同步应用，多pair共享的UAV只记一次损失。

每架活机只参考当前最近的其它活机，友敌均可，距离tie按flattened team/agent ID。最近距离>450时10动作全可选；≤450时：

- alpha>约0或abs(alpha)≤1e-12：仅{5,6,7}，右转/保持。
- alpha<−1e-12：仅{7,8,9}，保持/左转。

buffer内速度动作全部禁用。死亡UAV只开放canonical no-op2且active_mask=0。红方available_action_masks为(5,10)bool；parallel返回(E,5,10)，env可为双方计算同一helper。规则蓝方在buffer中优先取5或9，不取其它策略动作。

该约束不保证绝不碰撞。尤其dt=1、v=100、两机相向时，每步相对位移可达200m；观测距离可能从487m跨到287m才触发，且触发动作改变的是下一步heading，本步仍沿旧heading运动。不能为了让结果好看而改触发阈值、动作大小或动力学时序。

## 完整reward与适配

对队T：

```text
R_T = kill_reward + own_loss_reward + terminal_reward + timeout_reward
    + distance_reward + attack_angle_reward + collision_avoidance_reward
kill_reward = 20 * 本步武器击毁敌机数
own_loss_reward = -20 * (本步己方武器死亡数 + 本步己方碰撞死亡数)
```

终局：只有敌方全灭时+100，只有己方全灭时−100；双方同时全灭draw，terminal_reward=0。这一处理及“最后击杀+20与消灭+100叠加”是明确的项目约定。双方仍有存活且到300步，数量优势+80、劣势−80、相等0；若同一步全灭，消灭结果优先，不再叠加timeout奖励。

辅助项在最终存活agent上计算，再对本队所有存活agent求mean；未满足条件的agent贡献0，也参与分母；本队无存活时辅助项为0。这是shared-team adaptation，与论文individual interface不同。

距离项：分别从转移前和转移后本队detected敌机集合计算每架agent到最近敌机的距离。前后均有contact且d_prev>1500m时值为(d_prev−d_curr)/1000，否则0。两个时刻最近敌机可不同；目标首次出现/全部消失时不产生虚假跳变奖励。

角度项：同样要求前后均有contact，取两个时刻detected敌机上的min abs(alpha)，值为theta_prev−theta_curr，使用radians。无contact为0。

碰撞辅助项：只在当前最近存活其它飞机j距离d_curr<600m时计算(d_curr−d_prev_same_j)/1000。取当前最近j在旧位置快照上的距离，不将两个不同neighbor的距离相减；这是对原文(3.5)的明确编码约定。远离为正、接近为负，每架只参考一名邻机，无额外任意scale。

所有reward返回shape=(2,5)，每队5架共享同值。info给出红方当前七项分量、total_reward、双方team_rewards；episode_summary包含累计七项、return/total_return、alive、weapon kill、collision、outside诊断和长度。七项累计之和等于total_return。彻底删除旧OOB、potential、Phi、pair quality及only-negative near penalty。

## 规则策略

三种Strategy每步重算detected目标，均优先执行避碰。正常转向把desired_heading误差量化到最近的[-5,-2,0,2,5]°，误差等距按action ID。转向量化为0时才进行速度恢复100的离散动作；这是当前最小确定性实现选择。

Strategy1：desired_heading为本队初始方向（红0/蓝pi），速度100；避碰结束逐步回正。

Strategy2：每机选择最近detected enemy，按(distance,enemy_id)；直接朝当前位置，无tail point，无contact则恢复初始方向。

Strategy3：每机先按(distance,enemy_id)排列detected targets并提议最近者。容量=min(alive_friends,2+dead_enemies)。每个目标保留距离最近的capacity名申请者，按agent ID打破tie，拒绝者尝试next-nearest，直到分配完或所有候选用尽。不求全局最优。容量中的敌死亡数按任务指定的全局存活计数计算，但不据隐藏敌机位置分配目标。无可用目标时朝最远活队友，等距离按ID；没有其它队友时恢复初始方向。成功分配则直接追当前位置。

## Categorical MAPPO、并行和课程

共享Actor为Linear–Tanh–Linear–Tanh–Linear(10 logits)，Categorical采样；确定性为masked logits argmax。invalid logits设为float最小有限值，概率为0；无全invalid分布。PPO evaluate_actions必须使用RolloutBuffer保存的采样时mask。actions为int64，shape=(T,E,5)；action_masks为bool，shape=(T,E,5,10)。active_mask只控制Actor损失贡献，死亡动作canonical2。

并行仍一环境一独立常驻进程，先发送全部动作后收集结果。reset返回obs/state/active/action masks/infos。terminal自动reset时，供下一次采样的四类数据均来自reset state；transition info保留terminal summary。buffer.insert在替换为next state前保存旧action mask。

team GAE、PPO clip、value clip、advantage normalization、gradient clipping、entropy bonus、更新后critic EV保持。正式16环境、rollout128，共2048transitions/update；gamma=.99、lambda=.95、10epochs、minibatch512及MLP hidden128为项目现有设置，并非论文表4.2完整配置。

Course1/2/3分别对Strategy1/2/3进行2M/2M/6M额外交互。--init-checkpoint只载入Actor/Critic权重，optimizer/global_step/episode counter/best evaluation保持新实例初值；--resume恢复上述断点状态。两者互斥。环境/RNG逐位重放不属于原有resume保证，本轮没有加入完整环境快照系统。

三个course配置均依赖config.py中的Xu默认环境/奖励，base.yaml展开列出相同数值，测试检查一致性。旧continuous checkpoint架构不同，不兼容、不迁移。未实现论文信用分配/attention/GRU，不据本基线结果推断论文算法性能。
