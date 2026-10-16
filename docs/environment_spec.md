# V2.3 Greedy-Shooter Blue：固定5v5连续二维基线

基于多篇二维无人机集群对抗文献共同建模范式构建的连续5v5基线，不是任何单篇论文的完全复现。本轮仅将固定Blue换为Greedy-Shooter Rule Opponent；V2.1的动力学、初始化、Radar、Weapon、collision/OOB、Reward、Actor68D、Critic60D、Gaussian网络与PPO核心完全冻结。V2.3仅标识本次Blue-policy实验，不代表重构环境物理。

## 1. 文献原则与实现约定分工

本轮用PDF技能核对于欣萌论文的封面与第三章原文。正式Blue来源与冻结环境来源分开，不混用作者、题名或低层动作接口：

- 于欣萌：《基于强化学习的无人机集群对抗方法研究》，大连理工大学硕士论文，2025，第3.2.2节(2)，PDF第40–41页/印刷页30–31，公式(3.17)–(3.19)：每步在探测内按欧氏距离选最近敌机，导航方向指向该目标当前位置，允许多机同目标，无显式协商/去冲突。论文原动作是二维连续控制力/导航向量，不是本项目的加速度/转率接口。此PDF未随项目保存，故只列作者、题名和章节，不创建本机绝对路径链接。
- [孙笠晋、李智：基于多智能体强化学习的无人机对抗技术研究](../基于多智能体强化学习的无人机对抗技术研究_孙笠晋%20(1).pdf)，PDF第4页/印刷页58，第2节和3.1节：二维、速度/航向、加速度与角速度动作、10000m地图。原文8v8、MADDPG，不宣称本项目复现其算法或积分顺序。
- 邓岩：《基于深度强化学习的无人机集群空战决策研究》，PDF第30页/印刷页21，表3.1：探测3000m、攻击1000m。该文含弹药、探测角、概率等，本项目只沿用距离尺度，不引入其余机制。
- 刘莹春：《基于强化学习的无人机集群协同打击策略研究》，第3章3.2.3、3.3.1及式(3.11)，PDF第38–40页/印刷页27–29：碰撞作为安全惩罚事件，以及攻击/边界事件反馈。只保留该原则作为冻结的V2.1奖励语义来源，不借用其低层动作、概率攻击或空间尺度。未随项目保存的论文只列作者/题名/章节，不创建本机绝对路径链接。

| Item | Literature-supported principle | Current implementation | Literature parameter / implementation convention | Notes |
|---|---|---|---|---|
| 二维状态与连续动作 | 孙笠晋第2/3.1节，位置/速度/航向与加速度/角速度 | [x,y,v,h]，[a_norm,omega_norm] | 原则有文献支持，动作归一化/积分时序为约定 | action-immediate semi-implicit保留V2 |
| 定高/固定5v5 | 二维集群仿真，刘莹春3.3.1为5v5 | 无z，双方各5架 | 场景约束与文献同类设置 | 不泛化NvN |
| 地图 | 孙笠晋3.1为10000m×10000m | ±5000m方形 | 尺度有文献支持，坐标原点为约定 | 对称OOB死亡是本项目规则 |
| 探测/攻击距离 | 邓岩表3.1 | 3000m / 1000m | literature parameter，来自距离尺度而非完整模型 | 不继承其导弹、命中概率或探测锥 |
| Radar/共享 | 有限观测、队内信息共享的通用范式 | 确定性360°、任一友机发现即共享、无记忆 | implementation convention | 不将隐藏敌机传给Blue |
| dt | 时间离散化 | 1s | implementation convention | 不宣称唯一论文值 |
| 初速/速度范围 | 有界运动学 | U(95,105) / 50–150m/s | implementation convention | 初速每机独立 |
| 加速度/转率 | 连续受限控制 | ±10m/s² / ±10°/s | implementation convention | 不是瞬时反向 |
| 出生与heading | 独立随机初始化 | 红x[-3800,-3200]，蓝[3200,3800]，y[-3000,3000]、同队间隔500，航向±5° | implementation convention | 无镜像五航线 |
| 68D / 60D编码 | 局部Actor/全局Critic | 固定ID slots、三角编码、valid、alive | implementation convention | 不用nearest-K |
| 武器角度 | 距离与攻击/目标几何判定 | ATA±30°、aspect±45° | 冻结的implementation convention | 尤其45° target aspect不是上述文献直接参数 |
| 碰撞 | 刘莹春式(3.11)以空间接近触发惩罚 | 50m swept检测、每步每机去重、不致死 | 原则有依据；50m与swept是implementation convention | 原文阈值/尺度/惩罚不照搬 |
| Blue目标选择 | 于欣萌(3.17)–(3.18)，探测内最近敌机 | 每机每步在team detected集合按(distance,ID)选目标 | 最近准则为文献规则；team sharing和等距ID为implementation adaptation/convention | 无target memory或assignment |
| Blue移动方向 | 于欣萌(3.19)，导航向量沿p_target-p_own | desired=atan2(delta_y,delta_x)，经原有限转率control输出[0,omega_norm] | 方向有直接依据；有限转率与a_norm=0为implementation adaptation | 不照搬二维控制力或增益κ，不引入基地目标 |
| 多机同目标 | 于欣萌PDF第41页，Greedy不去目标冲突 | 多架Blue可独立选择同一Red | 文献直接规则 | 无E-CBAA、capacity、Boids或显式协作 |
| Blue边界返回 | 本轮明确指定的简化反向航向规则 | 当前heading名义预测下一步越界时，朝h+pi转向 | continuous finite-turn implementation adaptation | 不声称原文证明有限转率下必不出界；无margin |
| Reward | 击毁/损失/碰撞/边界/结果的事件反馈原则 | +10/-10/-5/-20/±20，shared-team sum | 数值和shared-team reward adaptation为implementation convention | 不照搬原文步进惩罚或概率反馈 |

Greedy Shooter没有显式协商，不意味着本项目取消了冻结的team radar sharing：候选集仍来自team_detected_enemy_mask。论文中的导航方向被适配为目标bearing和有限转率控制，a_norm固定0并保留原boundary return；这些都标记为implementation adaptation，而非论文低层公式完全复现。

## 2. 最终完整配置

| 配置项 | 默认值 |
|---|---|
| N_AGENTS / OBS_DIM / STATE_DIM / ACTION_DIM | 5 / 68 / 60 / 2 |
| map_size / dt / episode_steps | 10000m / 1s / 300 |
| speed_min / speed_max | 50 / 150m/s |
| initial_speed_min / max | 95 / 105m/s |
| max_acceleration / max_turn_rate_degrees | 10m/s² / 10°/s |
| spawn_x_min / max | 3200 / 3800m，红取负蓝取正 |
| spawn_y_min / max / min_spawn_y_separation | -3000 / 3000 / 500m |
| heading_jitter_degrees | 5°，红围绕0、蓝围绕pi |
| radar_range / attack_range | 3000 / 1000m |
| attack_ata_degrees / attack_aspect_degrees | 30° / 45°，半角，inclusive |
| collision_distance | 50m，inclusive |
| weapon_kill_reward / weapon_loss_penalty | 10 / 10 |
| collision_penalty / oob_penalty / terminal_reward | 5 / 20 / 20 |
| hidden_size / Actor / Critic | 128 / 68-128-128-2 / 60-128-128-1 |
| log_std init / clamp | -.5 / [-5,1] |
| actor_learning_rate / critic_learning_rate | 3e-4 / 3e-4 |
| gamma / gae_lambda | .99 / .95 |
| clip_ratio / value_clip | .2 / .2 |
| entropy_coefficient / value_coefficient | .01 / .5 |
| max_grad_norm / update_epochs / minibatch_size | .5 / 10 / 512 |
| seed / parallel_envs / rollout_length | 7 / 16 / 128 |
| total_env_steps / device | 2000000 / cuda |
| log_interval_steps / checkpoint_interval_steps | 20000 / 100000 |
| evaluation_interval_steps / evaluation_episodes | 100000 / 100 |
| output_dir | runs/v2_1_5v5_seed7 |
| render width / height / margin | 800 / 800 / 45 |
| show_trails / trail_length / icon_source | true / 80 / 对抗环境图.png |

configs/base.yaml为正式配置。smoke保持既有轻量设置：seed11、隐藏64、epoch2、batch32、1280步、2进程、rollout64、log256、checkpoint/eval640、内置eval4、CPU；未为V2.1调整PPO。正式网络/PPO与V2逐项相同，models.py和mappo.py未修改。

## 3. 冻结的运动学与初始化

```text
a = a_norm * 10
omega = omega_norm * radians(10)
v_next = clip(v + a*dt, 50, 150)
h_next = wrap(h + omega*dt)            # [-pi,pi)
p_next = p + dt*v_next*[cos(h_next),sin(h_next)]
```

动作当前步即影响位移，死亡机冻结；全环境动作(2,5,2) float32，单机范围[-1,1]²，非法shape/非有限/越界拒绝（仅1e-6浮点容差）。速度、heading扰动每机独立，红蓝坐标分别采样。每队最多1000次rejection保证y间隔≥500，fallback均匀点+独立小jitter并置换ID，红蓝独立。不恢复镜像五航线。

## 4. 冻结的Radar、Actor、Critic和Weapon

team_detected_enemy_mask(team)：存在任一存活友机距敌机≤3000m即detected，360°range-only、即时理想共享、无contact memory；离开全队覆盖立即隐藏。Blue不能读隐藏敌机。

alpha=wrap(LOS-own_heading)，beta=wrap(LOS-other_heading)。Actor固定68D，0-based：

| 索引 | 内容 |
|---|---|
| 0–4 | x/5000,y/5000,2*(v-50)/100-1,sin(h),cos(h) |
| 5–32 | 其余4友机，按ID升序跳过自身，各7D |
| 33–67 | 5敌机固定ID，各7D |

7D=[d/(sqrt(2)*10000),sin(alpha),cos(alpha),sin(beta),cos(beta),clip(delta_v/100,-1,1),valid]。存活友机始终已知，敌机只填detected活机。死亡/隐藏slot零；死亡Actor整个68D零。

Critic60D按R0–R4/B0–B4排列，各[x/5000,y/5000,v_norm,sin(h),cos(h),alive]。死亡slot零，全局状态不受Radar限制。

Weapon：d≤1000 AND |alpha|≤30° AND |beta|≤45°，无fire动作，直接击毁；尾追有效、迎头无效。每攻击者选最近合法目标，等距小ID；同一pre-weapon alive快照收集后同步结算，支持mutual kill，多攻击者同victim只计一次weapon kill。collision不再排除任何活机；OOB机依旧排除。

## 5. Collision/OOB/terminal顺序

validate → old positions → continuous movement → OOB → collision event detection → weapon collection → synchronous weapon → step+1 → terminal/winner → event reward → next obs/state/info。

OOB：移动后任一|coordinate|>5000则死亡，恰等于边界不死；不能参加本步collision/weapon，无clamp/reflect/teleport。

Collision：沿本步同步线性运动段求最小中心距离，≤50m触发。友友、敌敌、敌我均检测；本步与任意其它活机发生碰撞的每架UAV计一次event，多pair不重复。本步计过后下一步再次接近仍可再计，因此累计events可超过5。collision不修改alive、不冻结运动、不直接终止回合，仍允许本步开火；若之后被weapon击毁，真正死亡原因仍是weapon。

只有weapon和OOB造成死亡：alive+weapon_losses+oob_losses=5（每队）。collision_events不加入死亡数或death share。后续策略动作可能受碰撞惩罚训练影响，但胜负函数不直接读取collision计数。

任一方全灭或300步终止；仅一方全灭另一方胜，同步全灭平；timeout比alive数，相等平。300步是真terminal，GAE不改。

## 6. 纯event-based共享奖励

```text
kill_reward[T] = +10 * 本队本步真实weapon kills
weapon_loss_reward[T] = -10 * 本队本步weapon deaths
collision_reward[T] = -5 * 本队本步参与collision的独立UAV数
oob_reward[T] = -20 * 本队本步OOB deaths
terminal_reward[T] = win:+20 / loss:-20 / draw或未终局:0
R_team = 五项之和
```

同team活Actor共享同一transition reward，team GAE保持原样。敌方OOB/碰撞不产生本方kill reward。collision与weapon可同step叠加；最后一次weapon kill和terminal可叠加。每回合return为Red团队总reward累计，不乘5，严格等于五项累计和。

RewardConfig仅上述五个权重，REWARD_COMPONENTS仅kill_reward、weapon_loss_reward、collision_reward、oob_reward、terminal_reward。没有potential、distance/angle/speed shaping，也无固定步进惩罚。

## 7. Greedy Shooter完整伪代码

RuleBasedOpponent()无状态，act(env,team=1)返回(5,2) float32。Boundary Return > Greedy Direct Pursuit > Straight Flight，仅三层：

```text
actions = zeros(5,2)
detected = team_detected_enemy_mask(team)
for each alive UAV i:
    heading = boundary_heading(env,team,i)   # 与V2.2完全相同
    if heading is None:
        j = nearest currently detected enemy (distance, enemy_id)
        if no j: continue                   # [0,0]
        delta = p_enemy[j]-p_self[i]
        heading = atan2(delta.y,delta.x)
    actions[i] = control(env,team,i,heading)  # [0,clipped finite-turn command]
return actions
```

每步重选、不保存target；死亡/隐藏敌机不进入detected，允许多机同目标。无距离模式切换、速度策略、虚拟追踪点、预测拦截、避碰或强规则对手模块。目标位置重合时math.atan2(0,0)返回0，动作仍finite，不增加特殊战术状态。

boundary_heading保持原样：当前速度/heading名义预测下一步越过±5000时，desired=wrap(h+pi)，再通过原10°/s有限转率controller转向。没有margin/safe point/中心返回/瞬时heading翻转或clamp/teleport；仍可能OOB。

## 8. 未改变的连续MAPPO与采样

Gaussian mean MLP、状态无关log_std、tanh采样及Jacobian logprob correction、atanh重算、pre-squash Normal entropy近似、dead active mask、float32 (T,E,5,2) buffer、team GAE、所有PPO核心参数保留。无action masks。父进程批量推理、先向全部worker发送后收集，每个worker独立env与Blue，默认16个真正子进程和CUDA learner。

普通resume恢复optimizer/计数，init-checkpoint只载兼容权重，不构成课程。旧连续权重形状可能相容，但对手已变化，旧实验不能作为当前Blue训练证据；本轮只执行接口smoke和规则行为验证，没有50k或2M训练，不迁移旧Categorical权重。总steps按全部环境合计，最后horizon缩短、不向上取整。

## 9. 统计定义

统一字段red_collision_events/blue_collision_events，env累计collision_events；不保留错误语义的collision_losses或red_collision兼容key。其余包括双方alive、combat_kill、weapon_loss、oob、attack_window、radar_contact_steps、length、return/result与五reward分量。

attack window为pre-weapon全部合法pair数，包含碰撞事件参与者；contact steps每step同team至少一个contact加1（OOB后/weapon前）。相同range与共享使双方contact steps对称。

death_total仅weapon_loss+oob，weapon_death_share分母0时为0。碰撞另报每局events与collision episode rate（该回合至少一事件的比例），不计算collision death share。sanity额外报告weapon_combat_episode_rate。

episodes.csv逐回合；updates.csv每更新记录最近100回合均值/WDL与八PPO指标；TensorBoard/console同步。sanity四模式各100局：greedy-vs-greedy双方独立Greedy对象，random-vs-greedy红U(-1,1)²，passive-vs-greedy红[0,0]，direct-pursuit-vs-greedy红方使用原direct-pursuit helper。第四组是控制语义一致性对照，不是另一种强弱对手。本轮两个对称组100局统计完全相同，所有step helper与正式Red策略动作差为0。另记录timeout episode rate，不能把timeout等同draw。

当前结果见[validation_v2_3_greedy_shooter.md](validation_v2_3_greedy_shooter.md)。validation_v2_1_5v5.md与validation_v2_2_rear_pursuit.md原样保留为历史，不覆盖原始数据。旧策略实验不能视为当前Blue训练证据。
