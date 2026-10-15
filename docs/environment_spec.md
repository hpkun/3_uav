# Fixed 5v5 Continuous 2D UAV Combat Baseline V2 环境规范

本环境是多篇二维多无人机对抗文献共同范式下构建的通用baseline，不是某一篇论文复现。采用用户本轮确定的参数与语义；不宣称每项参数都有某篇论文的唯一依据。固定5v5、二维定高，状态[x,y,v,heading]，红方vanilla continuous MAPPO，蓝方固定规则。没有导弹、弹药、概率命中、显式fire、6DOF、感知噪声、通信延迟、循环网络、attention、GNN、自博弈或课程学习。

## 完整默认参数

| 类别/配置项 | 值或定义 |
|---|---|
| N_AGENTS / OBS_DIM / STATE_DIM / ACTION_DIM | 5 / 68 / 60 / 2 |
| map_size / bounds | 10000 m / x,y∈[-5000,5000] m |
| dt / episode_steps | 1 s / 300 |
| speed_min / speed_max | 50 / 150 m/s |
| initial_speed_min / max | 95 / 105 m/s，独立均匀采样 |
| max_acceleration / max_turn_rate_degrees | 10 m/s² / 10°/s |
| spawn_x_min / max | 3200 / 3800 m，红取负、蓝取正 |
| spawn_y_min / max / min_spawn_y_separation | -3000 / 3000 / 500 m |
| heading_jitter_degrees | 每架独立±5°，红围绕0、蓝围绕π |
| radar_range | 3000 m，含边界 |
| attack_range / attack_ata_degrees / attack_aspect_degrees | 1000 m / 30° / 45°，含边界，角为半角 |
| collision_distance | 50 m，含边界，swept collision |
| blue_boundary_margin | 1000 m |
| blue_collision_avoidance_distance | 300 m，严格小于且closing |
| blue_assignment_interval | 5 steps |
| blue_tail_distance / blue_far_distance / blue_speed_advantage | 700 m / 2000 m / 10 m/s |
| weapon_kill_reward / weapon_loss_penalty | 5 / 5 |
| oob_penalty / collision_penalty / terminal_reward | 5 / 5 / 20 |
| shaping_scale / shaping_gamma | 0.5 / 0.99 |
| potential_distance_weight / ata_weight / aspect_weight | 0.25 / 0.375 / 0.375 |
| Actor / Critic | 68–128–128–2 / 60–128–128–1，MLP Tanh隐藏层 |
| log_std初始化 / clamp | 两维状态无关可训练参数 -0.5 / [-5,1] |
| actor_learning_rate / critic_learning_rate | 0.0003 / 0.0003 |
| gamma / gae_lambda | 0.99 / 0.95 |
| clip_ratio / value_clip | 0.2 / 0.2 |
| entropy_coefficient / value_coefficient / max_grad_norm | 0.01 / 0.5 / 0.5 |
| update_epochs / minibatch_size | 10 / 512 |
| seed / blue_strategy | 7 / 2 |
| total_env_steps / parallel_envs / rollout_length | 2000000 / 16 / 128 |
| log_interval_steps / checkpoint_interval_steps | 20000 / 100000 |
| evaluation_interval_steps / evaluation_episodes | 100000 / 100 |
| output_dir / device | runs/v2_5v5_seed7 / cuda |
| render width / height / margin | 800 / 800 / 45 px |
| show_trails / trail_length / icon_source | true / 80 / 对抗环境图.png |

权威可执行参数为configs/base.yaml，dataclass与配置默认值经测试核对。smoke覆盖网络宽64、PPO epoch2/batch32、2进程、64步rollout、1280总步、CPU等短测试设置，不改变物理/reward。smoke日志间隔256，checkpoint/eval间隔640，每次内置评估4回合。

## Reset与连续动力学

红蓝x、各队y、每架速度与heading扰动独立由环境RNG采样。每队最多1000次rejection sampling选5个y，任意两架同队y间距≥500。fallback在区间内均匀布点，加入独立小jitter（最多100m且受间距余量限制）并随机置换ID；红蓝独立生成。不沿用镜像五航线。

单机动作[a_norm,omega_norm]∈[-1,1]²，同时控制加减速和转向：

```text
a = 10*a_norm
omega = radians(10)*omega_norm
v_next = clip(v + a*dt, 50, 150)
heading_next = wrap(heading + omega*dt)      # [-pi,pi)
p_next = p + dt*v_next*[cos(heading_next), sin(heading_next)]
```

全环境动作(2,5,2) float32。非法shape、非有限或越界动作拒绝，仅容许1e-6浮点误差后clip。死亡飞机冻结。不存在离散动作、action mask或环境替红方避碰。

## 感知、68D Actor与60D Critic

team T的敌机j可见，当且仅当至少一个存活友机与其距离≤3000m。队内即时共享、无损无延迟、无记忆；离开全队覆盖即隐藏。存活友机始终已知。蓝方也只读取这个detected集合。

对观察者i和实体j：LOS=atan2(yj-yi,xj-xi)，alpha=wrap(LOS-heading_i)，beta=wrap(LOS-heading_j)。以下索引0-based：

| Actor索引 | 内容 |
|---|---|
| 0,1 | own x/5000, y/5000 |
| 2 | own v_norm=2*(v-50)/100-1 |
| 3,4 | sin(own heading), cos(own heading) |
| 5–32 | 其余4友机按固定ID升序（跳过自身），每架7D |
| 33–67 | 5敌机固定ID0–4，每架7D |

每个7D slot逐维：[d/(sqrt(2)*10000), sin(alpha), cos(alpha), sin(beta), cos(beta), clip((v_j-v_i)/100,-1,1), valid]。死亡友机、未探测/死亡敌机全slot为0；自身死亡整个68D为0。valid代表当前有效可观测实体，不是永久存活记忆。

Critic依次R0–R4、B0–B4，每架[x/5000,y/5000,v_norm,sin(heading),cos(heading),alive]六维，共60D；死亡slot全0。Critic看真实全局，不受雷达限制，不将隐藏信息传入Actor或规则策略。

## 自动武器、碰撞、出界与终局

合法几何：d≤1000 AND |alpha|≤30° AND |beta|≤45°。beta约0是同向尾追，约180°是迎头，迎头不可开火。无需独立fire动作，满足几何即击毁。

每个攻击者从本步合法敌机中选最近者，等距按ID；双方均基于同一pre-weapon alive快照收集victims，再同时置dead。支持相互击毁；多人打同一目标只算一次死亡和kill。

碰撞覆盖任意友敌pair：计算两机本步线性运动段上同步时刻最小距离，≤50m则双方损毁。不是任意两条轨迹相交就碰撞，必须时间同步。每架每步多pair去重。

固定step顺序：动作校验 → 保存old positions/Phi → 更新v/heading并移动 → OOB → swept collision → pre-weapon快照/窗口与contact计数 → 同步weapon → step+1 → terminal/winner → event/terminal reward → potential shaping → obs/state/active/info。

移动后任一坐标绝对值>5000即OOB；恰在边界未OOB。无反射、teleport或物理clamp。OOB机不参与本步碰撞/攻击；碰撞机不参与本步攻击。因此同一架一局仅一种死亡原因。渲染可裁剪画面残骸坐标，不修改物理状态。

任一方全灭或到300步终止；仅蓝全灭红胜、仅红全灭蓝胜、同时全灭平局；timeout比较存活数，多者胜，相同平局。300步是真正terminal，不做外部截断bootstrap。

## Reward与Phi

本队各agent共享team reward；Actor loss只计存活agent。

```text
R_T = R_weapon + R_oob + R_collision + R_terminal + R_shape
R_weapon = +5*本步weapon kills -5*本步weapon deaths
R_oob = -5*本步本队OOB deaths
R_collision = -5*本步本队collision deaths
R_terminal = 胜:+20 / 负:-20 / 平或非终局:0
```

敌方OOB/collision无己方即时kill奖励，但可能影响最终胜负。最后weapon kill与terminal可叠加。Red-Blue撞击双方各-5，没有kill reward。

```text
q_d = clip((3000-d)/(3000-1000),0,1)
q_alpha = (1+cos(alpha))/2
q_beta = (1+cos(beta))/2
q_ij = .25*q_d + .375*q_alpha + .375*q_beta
Q_T = mean_{team-detected alive enemy j}(max_{alive friend i}(q_ij))
无detected enemy则Q_T=0
Phi = Q_red-Q_blue
R_shape_red = .5*(.99*Phi_next-Phi_current)
R_shape_blue = -R_shape_red
```

终局强制Phi_next=0。质量仅针对team-detected敌机，攻击者为全部存活友机，允许队友共享contact。shaping零和，自身OOB/collision惩罚不要求零和。episode return累计Red团队reward，不乘5，等于五分量累计和。

## 三种连续蓝方

优先级：boundary safety > collision avoidance > combat > patrol。仅选择动作，不改env物理状态。

控制器：a_norm=clip((desired_v-v)/(10*dt),-1,1)；omega_norm=clip(wrap(desired_heading-heading)/(radians(10)*dt),-1,1)。

boundary：距任意边界≤1000且朝其外侧，朝clip(current_xy,-3500,3500)安全点转向，若重合则朝中心，目标速度100。

avoidance：最近其它已知活机（友机及detected敌机），距离<300且dot(p_other-p_own,v_other-v_own)<0时，朝远离方向转向，目标速度max(50,current_v-20)。无mask/shield。300m内敌机自然在雷达覆盖内。

Strategy1 Direct Pursuit：每架每步追最近detected敌机当前位置，等距按ID；目标速度120（未唯一规定部分的实现选择），无目标则patrol。

Strategy2 Assigned Tail Pursuit（默认）：全部存活友机与detected敌机pair按(distance,friend_id,enemy_id)升序greedy一对一匹配；目标数少时剩余友机追最近敌机。每5步、target死亡/失联，或无target时出现contact，重分配。目标点p_target-700*[cos(h_target),sin(h_target)]。距离>2000速度150，否则clip(v_target+10,50,150)。

Strategy3 Threat-aware Tail Pursuit：无威胁时与S2一致；选择最近detected威胁，要求敌→己距离≤1500、敌ATA≤45°、己作为目标aspect≤70°。比较己±10°/s的一步候选，选使敌→己aspect绝对值更大的方向，速度目标150。预测使用本次加速后己速度和敌机恒速度/航向一步外推，等分固定右转（实现选择）。无额外FSM，仍受boundary/avoidance优先级覆盖。

patrol：各机首次act保存自己的initial_y；在x=-1000、+1000且同一initial_y的两个waypoint往返，首次选较近者，距点<300时切换；目标速度100。红蓝rule-vs-rule的controller state彼此独立，每回合reset。

## 连续MAPPO与真正并行采样

Actor输出两维mean，状态无关可训练log_std初始化-0.5并clamp[-5,1]。u~Normal(mean,exp(log_std))、a=tanh(u)；deterministic为tanh(mean)。

```text
log pi(a|s) = sum_k [Normal.log_prob(u_k)-log(1-a_k²+1e-6)]
```

PPO重算将action clamp至[-1+1e-6,1-1e-6]再atanh，使用同样Jacobian修正。entropy为pre-squash Normal entropy近似，不宣称tanh后精确entropy。死亡action/logprob设0，active mask排除actor loss/entropy/KL等统计。Buffer actions为float32 (T,E,5,2)，无categorical action mask。保留GAE、PPO clipping/value clipping/gradient clipping等核心设置；不提供旧离散checkpoint迁移。

每个环境独立持久子进程，拥有独立env及blue controller；父进程批量推理，先向全部worker发动作，再收集。worker能并行执行，接收循环不等于串行环境。默认16workers、CUDA learner，NumPy环境在CPU。总步数为全部环境合计，最后horizon按剩余步缩短，不向上取整。

## 统计与验收

combat_kills/weapon_losses仅weapon；OOB/collision去重。attack_window_count为post-OOB/collision、pre-weapon快照中所有合法attacker-target pair累计，不是kill数；radar_contact_steps为该时点team存在contact则+1。对称圆形range使双方contact步相同，这是几何对称性。

episode summary包含双方alive、kill/loss、OOB、collision、window、contact，length、return、result及五reward分量。episodes.csv存逐回合；updates.csv每次更新存最近100回合均值/WDL及八PPO指标，TensorBoard和控制台同步。断点续训保留optimizer与计数；仅权重初始化重置二者，不自动形成课程。

实际测试、四组各100回合sanity、death share和行为问题见validation_v2_5v5.md。旧Xu与旧continuous文档/数据仅供历史参考，不是V2证据。
