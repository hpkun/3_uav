# V2.1 二维连续5v5：完整实现与实际验收

日期：2026-10-16；WSL Ubuntu，conda uav，项目/mnt/c/Users/HPK/Desktop/3_uav。

结论：V2.1已经出现“探测→追击/逃逸→合法攻击窗口→weapon kill”的基本闭环。rule-vs-rule和pursuit-vs-rule各100局全部出现真实weapon combat。75项测试通过、1280步多更新smoke通过、50,000环境步学习性验证完成。正式2M未运行。

这不是“环境从此没有问题”：rule-vs-rule中OOB占实际死亡44.02%，碰撞事件惩罚对规则Red return影响很大；direct pursuit能够100/100战胜规则Blue，说明对手较弱；50k单seed结果仅支持初步可学习性，不支持论文算法性能结论。

## 1. 实际修改文件

- src/uav_mappo_2d/config.py：删除旧Blue与shaping配置，新增可为null的blue_escape_distance，更新五个事件权重与output_dir。
- src/uav_mappo_2d/env.py：collision_events不再致死、同step可weapon；删除全部potential函数/状态；拆开kill/loss reward；统一新统计。
- src/uav_mappo_2d/opponents.py：唯一无状态nearest-target chase/escape及名义预测boundary return。
- src/uav_mappo_2d/parallel_env.py、train.py、evaluate.py：移除blue_strategy传参/CLI，更新collision_events与五reward日志；评估补充碰撞回合率/真实死亡占比。
- configs/base.yaml、configs/smoke.yaml：同步新字段，另设V2.1输出目录；正式PPO与轻量smoke原有参数未变。
- scripts/sanity_check.py：四模式；pursuit恒a=0；death audit只含weapon/OOB，碰撞另报回合率；补充weapon combat回合率。
- scripts/behavioral_audit.py：只读记录真实chase/escape循环、合法weapon几何、OOB及事件前5帧；不参与策略控制。
- tests/conftest.py、test_environment.py、test_rewards.py、test_opponents.py、test_parallel_env.py、test_checkpoints.py：新语义测试；连续Actor/PPO原测试继续通过。
- README.md、docs/environment_spec.md、pyproject.toml：V2.1定位、完整参数和文献原则/实现约定。
- docs/validation_v2_5v5.md：只在顶部加HISTORICAL / SUPERSEDED BY V2.1，保留原行为数据。
- docs/validation_v2_1_5v5.md：本报告。

models.py、mappo.py、rendering.py未修改；运动学更新、初始化采样、Radar、观测编码、global_state、Weapon几何与同步结算保持原样。另对12个冻结env方法做了与V2 HEAD的AST比较，并对上述三份文件做全文比较，全部相同，证据见frozen_audit.json。历史runs数据未删除或覆盖。本轮所有输出单独位于runs/v2_1_validation/。

## 2. 被删除的旧机制

删除Strategy1/2/3与blue_strategy；persistent targets、greedy assignment/5步重分配、tail700m/远距加速/速度优势、patrol lanes/waypoints、1000m boundary margin与safe point/中心返回、300m collision avoidance、threat break/一步预测与所有相关配置、状态和辅助函数。

删除collision致死与collision_losses；现在同一活机每步可再次产生event。删除所有potential/shaping配置、_potential、_pair_quality、_team_attack_quality、_compute_potential及对应reward。没有引入head-on特判、侧向通过、right-pass、offset、shield、FSM、MPC/ORCA/Boids或任何新战术模块。

## 3. 最终完整参数

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
| blue_escape_distance | null → 运行时attack_range；允许显式数值 |
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


冻结的动力学：v_next=clip(v+10*a_norm*dt,50,150)，h_next=wrap(h+radians(10)*omega_norm*dt)，p_next=p+dt*v_next*[cos(h_next),sin(h_next)]。动作[-1,1]²可同时加减速和转向。68D局部Actor与60D全局Critic、Gaussian+tanh修正logprob、dead active mask、float动作buffer及GAE/PPO全部保留。

## 4. Collision新语义与OOB顺序

movement → OOB → swept collision event detection → weapon → terminal/reward。

50m inclusive、同步线性运动段最小距离；友友/敌敌/敌我均检测。每step参与至少一个pair的每架UAV计1event，多pair去重；下一step再次接近可再次计。collision_events可大于5，不影响alive，也不屏蔽本步或后续weapon/运动。OOB先死亡，不参与本步collision/weapon。

真正死亡仅weapon和OOB，每队alive+weapon_losses+oob_losses=5。collision不进入death_total。胜负函数只比较终局alive，不读取collision；但碰撞惩罚仍可影响RL选择的动作，不能把“不会直接致死”理解为对学习毫无影响。

## 5. 完整Reward

```text
kill_reward[T] = 10 * 本队本步weapon kills
weapon_loss_reward[T] = -10 * 本队本步weapon deaths
collision_reward[T] = -5 * 本队本步发生collision的独立UAV数
oob_reward[T] = -20 * 本队本步OOB deaths
terminal_reward[T] = win:+20 / loss:-20 / draw或非终局:0
R_team[T] = 五项之和
episode_return = sum_steps(R_team[Red])
```

同team活Actor共享团队transition reward，不乘agent数，team GAE不变。敌方OOB/碰撞不给本队kill reward。碰撞可与当步weapon、terminal叠加；无shaping或固定步进奖励/惩罚。smoke与50k所有回合分量核账最大误差均为0。

## 6. Blue完整伪代码

```text
actions = zeros(5,2)
detected = team_detected_enemy_mask(team)
threshold = attack_range if blue_escape_distance is null else blue_escape_distance
for each alive i:
    predicted = p_i + v_i*[cos(h_i),sin(h_i)]*dt
    if any(abs(predicted)>5000):
        desired_heading = wrap(h_i+pi)
    else:
        target = nearest detected enemy; equal distance -> smaller enemy ID
        if none: continue             # [0,0], keep current speed/heading
        delta = p_target-p_i
        if norm(delta)<=threshold: delta = -delta
        desired_heading = atan2(delta.y,delta.x)
    omega_norm = clip(wrap(desired_heading-h_i)/(radians(10)*dt),-1,1)
    actions[i] = [0,omega_norm]
return actions
```

无状态、每步按当前距离选最近detected敌机，不读隐藏敌机、不assignment、不记target、不调速。priority只有Boundary > Chase/Escape > Straight。boundary反向目标经有限转率执行，不直接改状态，不设margin。精确pi的wrap采用[-pi,pi)通用数值归属，不额外添加左右通过规则。

## 7. 文献与implementation convention

完整分工表及页码在[environment_spec.md](environment_spec.md)第1节。核对了孙笠晋PDF第4页（二维、加速度/角速度、10km尺度）、邓岩PDF第30页表3.1（3000m/1000m），以及刘莹春PDF第39–40页（nearest-target、近距escape/否则chase、碰撞惩罚事件）。

blue_escape_distance:null运行时跟随attack_range，不在构造阶段写死；显式数值也支持。文献描述了距离触发追逃，但没有公开具体阈值；本项目统一到1000m属于implementation convention，不是literature parameter。d==threshold归escape也是边界实现约定。

dt1、95–105初速、50–150速度、10m/s²、10°/s、出生坐标、68/60D编码、45°aspect、50m/swept、shared-team adaptation和有限转率boundary reversal全部明确标为implementation convention。没有将刘莹春原文的x/y加速度、概率武器、步进惩罚引入本项目。PDF技能的核对结果仅用于溯源与限制表述，没有改变用户指定参数。

## 8. Pytest实际结果

最终75 passed in 3.40s，见runs/v2_1_validation/pytest.log。包括：
- 旧字段/CLI/Blue状态与辅助函数确实不存在；null阈值跟随attack_range与显式阈值。
- 碰撞非致死、多pair去重、重复事件累计可超过5，下一step仍运动；碰撞与weapon同step均结算。
- OOB先结算、OOB -20、敌OOB无kill reward；五项reward隔离/核账、无potential。
- nearest每机独立/等距ID/每步重选、共享感知/隐藏独立、chase/escape等号、no-contact直飞。
- 边界名义预测严格>、反向有限转率/不修改状态、仍可真实OOB。
- V2冻结的初始化/Radar/观测/Weapon边界与同步、Gaussian tanh修正/dead mask、float buffer、GAE、PPO finite、checkpoint init/resume、真实子进程与auto-reset。

## 9. 1280步smoke实际结果

seed11、CPU、2个真实worker PID37241/37242、10次PPO update，完成11回合。Actor参数相同seed初始值L2变化0.342412；log_std=-0.496/-0.494。所有八PPO指标/CSV均finite，每个TensorBoard update scalar有10条；final.pt保存成功并实际加载后评估20局。输出runs/v2_1_validation/smoke/。

| 指标 | smoke训练11局均值 | final确定性20局 |
|---|---:|---:|
| W/D/L局数 | 0/0/11 | 0/0/20 |
| return | -106.364 | -113.250 |
| alive R/B | 0.000/4.545 | 0.000/4.550 |
| weapon kills R/B | 0.182/1.636 | 0.350/1.050 |
| weapon losses R/B | 1.636/0.182 | 1.050/0.350 |
| OOB R/B | 3.364/0.273 | 3.950/0.100 |
| collision events R/B | 0.909/6.364 | 1.450/3.700 |
| attack windows R/B | 0.182/1.727 | 0.400/1.050 |
| radar contact R/B | 81.273/81.273 | 96.300/96.300 |
| episode length | 103.818 | 115.450 |
| kill_reward | 1.818 | 3.500 |
| weapon_loss_reward | -16.364 | -10.500 |
| collision_reward | -4.545 | -7.250 |
| oob_reward | -67.273 | -79.000 |
| terminal_reward | -20.000 | -20.000 |

| smoke PPO | min | max | last |
|---|---:|---:|---:|
| actor_loss | -0.00810434 | 0.00533342 | -0.00810434 |
| critic_loss | 59.1180 | 352.324 | 181.704 |
| entropy | 1.83716 | 1.84782 | 1.84747 |
| approx_kl | 0.0000222247 | 0.000239069 | 0.0000222247 |
| clip_fraction | 0.00000 | 0.00000 | 0.00000 |
| actor_grad_norm | 0.766301 | 1.13138 | 1.06936 |
| critic_grad_norm | 8.16526 | 37.3204 | 27.1690 |
| explained_variance | 0.00412577 | 0.117974 | 0.0826373 |

短smoke胜率差不触发调参，不与正式128宽网络性能等同。

## 10. 四组100局sanity完整数据

统一seed200000–200099，每模式100局。rule双方独立RuleBasedOpponent；random红方U(-1,1)²；passive红方[0,0]（不是不移动）；pursuit红方nearest detected direct pursuit、a=0、无escape、只保留相同nominal boundary reversal；Blue均唯一规则。

| 指标（除局数/比例外均每回合均值） | rule-vs-rule | random-vs-rule | passive-vs-rule | pursuit-vs-rule |
|---|---:|---:|---:|---:|
| W/D/L局数（红视角） | 54/0/46 | 1/0/99 | 0/0/100 | 100/0/0 |
| W/D/L比例 | 54.00%/0.00%/46.00% | 1.00%/0.00%/99.00% | 0.00%/0.00%/100.00% | 100.00%/0.00%/0.00% |
| alive R/B | 1.810/1.580 | 0.010/4.500 | 0.000/4.990 | 4.920/0.000 |
| weapon kills R/B | 2.020/1.680 | 0.420/1.710 | 0.010/0.050 | 5.000/0.080 |
| weapon losses R/B | 1.680/2.020 | 1.710/0.420 | 0.050/0.010 | 0.080/5.000 |
| OOB R/B | 1.510/1.400 | 3.280/0.080 | 4.950/0.000 | 0.000/0.000 |
| collision events R/B | 10.340/7.790 | 1.040/6.940 | 0.950/4.670 | 2.610/1.300 |
| attack windows R/B | 2.020/1.680 | 0.420/1.740 | 0.010/0.050 | 5.270/0.080 |
| radar contact R/B | 191.030/191.030 | 83.640/83.640 | 69.840/69.840 | 45.040/45.040 |
| return | -76.900 | -103.300 | -124.150 | 56.150 |
| episode length | 213.720 | 104.090 | 88.960 | 63.160 |
| kill_reward | 20.200 | 4.200 | 0.100 | 50.000 |
| weapon_loss_reward | -16.800 | -17.100 | -0.500 | -0.800 |
| collision_reward | -51.700 | -5.200 | -4.750 | -13.050 |
| oob_reward | -30.200 | -65.600 | -99.000 | 0.000 |
| terminal_reward | 1.600 | -19.600 | -20.000 | 20.000 |
| collision episode rate red | 91.00% | 28.00% | 16.00% | 68.00% |
| collision episode rate blue | 92.00% | 40.00% | 37.00% | 52.00% |
| collision episode rate combined | 95.00% | 52.00% | 40.00% | 71.00% |
| weapon combat episode rate | 100.00% | 91.00% | 4.00% | 100.00% |

真正死亡占比只按weapon+OOB计算，collision不进入分母。零死亡安全返回0。

| 模式 | 范围 | death total | weapon | OOB | weapon death share | OOB share |
|---|---|---:|---:|---:|---:|---:|
| rule-vs-rule | red | 319 | 168 | 151 | 52.66% | 47.34% |
| rule-vs-rule | blue | 342 | 202 | 140 | 59.06% | 40.94% |
| rule-vs-rule | combined | 661 | 370 | 291 | 55.98% | 44.02% |
| random-vs-rule | red | 499 | 171 | 328 | 34.27% | 65.73% |
| random-vs-rule | blue | 50 | 42 | 8 | 84.00% | 16.00% |
| random-vs-rule | combined | 549 | 213 | 336 | 38.80% | 61.20% |
| passive-vs-rule | red | 500 | 5 | 495 | 1.00% | 99.00% |
| passive-vs-rule | blue | 1 | 1 | 0 | 100.00% | 0.00% |
| passive-vs-rule | combined | 501 | 6 | 495 | 1.20% | 98.80% |
| pursuit-vs-rule | red | 8 | 8 | 0 | 100.00% | 0.00% |
| pursuit-vs-rule | blue | 500 | 500 | 0 | 100.00% | 0.00% |
| pursuit-vs-rule | combined | 508 | 508 | 0 | 100.00% | 0.00% |

## 11. Weapon/OOB/collision行为分析与轨迹

rule-vs-rule每局都出现攻击窗口和击杀，100局合计370次weapon kill；pursuit-vs-rule合计508次。后者Red平均击杀5架、100/100胜、OOB为0，说明合法几何闭环成立，但也暴露Blue易被直接追击击败，不能以此声称高战术难度。

rule-vs-rule的OOB占死亡44.02%，没有超过weapon的55.98%，但不是可忽略项。random/passive的OOB分别占61.20%/98.80%，符合没有安全控制的动作定义，不能掩饰为weapon胜利。规则Blue对零动作Red几乎不击杀（0.05/局），因为近距escape后对方继续直飞出界；这不是双方完全没有Radar接触，双方仍有69.84 contact steps/局。

collision不再直接改变存活和胜负；rule-vs-rule仍95%回合有事件，Red平均10.34 events、惩罚-51.7，成为其return重要负项。按本轮要求不设置碰撞数硬门槛、不调权重。重复event可超过5，不代表统计错误；这是按每机每step计数的语义。固定规则Blue不会为了降低惩罚学习避碰，也没有避碰controller。

只读轨迹子集各20局（属于100局sanity子集，未新增训练）：

| 诊断 | rule-vs-rule | pursuit-vs-rule |
|---|---:|---:|
| Blue straight agent-steps | 2097 | 1955 |
| Blue chase agent-steps | 11043 | 1527 |
| Blue escape agent-steps | 4583 | 1471 |
| Blue boundary agent-steps | 27 | 0 |
| Blue chase→escape→chase次数 | 611 | 46 |
| weapon kills合计 | 76 | 102 |
| OOB deaths合计 | 57 | 0 |

rule seed200000的B0在step19前为chase、step31前距目标815.067m进入escape、step39前距离1072.548m再次chase（此前chase目标可由队友共享，即使自己与目标>3000m）。存在真实追逃再接敌循环，而不是只看代码推断。

合法weapon例：pursuit seed200000，第37步R2击毁B0，距离627.056m、ATA6.160°、aspect-35.153°，满足冻结的1000m/30°/45°。

真实OOB例：rule seed200000第116步B1，移动前x=4970.035，名义预测x=5066.451，规则已输出[0,-1]，有限转率移动后x=5069.918，仍出界死亡。没有偷偷加margin、反射或瞬时heading翻转。完整前5帧在behavioral_audit.json。

## 12. 是否运行50k及判断理由

运行。两个重点sanity模式100/100局均有weapon combat，规则双方370次击杀、weapon仍是较大的死亡来源，pursuit模式OOB为0。因此按“能够产生正常weapon combat且OOB没有明显主导”进入一次有限50k验证；不是把44.02% OOB解释为问题已经解决，也没有额外设定碰撞数门槛。

使用configs/base.yaml原始网络/PPO、seed7、CUDA和16个独立worker PID37478–37493，--steps 50000、--output runs/v2_1_validation/learnability。精确50000环境步（全部env合计），25次更新、322个完成回合；最后rollout缩短，总步数不向上取整。2M未运行。

## 13. 50k完整训练与deterministic evaluation

Actor参数相对相同seed初始值L2变化3.059528；log_std=-0.413/-0.465。所有CSV/PPO指标finite，八项TensorBoard update scalar各25条，reward核账误差0；checkpoint实际加载评估20局。详见training_audit.json、learnability/episodes.csv、updates.csv、tensorboard/和checkpoints/final.pt。

前100与后100回合为随机训练采样统计，不是固定种子确定性评估：

| 指标 | 前100局 | 全部322局 | 后100局 |
|---|---:|---:|---:|
| W/D/L局数 | 5/0/95 | 29/9/284 | 15/8/77 |
| return | -92.050 | -76.661 | -61.250 |
| alive R/B | 0.080/4.070 | 0.199/3.280 | 0.430/2.420 |
| weapon kills R/B | 0.750/2.180 | 1.357/2.677 | 2.090/2.610 |
| weapon losses R/B | 2.180/0.750 | 2.677/1.357 | 2.610/2.090 |
| OOB R/B | 2.740/0.180 | 2.124/0.363 | 1.960/0.490 |
| collision events R/B | 0.990/8.400 | 1.028/8.059 | 0.890/10.790 |
| attack windows R/B | 0.760/2.230 | 1.385/2.724 | 2.120/2.620 |
| radar contact R/B | 92.860/92.860 | 126.239/126.239 | 172.680/172.680 |
| episode length | 114.240 | 149.416 | 197.750 |
| kill_reward | 7.500 | 13.571 | 20.900 |
| weapon_loss_reward | -21.800 | -26.770 | -26.100 |
| collision_reward | -4.950 | -5.140 | -4.450 |
| oob_reward | -54.800 | -42.484 | -39.200 |
| terminal_reward | -18.000 | -15.839 | -12.400 |

按episode完成时global_step分入不重叠10k区间，不丢弃完整回合；不同区间局数不同，不能直接相加均值：

| 区间（env steps） | 完成局数 | W/D/L | mean return | weapon kill R/B | OOB R/B | mean length |
|---|---:|---|---:|---|---|---:|
| 0–10000 | 82 | 3/0/79 | -96.524 | 0.598/2.037 | 2.927/0.159 | 109.598 |
| 10000–20000 | 75 | 6/0/69 | -73.533 | 1.293/3.027 | 1.800/0.387 | 134.493 |
| 20000–30000 | 66 | 5/1/60 | -79.242 | 1.258/3.152 | 1.773/0.424 | 143.985 |
| 30000–40000 | 51 | 9/1/41 | -67.745 | 1.725/2.824 | 1.882/0.588 | 185.608 |
| 40000–50000 | 48 | 6/7/35 | -53.542 | 2.500/2.417 | 2.000/0.354 | 209.771 |

| 50k PPO指标 | min | max | final update |
|---|---:|---:|---:|
| actor_loss | -0.00833743 | -0.00339627 | -0.00437851 |
| critic_loss | 32.3783 | 187.098 | 33.8618 |
| entropy | 1.84192 | 1.95701 | 1.95701 |
| approx_kl | 0.00203401 | 0.00482061 | 0.00318184 |
| clip_fraction | 0.0163144 | 0.0598716 | 0.0340264 |
| actor_grad_norm | 0.272008 | 0.326853 | 0.286569 |
| critic_grad_norm | 2.19294 | 5.84519 | 4.29857 |
| explained_variance | 0.0221667 | 0.203844 | 0.203844 |

gradient norm记录clip前范数；大于.5并不意味着跳过gradient clipping。完整每update数值见updates.csv，不把单次损失下降解释为策略必然变强。

确定性评估对比：同一正式128宽初始seed7权重与50k final，在同一20局seed400000–400019，deterministic=tanh(mean)，未用smoke64宽模型作初始对照。

| 指标 | 训练前 | 50k后 |
|---|---:|---:|
| W/D/L局数 | 0/0/20 | 6/2/12 |
| return | -119.500 | -44.000 |
| alive R/B | 0.000/5.000 | 0.900/1.950 |
| weapon kills R/B | 0.000/0.150 | 2.300/2.350 |
| weapon losses R/B | 0.150/0.000 | 2.350/2.300 |
| OOB R/B | 4.850/0.000 | 1.750/0.750 |
| collision events R/B | 0.200/0.800 | 0.500/12.300 |
| attack windows R/B | 0.000/0.150 | 2.400/2.350 |
| radar contact R/B | 70.750/70.750 | 215.250/215.250 |
| episode length | 89.900 | 241.350 |
| kill_reward | 0.000 | 23.000 |
| weapon_loss_reward | -1.500 | -23.500 |
| collision_reward | -1.000 | -2.500 |
| oob_reward | -97.000 | -35.000 |
| terminal_reward | -20.000 | -6.000 |
| collision episode rate合并 | 25.00% | 75.00% |
| weapon death share合并 | 3.00% | 65.03% |

同seed20局显示武器击杀从0增至2.3/局，Red OOB从4.85降至1.75，return -119.5→-44，支持“开始学习”而非只验证参数有更新。训练后win30%/draw10%/loss60%，仍不能据此声称策略收敛、泛化或优于其他算法。

## 14. 当前仍存在的问题与未做事项

1. OOB尚未解决。规则双方44.02%真实死亡来自OOB；训练后Red仍1.75 OOB/局。名义一步反向目标在有限转率下不是安全保证，本轮按要求原样保留。
2. Blue较弱且固定。direct pursuit100/100胜，说明本基线易被简单追击利用；不增加新战术弥补，不据此夸大MAPPO性能。
3. 碰撞仍频繁且可重复扣分。规则Red collision reward -51.7/局，可能影响RL任务偏好；当前权重/语义为用户指定，未自动调小。
4. 去除shaping后早期奖励稀疏。50k仍大部分失败，PPO的critic EV最终约.204，不是充分拟合或可靠价值预测的证明。
5. 仅seed7的一次50k和20局deterministic。没有多seed、置信区间、大样本独立测试、正式2M或算法对比。现有改善不能写成最终学术结论。
6. 没有发现必须修改物理/Weapon/Observation才可出现kill的证据；已经形成基本追逃/武器闭环。未加入任何额外控制层、复杂工程框架或战术模块。

## 复现证据

- pytest.log：75项。
- smoke.log、smoke/episodes.csv、updates.csv、tensorboard/、checkpoints/；smoke_evaluation.json。
- sanity.json：4×100完整统计。
- behavioral_audit.json：各20局追逃循环、weapon/OOB具体事件与前5帧。
- learnability.log、learnability/episodes.csv、updates.csv、tensorboard/、checkpoints/final.pt。
- training_audit.json：smoke/50k有限性、参数L2、PPO ranges、分量核账、完整训练均值/分段趋势。
- frozen_audit.json：冻结env方法AST与模型/PPO/渲染源码一致性核对。
- initial_evaluation.json、learnability_evaluation.json：同seed20局训练前后确定性评估。
- 所有上述路径均在runs/v2_1_validation/，受既有.gitignore忽略；本报告保存关键实际数值。

所有实验均经wsl -d Ubuntu、conda activate uav运行；本轮没有复用V2训练结果。历史V2文档只加superseded标识，历史实验数据保留。
