# HISTORICAL / SUPERSEDED BY V2.1

本文保留V2原始行为数据，不代表当前V2.1。V2.1的碰撞是非致死安全事件，已删除potential及旧Blue机制；当前验收见validation_v2_1_5v5.md。

# V2 固定5v5连续对抗：实际验收报告

日期：2026-10-15。运行环境：WSL Ubuntu，conda uav；项目路径/mnt/c/Users/HPK/Desktop/3_uav。

结论：实现与数值链路验收通过，行为验收未达到正式训练门槛。武器击杀已经出现，但规则双方98.92%、简单追击对规则方97.04%的死亡仍来自碰撞。本轮没有修改参数美化结果，没有运行20k–50k学习性训练，更没有启动2M。不能宣称当前已经建立以尾后武器击杀为主的可学习科研基线。

## 1. 修改文件列表与范围

- configs/base.yaml、configs/smoke.yaml：V2参数、连续策略与默认S2。
- src/uav_mappo_2d/config.py、env.py：连续动力学、随机独立出生、感知、68/60D编码、OOB/碰撞/武器事件顺序、新reward及统计。
- src/uav_mappo_2d/models.py、mappo.py：Gaussian+tanh Actor、float动作buffer、active mask、连续PPO。
- src/uav_mappo_2d/opponents.py：S1/S2/S3、boundary/avoidance/patrol。
- src/uav_mappo_2d/parallel_env.py、train.py、evaluate.py：真正独立worker、连续动作接口、完整统计、updates.csv与评估JSON。
- src/uav_mappo_2d/rendering.py：明确绘图裁剪不改变物理状态；保留原图标及轨迹。
- scripts/sanity_check.py：四模式、完整统计及分阵营death share。
- scripts/behavioral_audit.py：新增只读轨迹诊断，保留碰撞/攻击前5帧，不改环境参数或状态。
- tests/conftest.py、test_environment.py、test_rewards.py、test_opponents.py、test_mappo.py、test_parallel_env.py、test_checkpoints.py：按V2语义重建测试。
- README.md、docs/environment_spec.md、pyproject.toml：统一V2定位。
- docs/validation_fixed_5v5.md、validation_xu_5v5.md：顶部标记HISTORICAL / DEPRECATED，原历史数据说明保留。
- 删除active configs/xu_course1.yaml、xu_course2.yaml、xu_course3.yaml，以及旧课程专用tests/test_curriculum.py。未删除历史实验目录；受Git跟踪的删除可通过版本历史恢复。

未增加导弹、弹药、概率命中、fire action、GRU/attention/GNN、自博弈、课程或复杂工程组件。旧Categorical checkpoint不迁移。weights-only初始化与resume是普通接口，不新增课程流程。

## 2. 最终完整参数、动力学与接口（对应最终要求2–14项）

[环境规范](environment_spec.md)包含完整参数表、逐维Actor/critic结构、全部公式和三种规则策略定义；权威可执行参数是configs/base.yaml。摘要如下：

| 项 | V2 |
|---|---|
| agents / obs / state / action | 固定5 / 68 / 60 / 2 |
| map / dt / episode | 10000m方形、±5000m / 1s / 300步 |
| initial speed / speed / acceleration / turn rate | U(95,105) / 50–150m/s / ±10m/s² / ±10°/s |
| spawn | 红x U(-3800,-3200)，蓝x U(3200,3800)，每队独立y U(-3000,3000)，同队间隔≥500m；航向独立±5° |
| radar / weapon / collision | 3000m / 1000m、ATA±30°、aspect±45° / 50m swept |
| Blue | 默认S2；boundary margin1000、avoidance300m且closing、assignment5步、tail700m |
| reward | weapon +5/-5，自身OOB -5/collision -5，terminal ±20/0，shape scale .5/gamma .99 |
| Actor / Critic | 68-128-128-2 / 60-128-128-1，Tanh MLP |
| PPO | lr两者3e-4、gamma.99、lambda.95、clip/valueclip.2、entropy.01、valuecoef.5、gradclip.5、epochs10、batch512 |
| 正式采样 | seed7、16独立进程、CUDA、rollout128、总2M步；本轮不运行 |

动作立即生效的semi-implicit公式：

```text
a = 10*a_norm; omega = radians(10)*omega_norm
v_next = clip(v + a*dt,50,150)
h_next = wrap(h + omega*dt)
p_next = p + dt*v_next*[cos(h_next),sin(h_next)]
```

Actor输出mean及两维可训练log_std（初始-.5，clamp[-5,1]），u~Normal，a=tanh(u)，确定性输出tanh(mean)。logprob=sum[Normal.log_prob(u)-log(1-a²+1e-6)]；PPO重算将a clamp后atanh。entropy采用pre-squash Normal近似。死亡action/logprob=0且不进入actor loss。Buffer float32 (T,E,5,2)，无action masks。

68D：self[0:5]为x/5000,y/5000,2*(v-50)/100-1,sin(h),cos(h)；友机[5:33]四slot，敌机[33:68]五slot。每slot为d/(sqrt(2)*10000),sin/cos(alpha),sin/cos(beta),clip(delta_v/100,-1,1),valid。友机跳过自身按ID升序；敌机固定ID。死亡或隐藏slot零，自身死亡全零。
60D：R0–R4/B0–B4各[x/5000,y/5000,v_norm,sin(h),cos(h),alive]；死亡零，critic不受雷达限制。

雷达是确定性360°、3000m inclusive、任一活友机探测即全队共享、无记忆；规则蓝方不读隐藏敌机。alpha=wrap(LOS-h_attacker)，beta=wrap(LOS-h_target)。合法武器要求距离≤1000、|alpha|≤30°、|beta|≤45°；尾追可、迎头不可。最近合法目标、等距ID、同快照收集并同步击毁，重复victim只计一次。

结算：validate → old positions/Phi → 动力学 → OOB → swept collision → pre-weapon window/contact → 同步weapon → step+1 → terminal → event+terminal+potential。OOB优先，死亡原因互斥，无反射/clamp/teleport或Red shield。

```text
R_T = 5*kills -5*weapon_losses -5*own_oob -5*own_collision
      + terminal(胜20,负-20,平0) + R_shape_T
q_d = clip((3000-d)/2000,0,1)
q_ij = .25*q_d + .375*(1+cos(alpha))/2 + .375*(1+cos(beta))/2
Q_T = mean_detected_enemy(max_alive_attacker(q_ij)); 无contact=0
Phi = Q_red-Q_blue
R_shape_red = .5*(.99*Phi_next-Phi_old); Blue取反；terminal Phi_next=0
```

敌方OOB/collision不给本方即时kill奖励；episode return不乘agent数，等于五分量累计和。attack windows累计全部合法pair；contact steps在post-OOB/collision、pre-weapon快照采样。

Blue优先级boundary > closing avoidance > combat > patrol。S1最近detected敌机direct pursuit（120m/s）；S2每5步或失联/死亡等触发greedy one-to-one，少目标时余机追最近，追target后方700m，远距>2000速度150，否则target速度+10 clipped；S3在S2基础上，对≤1500m、敌ATA≤45°/target aspect≤70°威胁进行一步±最大转率break并加速。预测敌恒速度/航向、己本次加速后速度，等分右转是明确实现选择，不是新增FSM。
boundary朝clip(own_xy,-3500,3500)安全点、100m/s；avoidance在<300m且closing时朝away、速度max(50,v-20)。patrol保存各自初始y，在x±1000两点往返、距离<300切换、速度100。

## 3. pytest与多进程smoke

执行pytest（覆盖默认-q，额外显示明确计数）：72 passed in 6.10s。输出见runs/v2_validation/pytest.log。测试含出生fallback、雷达边界/共享/隐藏、动作与semi-implicit、武器边界/重复/同步、50m swept collision、OOB优先、reward分量、potential、三种Blue优先级、Gaussian logprob、dead-mask梯度隔离、float buffer、GAE、PPO更新、checkpoint init/resume、多进程终局auto-reset。

最终1280步smoke：2个真实worker，PID36095/36096；64步rollout、10次更新、12个完成回合，CPU、网络宽64。输出runs/v2_validation/smoke/，日志runs/v2_validation/smoke.log。与正式128宽模型结构不同，只验证链路。

Actor参数相对相同seed初始值L2变化0.298265；log_std变为-0.496113 / -0.498996。说明连续Actor确实更新，不能等价为学会作战。所有CSV数值finite；五reward分量与return最大误差7.105e-15。八项TensorBoard update scalar均有10条。

| PPO指标 | min | max | last |
|---|---:|---:|---:|
| actor_loss | -0.00291701 | 0.0115347 | -0.00291701 |
| critic_loss | 3.48765 | 89.7205 | 25.2819 |
| entropy | 1.83827 | 1.84202 | 1.84202 |
| approx_kl | 0.0000189525 | 0.000310731 | 0.0000372402 |
| clip_fraction | 0.00000 | 0.000844595 | 0.00000 |
| actor_grad_norm | 0.818872 | 1.15188 | 0.922217 |
| critic_grad_norm | 2.30587 | 19.2912 | 8.62003 |
| explained_variance | 0.0117245 | 0.143788 | 0.0117245 |

最终训练窗口（12局）W/D/L=0/0/12，return=-43.594，alive R/B=0/4，kill=.25/1.58，weapon loss=1.58/.25，OOB=2.67/0，collision=.75/.75，window=.25/1.58，contact=77.17/77.17，length=96.33。只有1280步，不能据此判断学习趋势。

另用正式base配置跑256步CUDA/16进程链路探针（1次更新、尚无完成回合）：实际device=cuda，16个不同worker PID36244–36259，final.pt成功保存；不是20k学习性实验。日志runs/v2_validation/cuda16_probe.log，结果runs/v2_validation/cuda16_probe/。从代码确认先向所有worker发送step，再接收，各worker各自执行env与Blue。CUDA learner不代表CPU环境上GPU。

该CUDA更新八项指标均finite：actor_loss=-.00757123、critic_loss=.00315014、entropy=1.84039239、approx_kl=.00434126、clip_fraction=.05476562、actor_grad_norm=.42539577、critic_grad_norm=.23050199、explained_variance=.89603668；样本很短且尚无终局，不解释为策略性能。

## 4. 确定性smoke评估（20回合）

final checkpoint，独立seed300000–300019，tanh(mean)评估，0胜/0平/20负，return=-44.574092。

| 指标 | Red | Blue |
|---|---:|---:|
| alive | 0 | 4.20 |
| weapon kills | .05 | 1.90 |
| weapon losses | 1.90 | .05 |
| OOB | 2.45 | 0 |
| collision deaths | .65 | .75 |
| attack windows | .05 | 1.95 |
| radar contact steps | 85.80 | 85.80 |

length104.95，五reward依次weapon=-9.25、OOB=-12.25、collision=-3.25、terminal=-20、shaping=.175908。结果runs/v2_validation/smoke_evaluation.json。红方仍不具备安全飞行能力，不能以此短评估调环境参数。

## 5. 四种sanity：各100局完整结果

统一seed200000–200099，每个模式100局。rule-vs-rule双方独立S2 controller；random红方每步独立Uniform(-1,1)²；passive红方零动作；pursuit红方最近detected当前位置、120m/s、基本boundary，不用tail/avoidance；蓝方均S2。值为每回合均值，W/D/L均从红方视角。

| 指标 | rule-vs-rule | random-vs-rule | passive-vs-rule | pursuit-vs-rule |
|---|---:|---:|---:|---:|
| W/D/L局数 | 5/86/9 | 0/0/100 | 0/0/100 | 22/70/8 |
| alive R/B | 0.82/0.87 | 0.00/4.33 | 0.00/4.32 | 1.03/0.86 |
| weapon kills R/B | 0.04/0.05 | 0.04/1.52 | 0.00/0.80 | 0.22/0.02 |
| weapon losses R/B | 0.05/0.04 | 1.52/0.04 | 0.80/0.00 | 0.02/0.22 |
| OOB R/B | 0.00/0.00 | 2.89/0.00 | 3.60/0.00 | 0.00/0.00 |
| collision deaths R/B | 4.13/4.09 | 0.59/0.63 | 0.60/0.68 | 3.95/3.92 |
| attack windows R/B | 0.04/0.05 | 0.04/1.53 | 0.00/0.80 | 0.22/0.02 |
| radar contact steps R/B | 156.83/156.83 | 85.28/85.28 | 68.85/68.85 | 195.09/195.09 |
| episode length | 186.370 | 105.080 | 87.980 | 212.050 |
| return | -21.503 | -44.625 | -44.859 | -16.012 |
| weapon reward | -0.050 | -7.400 | -4.000 | 1.000 |
| OOB reward | 0.000 | -14.450 | -18.000 | 0.000 |
| collision reward | -20.650 | -2.950 | -3.000 | -19.750 |
| terminal reward | -0.800 | -20.000 | -20.000 | 2.800 |
| shaping reward | -0.002521 | 0.174534 | 0.140880 | -0.062399 |

四类均有雷达接触；rule/pursuit均有真实weapon kill，但频率非常低。rule双方合计100局仅9个weapon kill，pursuit合计24个，不能称作“稳定形成探测→尾后机动→击杀”。

## 6. 死亡原因审计

death_total=weapon_losses+oob+collision。分母是累计死亡架次，不是回合数，也不把collision pair当作death计数。零死亡时share定义0。

| 模式 | 范围 | 总死亡 | weapon | collision | OOB | weapon share | collision share | OOB share |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| rule-vs-rule | red | 418 | 5 | 413 | 0 | 1.20% | 98.80% | 0.00% |
| rule-vs-rule | blue | 413 | 4 | 409 | 0 | 0.97% | 99.03% | 0.00% |
| rule-vs-rule | combined | 831 | 9 | 822 | 0 | 1.08% | 98.92% | 0.00% |
| random-vs-rule | red | 500 | 152 | 59 | 289 | 30.40% | 11.80% | 57.80% |
| random-vs-rule | blue | 67 | 4 | 63 | 0 | 5.97% | 94.03% | 0.00% |
| random-vs-rule | combined | 567 | 156 | 122 | 289 | 27.51% | 21.52% | 50.97% |
| passive-vs-rule | red | 500 | 80 | 60 | 360 | 16.00% | 12.00% | 72.00% |
| passive-vs-rule | blue | 68 | 0 | 68 | 0 | 0.00% | 100.00% | 0.00% |
| passive-vs-rule | combined | 568 | 80 | 128 | 360 | 14.08% | 22.54% | 63.38% |
| pursuit-vs-rule | red | 397 | 2 | 395 | 0 | 0.50% | 99.50% | 0.00% |
| pursuit-vs-rule | blue | 414 | 22 | 392 | 0 | 5.31% | 94.69% | 0.00% |
| pursuit-vs-rule | combined | 811 | 24 | 787 | 0 | 2.96% | 97.04% | 0.00% |

rule与pursuit没有OOB，但碰撞压倒性主导；random/passive的OOB主导，与无边界控制的动作定义一致。不能因为S2避免了OOB，就认定其形成了有效空战。

## 7. 具体轨迹原因（只读审计）

额外各20局（seed200000–200019，属于上述100局子集），scripts/behavioral_audit.py重放，不改变任何环境/策略。记录碰撞前5帧位置/heading/actions与实际swept最小距离。完整runs/v2_validation/behavioral_audit.json。

| 指标 | rule-vs-rule | pursuit-vs-rule |
|---|---:|---:|
| collision pair数（不是死亡架次） | 78 | 74 |
| 敌我pair | 76 | 72 |
| 同队pair | 2 | 2 |
| 碰撞前heading差>135° | 74 | 72 |
| 碰撞步起点距离<300m | 63 | 74 |
| 双方执行avoidance | 61 | 1 |
| 一方执行avoidance | 3 | 71 |
| 无一方执行avoidance | 14 | 2 |

从数据可见，主要是敌我迎头碰撞，不是同队出生过近。pursuit红方没有避碰；“双方avoidance”的1个是同队Blue-Blue pair。

具体案例：rule seed200000，R3/B1第30步相撞。第26–29步前heading差几乎180°，双方接近150m/s，控制器输出转向接近0，持续形成互相对准的迎头路线。第30步前距离159.886m，双方才输出[-1,-1]减速/最大右转；本步终点距离125.649m，看终点似乎没撞，但同步swept最小距离27.766m，因此正确判碰撞。这不是swept误报或同一目标重复统计。

原因解释（由规则与轨迹推断）：
1. 对迎头敌机，其“尾后700m点”位于敌机的远端，初期仍使飞机朝敌机方向飞；独立随机出生消除了固定镜像lane，却不能消除相互追击逐渐对准的几何。
2. 迎头beta≈180°，自动武器合法条件不成立；在穿越前不会击杀解除碰撞风险。
3. 两机最高相对闭合约300m/s，进入300m避碰区往往只余一个decision step；150m/s、10°/s对应转弯半径约859m，减速每步最多10m/s。朝away的desired heading接近180°，有限转率无法瞬间反向，临近才启用不足以可靠避碰。即使双方已正确进入avoidance，也可能撞。
4. 少数错位交会或后续机动能形成武器窗口，说明武器链路有效，不代表窗口足够常见。

真实武器例子：pursuit seed200000第77步R0击毁B4，距离996.052m、ATA5.328°、aspect -40.025°；seed200001第113步R3攻击B3，距离478.685m、ATA -12.815°、aspect44.180°。这些都满足当前武器几何，没有靠放宽阈值得到。

## 8. 学习性验证决策、趋势与剩余问题

依用户条件，sanity基本合理后才运行20k–50k。本轮rule/pursuit的weapon death share分别只有1.08%/2.96%，collision为98.92%/97.04%，条件不满足。因此未执行20k–50k，没有该阶段训练趋势可报告。实际只做1280步多更新smoke（另有同seed重复验证一次）及256步CUDA16链路探针，没有2M或其他长训。

目前完成：V2物理/接口/reward/三种Blue/连续MAPPO/真正多进程/日志/测试/实际行为数据与文档。尚未完成：以weapon kill主导的合理规则对抗行为验证、正式网络20k–50k学习性验证、多seed长期训练和可靠学术性能结论。

下一步需要用户决策，不擅自改变核心参数。可讨论的方向是规则策略的提前侧向解冲突或避免互相迎头汇聚，而不是立即修改武器角度、碰撞半径、reward或通过屏蔽Red动作掩盖问题。是否允许改变当前300m才触发/直接away的规则语义，需要确认后再实施；本轮不实现。

## 9. 证据与复现

- runs/v2_validation/pytest.log：72 tests。
- runs/v2_validation/smoke.log、smoke/episodes.csv、smoke/updates.csv、smoke/tensorboard/、smoke/checkpoints/。
- runs/v2_validation/smoke_audit.json：参数变化、finite、PPO ranges、reward核账。
- runs/v2_validation/smoke_evaluation.json：20局确定性评估。
- runs/v2_validation/cuda16_probe.log、cuda16_probe/：默认CUDA/16 worker短探针。
- runs/v2_validation/sanity.json：四类各100局完整均值、双方/合并death audit。
- runs/v2_validation/behavioral_audit.json：实际碰撞与weapon事件、前5帧轨迹。

原始输出在runs下，受既有.gitignore忽略；此文档保留关键数值。不使用任何旧Xu50k或旧continuous500k作为当前证据。
