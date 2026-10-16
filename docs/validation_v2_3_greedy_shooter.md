# V2.3 Greedy-Shooter Blue 验证报告

本轮只替换固定 Blue policy，并更新对应测试、sanity、行为审计和文档。未运行 50k 或 2M。结论：最近可见目标直追已恢复基本武器对抗，但碰撞惩罚仍主导 return，超时仍有 45%；建议先审阅本报告，下一轮单独审查 collision event/reward semantics，再决定是否进入 50k，不自动调整任何参数。

## 1. 修改文件与冻结范围

| 文件 | 修改 |
|---|---|
| src/uav_mappo_2d/opponents.py | 删除 rear_point，改为最近可见敌机当前位置直追 |
| tests/test_opponents.py | 替换旧虚拟点测试，覆盖 Greedy 目标/动作/边界/有限性 |
| scripts/sanity_check.py | 四组明确命名，增加超时统计和 helper 逐步一致性检查 |
| scripts/behavioral_audit.py | 删除 rear-point 诊断，记录真实目标及几何、动作、事件 |
| README.md | 当前正式 Blue、论文映射、验证入口 |
| docs/environment_spec.md | 当前规则与 implementation adaptation，保留环境参数 |
| docs/validation_v2_3_greedy_shooter.md | 本报告 |

与本轮开始的 Git HEAD 比较：env、config、models、mappo、parallel_env、train、evaluate、rendering、base.yaml、smoke.yaml 均未改变；control、boundary_heading、nearest_target 和原 Red direct-pursuit helper 的 AST 未改变。V2.1/V2.2 validation 原文未覆盖。证据：runs/v2_3_greedy_shooter/implementation_audit.json。

冻结设置仍为：5v5；10 km × 10 km；dt=1 s；300步；速度50–150 m/s、初速95–105；加速度上限10 m/s²；转率10°/s；team Radar 3000 m；Weapon 1000 m、ATA±30°、target aspect±45°；50 m swept collision 只计安全事件不致死；OOB致死；Actor68D/Critic60D；Gaussian Actor/MAPPO/PPO及训练超参数不变。Reward仍为武器击杀+10、武器损失−10、碰撞事件−5、OOB−20、终局±20/0，无potential。

## 2. 最终伪代码

```python
actions = zeros((5, 2))
detected = env.team_detected_enemy_mask(team)
for i in alive_agents(team):
    heading = boundary_heading(env, team, i)  # 原函数，完全不改
    if heading is None:
        target = nearest_target(env, team, i, detected)  # 距离、ID排序
        if target is None:
            continue
        delta = enemy_position[target] - own_position[i]
        heading = atan2(delta.y, delta.x)
    actions[i] = control(env, team, i, heading)
return actions
```

优先级仅 Boundary Return > Greedy Direct Pursuit > Straight Flight。a_norm恒为0；omega_norm=clip(wrap(desired-current)/(radians(max_turn_rate)*dt),−1,1)。无状态、无目标记忆、允许多对一；每步重新选择；不读 hidden/dead 敌机。没有加入分配、去冲突、逃逸、避碰、预测拦截、巡逻或速度策略。同位置 atan2(0,0)=0，动作仍有限，无额外战术 fallback。

## 3. 文献直接规则与项目适配

核对原文：于欣萌《基于强化学习的无人机集群对抗方法研究》（大连理工大学硕士论文，2025），第3章 §3.2.2(2)“贪婪射手”，式(3.17)–(3.19)，正文30–31页（PDF40–41页）。

| 内容 | 性质 |
|---|---|
| 在探测候选中按距离选最近目标、逐步重算 | 文献直接规则 |
| 朝目标当前位置的相对位置方向逼近 | 文献直接规则 |
| 无显式目标去冲突、可多机同目标 | 文献基线原则 |
| 使用冻结的 team_detected_enemy_mask | implementation adaptation；当前环境包含团队Radar共享，不等于原文个体探测范围 |
| 导航向量转为 desired heading，经有限转率输出 [a_norm,omega_norm] | implementation adaptation；不是照搬论文二维连续控制向量 |
| a_norm=0、等距ID优先、原boundary return | implementation adaptation |

原文候选还包含敌方基地；当前冻结环境没有基地，本轮没有引入。论文原导航有向量幅值/增益，当前仅复用方向思想，不引入其低层增益。这里“无显式通信/协作”指 Greedy policy 不额外做协商或分配，不否认当前环境已经提供团队共享探测。

## 4. 为什么放弃 V2.2

历史 V2.2 dynamic rear-point 实验在对称组只有0.31次武器击杀/局、25%武器交战回合、78%平局、平均300步，同队空间汇聚及重复碰撞显著，且长期跟踪不一定改善ATA至攻击窗口。结合旧行为审计，这是放弃该实现的实验依据；本轮没有继续调 rear distance，而换用公开完整的 Greedy Shooter 基线。它是该冻结环境下的失败结果，不宣称证明所有rear pursuit算法都失败。历史报告保留为 docs/validation_v2_2_rear_pursuit.md。

## 5. pytest 与 smoke

在 WSL Ubuntu 的 conda uav 环境实际执行。pytest结果：84 passed in 5.27s（命令 pytest -o addopts='' -q；覆盖项目默认addopts以得到明确简洁汇总）。

测试覆盖最近目标、等距ID、多机同目标、四象限直追、目标自身航向无关、500/1000/1500/2500 m无模式切换、零加速度、hidden/dead/no-contact、共享探测、边界优先和原有限转向、合法退化几何动作有限且在[-1,1]、旧机制不存在。

Smoke使用现有 configs/smoke.yaml，1280环境步、10次update、14个完整回合；CPU，2个真实进程worker（PID39885/39886）。不是正式训练，也不是策略能力评估。采样/update正常；checkpoint已保存并重新加载；Actor参数L2变化0.2984543；所有PPO指标有限；奖励分项求和最大误差0。末次episode汇总return −91.4286，W/D/L=0/0/14，仅证明接口联通。

| PPO指标 | min | max | last |
|---|---:|---:|---:|
| actor_loss | -0.009205905 | 0.01468439 | 0.01218891 |
| critic_loss | 81.82076 | 220.6341 | 220.6341 |
| entropy | 1.839705 | 1.844329 | 1.842672 |
| approx_kl | 0.00001723415 | 0.0002297249 | 0.00002082330 |
| clip_fraction | 0.000000 | 0.000000 | 0.000000 |
| actor_grad_norm | 0.8463578 | 1.185471 | 0.9395279 |
| critic_grad_norm | 11.06409 | 29.88433 | 28.62572 |
| explained_variance | -0.05348074 | 0.08429396 | 0.03874296 |

## 6. 四组各100局完整 sanity

每组seed=200000–200099，与V2.2一致。R/B为红/蓝，每局均值；return及奖励分项为Red团队口径。attack windows为环境原统计事件次数，不是新增的逐帧可攻击时长。

| 指标 | Greedy / Greedy | Random / Greedy | Passive / Greedy | Direct helper / Greedy |
|---|---:|---:|---:|---:|
| W/D/L | 53/6/41 | 2/0/98 | 0/0/100 | 53/6/41 |
| return | -114.55 | -92.25 | -100.1 | -114.55 |
| alive R/B | 2.32 / 1.87 | 0.02 / 4.69 | 0 / 4.99 | 2.32 / 1.87 |
| combat_kill R/B | 2.52 / 2.01 | 0.14 / 3.62 | 0.01 / 3.48 | 2.52 / 2.01 |
| weapon_loss R/B | 2.01 / 2.52 | 3.62 / 0.14 | 3.48 / 0.01 | 2.01 / 2.52 |
| oob R/B | 0.67 / 0.61 | 1.36 / 0.17 | 1.52 / 0 | 0.67 / 0.61 |
| collision_events R/B | 21.73 / 21.04 | 2.21 / 5.66 | 3 / 4.74 | 21.73 / 21.04 |
| attack_window R/B | 2.59 / 2.06 | 0.14 / 3.73 | 0.01 / 3.53 | 2.59 / 2.06 |
| radar_contact_steps R/B | 214.85 / 214.85 | 61.52 / 61.52 | 62.21 / 62.21 | 214.85 / 214.85 |
| episode length | 233.48 | 83.2 | 81.75 | 233.48 |
| collision episode rate R/B/任一 | 100% / 100% / 100% | 78% / 81% / 85% | 95% / 98% / 98% | 100% / 100% / 100% |
| weapon combat episode rate | 100% | 100% | 100% | 100% |
| timeout episode rate | 45% | 0% | 0% | 45% |
| weapon death share 合计 | 77.969% | 71.0775% | 69.6607% | 77.969% |
| OOB death share 合计 | 22.031% | 28.9225% | 30.3393% | 22.031% |
| kill_reward | 25.2 | 1.4 | 0.1 | 25.2 |
| weapon_loss_reward | -20.1 | -36.2 | -34.8 | -20.1 |
| collision_reward | -108.65 | -11.05 | -15 | -108.65 |
| oob_reward | -13.4 | -27.2 | -30.4 | -13.4 |
| terminal_reward | 2.4 | -19.2 | -20 | 2.4 |

死亡审计只允许武器/OOB两种实际死因；collision不致死，不能纳入死亡分类。

| 组别 | 阵营 | 死亡总数 | 武器死亡 | OOB | 武器占比 | OOB占比 |
|---|---|---:|---:|---:|---:|---:|
| greedy-vs-greedy | red | 268 | 201 | 67 | 75% | 25% |
| greedy-vs-greedy | blue | 313 | 252 | 61 | 80.5112% | 19.4888% |
| greedy-vs-greedy | combined | 581 | 453 | 128 | 77.969% | 22.031% |
| random-vs-greedy | red | 498 | 362 | 136 | 72.6908% | 27.3092% |
| random-vs-greedy | blue | 31 | 14 | 17 | 45.1613% | 54.8387% |
| random-vs-greedy | combined | 529 | 376 | 153 | 71.0775% | 28.9225% |
| passive-vs-greedy | red | 500 | 348 | 152 | 69.6% | 30.4% |
| passive-vs-greedy | blue | 1 | 1 | 0 | 100% | 0% |
| passive-vs-greedy | combined | 501 | 349 | 152 | 69.6607% | 30.3393% |
| direct-pursuit-vs-greedy | red | 268 | 201 | 67 | 75% | 25% |
| direct-pursuit-vs-greedy | blue | 313 | 252 | 61 | 80.5112% | 19.4888% |
| direct-pursuit-vs-greedy | combined | 581 | 453 | 128 | 77.969% | 22.031% |

## 7. 三代对称规则组对比

三组均为各自正式规则双方对战、100局相同seed区间。环境/武器/奖励冻结；对比的是不同规则下的行为，不是训练后的MAPPO。

| 指标 | V2.1 Chase/Escape | V2.2 Rear-Pursuit | V2.3 Greedy Shooter |
|---|---:|---:|---:|
| W/D/L | 54/0/46 | 8/78/14 | 53/6/41 |
| weapon kills R/B | 2.02/1.68 | 0.12/0.19 | 2.52/2.01 |
| weapon kills 合计 | 3.70 | 0.31 | 4.53 |
| weapon combat episode rate | 100% | 25% | 100% |
| attack windows R/B | 2.02/1.68 | 0.12/0.24 | 2.59/2.06 |
| collision events R/B | 10.34/7.79 | 56.16/48.58 | 21.73/21.04 |
| OOB R/B | 1.51/1.40 | 0.01/0.01 | 0.67/0.61 |
| episode length | 213.72 | 300 | 233.48 |
| return | −76.90 | −282.90 | −114.55 |
| weapon death share | 55.9758% | 93.9394% | 77.9690% |
| timeout | 历史未记录 | 100%（均值300且上限300） | 45% |

V2.1未记录完整100局timeout，不能用draw=0反推出timeout=0。V2.3碰撞合计42.77次/局，较V2.2的104.74下降59.16%，但仍高于V2.1的18.13。武器对抗恢复不代表碰撞问题已解决。

原 direct-pursuit Red 跨三代对手：

| 指标 | 对V2.1 Blue | 对V2.2 Blue | 对V2.3 Blue |
|---|---:|---:|---:|
| W/D/L | 100/0/0 | 67/21/12 | 53/6/41 |
| weapon kills R/B | 5/0.08 | 2.13/0.45 | 2.52/2.01 |
| weapon combat episode rate | 100% | 87% | 100% |
| attack windows R/B | 5.27/0.08 | 2.18/0.46 | 2.59/2.06 |
| collision events R/B | 2.61/1.30 | 29.37/39.98 | 21.73/21.04 |
| OOB R/B | 0/0 | 0/0 | 0.67/0.61 |
| episode length | 63.16 | 282.71 | 233.48 |

当前第四组是实现一致性检查，不是另一个独立强度档位。两组Red实现每一步输出完全一致（max abs difference=0、exact mismatch steps=0），所有汇总指标一致，无需改helper或调参数。

## 8. Greedy-vs-Greedy重点验收

每个回合均发生武器交战；平均4.53架武器击杀，攻击窗口R/B=2.59/2.06。平局从78%降到6%，平均回合从300降到233.48。45局达到300步上限，但只有6局判平；timeout和draw是不同口径，按冻结终局规则仍可在超时时判胜负。

死亡中武器占77.969%、OOB占22.031%；OOB没有成为主要死因，但由V2.2的0.01/0.01升到0.67/0.61，不能称为没有退步。原one-step boundary只在下一步预测越界时才转向，有限10°/s无法保证及时脱离边缘，本轮按要求冻结，没有修正。

对称组R53胜/B41胜/6平。94个非平局中Red胜率56.38%，Wilson95%区间46.30%–65.96%，对50%精确二项双侧p≈0.256。该100局未提供显著阵营偏向证据，但不证明长期完全对称；尚未追加多seed区间或交换初始化的对照，不能把抽样波动过度解释为结构性优势。

## 9. 逐步行为审计

主对象greedy-vs-greedy；对照direct-pursuit-vs-greedy。各20局，seed200000–200019，各16407条Blue存活动作记录。两组轨迹相同，因此不能视作40局独立样本。

记录目标ID/距离、双方当前位置、自身/期望航向、航向误差、转向/加速度、选中目标ATA/aspect和窗口前后、实际开火/受害者及其几何、碰撞、OOB、目标切换和连续追踪步数。不再有rear_x/rear_y/rear_distance等虚拟点字段。用独立距离/ID排序检查目标，用独立bearing/控制计算核对动作；实际武器目标可能不是机动目标，CSV分别记录两者，避免混淆。

| 审计项 | Greedy主组 | helper对照 |
|---|---:|---:|
| 最近detected目标检查 | 16407 | 16407 |
| 独立动作检查 | 16407 | 16407 |
| 最大动作误差（float32） | 2.97854e−8 | 2.97854e−8 |
| helper逐步差异 | 0 | 0 |
| 直追agent-steps | 14437 | 14437 |
| 无目标直飞agent-steps | 1955 | 1955 |
| boundary agent-steps | 15 | 15 |
| 目标切换 | 643 | 643 |
| 最长连续同目标 | 279步 | 279步 |
| Blue开火/击杀 | 33/33 | 33/33 |
| Blue有击杀/窗口回合 | 15/20 | 15/20 |
| Blue collision events | 444 | 444 |
| 同队碰撞pair-step（双阵营合计） | 65 | 65 |
| 跨队碰撞pair-step | 359 | 359 |
| Blue OOB | 16 | 16 |
| timeout | 13/20 | 13/20 |

审计子集的65%超时不同于完整100局的45%，只是前20个seed的子集差异；不拿子集替换完整结果。按运动线段独立重建collision pair和参与者，核对每队事件增量；pair-step不是agent-event，不能把两列直接相加当环境事件。

具体真实击杀：seed200000、第36步、Blue4选择Red1。步前自身位置(282.8242,−777.4642)，目标(66.4078,−1328.7102)，距离592.2062 m；自身航向−119.5121°，当前目标bearing −111.4347°，误差8.0774°，动作[0,0.8077371]。步前ATA8.0774°/aspect−45.5828°、窗口False；移动后距离576.7306 m、ATA5.9824°/aspect−29.6004°、窗口True，并实际武器击杀Red1。该步无碰撞/OOB。这说明并非只打印伪造目标或靠OOB结束。

删除虚拟点不代表不再长追：最长同一实体目标279步，step≥200仍有5014条直追动作、4次Blue开火。仍可出现有限转率直追绕转/错失窗口，不能宣称消除了所有绕圈。

旧V2.2同seed20局审计的同队pair-step为865、跨队277；当前为65/359。同队汇聚明显减少，而跨队近距穿越仍在。替换policy与同队碰撞下降一致，但没有额外控制变量实验，不能把所有变化都归因于单一几何机制。

## 10. Return及剩余问题、是否进入50k

对称Greedy组Red reward：
25.20（kill）−20.10（loss）−108.65（collision）−13.40（OOB）+2.40（terminal）=−114.55。

碰撞单项−108.65比其他分项净和−5.90大得多；100%回合存在碰撞。collision按50m swept与原去重语义计事件、不致死，同一近距交会在多步反复计数仍可能影响训练目标。这里没有修改距离、penalty、cooldown、去重窗口或添加避碰。

判定：稳定接敌、真实窗口/武器击杀、从V2.2低交战高平局退化中恢复、碰撞显著下降、武器为主要死因、helper一致等基本验证通过。但45%超时、仍偏高的碰撞及其主导return、上升的OOB、有限转率长期直追仍需保留为问题。短smoke不证明MAPPO可学或已收敛。

建议下一步先单独审查collision event/reward semantics，再由用户决定50k。不是因为未达到人为胜率门槛，也不是直接建议更强Blue。当前不实现任何新策略，不自动50k，不运行2M。

## 11. 原始结果与复现

原始文件位于 runs/v2_3_greedy_shooter/：
- pytest.log、smoke.log、smoke/（checkpoint与训练统计）
- sanity.json（四组100局完整结果）
- behavioral_audit.json、greedy-vs-greedy_blue_steps.csv、direct-pursuit-vs-greedy_blue_steps.csv
- implementation_audit.json（冻结/短训核对）、analysis_summary.json（对称性与轨迹汇总）

历史对比读取 runs/v2_1_validation/sanity.json 与 runs/v2_2_rear_pursuit/sanity.json，未覆盖。

以下命令在WSL Ubuntu、conda uav、项目目录内执行：
```bash
pytest -o addopts='' -q
uav2d-train --config configs/smoke.yaml --output runs/v2_3_greedy_shooter/smoke
python scripts/sanity_check.py --mode all --episodes 100 --seed-start 200000 --output runs/v2_3_greedy_shooter/sanity.json
python scripts/behavioral_audit.py --episodes 20 --seed-start 200000 --output runs/v2_3_greedy_shooter/behavioral_audit.json
```

已有结果已保存，复跑应使用新的output以免覆盖。测试与审计不是50k/2M训练授权。

