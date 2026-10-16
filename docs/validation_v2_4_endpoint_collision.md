# V2.4 Endpoint Collision 验收报告

## 1. 结论

只完成两项正式行为修改：collision由swept minimum-distance改为post-movement endpoint center-distance；collision_penalty由5.0改为0.2。threshold保持50m，RR/BB/RB全部保留，每UAV每步最多一个event，non-lethal且不阻止同一步weapon。

验收通过：107项pytest；100局/23348步Greedy对照轨迹完全相同；100/100个seed与V2.3 audit Liu-style反事实逐局一致，最大误差5.684341886080802e−14（<1e−9）；1280步MAPPO smoke通过。未运行50k/2M，未做参数搜索或额外调整。

建议用户检查后可以进入50k短程训练，以验证可学习性；本轮没有自动执行。不能从规则组胜率或1280步smoke推断MAPPO已有效或收敛。

## 2. 为什么修改、文献依据和适配

V2.3历史审计（100局seed200000–200099）：swept events R/B=21.73/21.04、collision reward=−108.65/−105.20；58.95% pair-step为Swept-only，71.88%为RB，collision pooled absolute share约58%。用户明确选择endpoint+0.2，把collision恢复为辅助安全约束，本轮不再探索其它规则。

核对原文：刘莹春《基于强化学习的无人机集群协同打击策略研究》，§3.2.3式(3.11)，PDF39页/正文28页；相邻§3.3.1也描述距离碰撞与−0.2惩罚。原公式是存在任一其它UAV的当前距离<0.5时对个体施加−0.2。

| 内容 | 定位 |
|---|---|
| 当前中心距离触发小额collision penalty | 文献安全惩罚语义 |
| collision系数0.2 | 参考原文式(3.11) |
| 50m绝对尺度（不是0.5归一化尺度） | implementation adaptation，保留项目物理尺度 |
| inclusive ≤50m，而非原文严格< | implementation adaptation，保留本项目inclusive阈值约定 |
| shared-team按本队unique participants累加 | implementation adaptation |
| RR/BB/RB双方均实现，post-OOB/pre-weapon、非致死 | 当前明确采用的项目实现约定 |

不声称完整复现论文；没有引入原文步进惩罚、概率攻击、归一化地图、原动作接口或其它算法。

## 3. 修改文件

| 文件 | 修改范围 |
|---|---|
| src/uav_mappo_2d/env.py | 删step旧位置复制，_resolve_collisions()无参数，endpoint距离计算；更新版本docstring |
| src/uav_mappo_2d/config.py | RewardConfig.collision_penalty=0.2，只有此默认值变化 |
| configs/base.yaml | collision_penalty: 0.2，其余参数不变 |
| tests/test_environment.py | 当前endpoint语义测试，保留非致死/顺序/去重验证 |
| tests/test_rewards.py | 当前−0.2/−0.4奖励及与weapon叠加 |
| tests/test_collision_audit.py | 明确V2.3历史Current与V2.4生产映射，保留历史swept数学 |
| scripts/validate_v2_4.py（新增） | 独立验收、逐步物理对照、逐局反事实、可选原1280步smoke |
| README.md | 当前V2.4、来源/适配、验证命令和历史工具边界 |
| docs/environment_spec.md | 当前collision与公式、范围和来源 |
| docs/validation_v2_4_endpoint_collision.md（新增） | 本报告 |

核心代码对比只有env.py、config.py、base.yaml按上述范围改变。scope_validation.json保存前后SHA256：opponents/models/mappo/train/evaluate/parallel_env/rendering/smoke.yaml均未改变；四份历史validation/audit文档和scripts/collision_audit.py均未改变。env.py内动力学、初始化、Radar、Weapon、OOB、终止/胜负、obs/state等方法未改动。

无collision cooldown、shield、avoidance、debounce、pair memory或新环境状态。未改episode_steps/map/speed/turn rate/terminal/shaping/网络/PPO/16进程CUDA默认训练设置。

## 4. 当前生产函数语义

```python
def _resolve_collisions(self):
    first, second = upper_triangle_pairs(10)
    valid = alive[first] & alive[second]       # post-OOB / pre-weapon
    first, second = first[valid], second[valid]
    delta = current_positions[first] - current_positions[second]
    collided = sum(delta * delta, axis=1) <= 50.0 ** 2
    involved = unique(concat(first[collided], second[collided]))
    events = bincount(involved // 5, minlength=2)
    collision_events += events
    return events
```

生产函数不读取previous positions、relative segment或tau；空集合/单存活机安全。相交但两端分离的轨迹不再产生正式event。collision不改变alive，不增加kill/loss，不终止回合或抑制开火；它也不是防穿越的物理shield，endpoint检测并不保证步间轨迹不穿越。

顺序保持movement→OOB→collision→weapon→termination。真正死因仍只有weapon/OOB。red_collision_events/blue_collision_events从V2.4开始明确表示endpoint unique participant-events，不与历史swept计数混读。

Reward新公式：
```text
R_team = 10*weapon_kills
       - 10*weapon_losses
       - 0.2*endpoint_unique_participant_events
       - 20*OOB
       + terminal(+20/0/-20)
```

同一UAV同一步多个RR/RB pair仍只算一次；多机共享同team reward的原数据链不变。没有potential。

## 5. pytest：当前与历史测试分工

实际执行：pytest -o addopts='' -q（覆盖项目默认addopts以显示清晰汇总）。最终结果107 passed in 5.36s。

当前生产测试覆盖49.9m、恰50m、>50m；RR/BB/RB；Swept-only两机交叉但endpoint分离不计碰撞；multi-pair去重；跨步继续计数；空/单机集合；non-lethal；OOB先排除；同一endpoint碰撞pair仍可完成weapon击杀；每参与者−0.2、两参与者−0.4、episode metrics。

tests/test_collision_audit.py中pair_geometry、Swept-only、historical Current=−5、Liu/Gao映射数学保持历史定义。原test_current_recompute_matches_unmodified_env改名test_v24_production_matches_historical_liu_counterfactual：历史Current仍断言−10，当前生产应匹配Liu-style −0.4。没有把历史Current篡改成endpoint/−0.2。

scripts/collision_audit.py与scripts/behavioral_audit.py保留V2.3历史swept审计断言，不在V2.4上作为当前生产验收运行；旧报告和原始数据不覆盖。

## 6. 100局确定性验收方法

seed200000–200099，configs/base.yaml，双方原RuleBasedOpponent()，没有另写Greedy。

为验证完整轨迹而非仅均值，validate_v2_4.py从冻结提交c33ed9e9953087792893a6c9d2d1330abd694a77只读加载V2.3 env.py，在隔离模块中使用原类；源码SHA256先与V2.3 audit记录核验，旧实例仅恢复其collision_penalty=5。这不是patch当前生产类，不提供正式“兼容模式”；该历史提交不可用或hash不符即停止。

逐step对照：双方动作、positions/headings/speeds/alive、combat kills、weapon losses、OOB、windows、contact、68D观测、60D state、RNG状态、done/truncated/winner及非collision四项reward。旧实例每局summary也与历史audit原文逐项一致。

共23348步，trajectory mismatch=0，100/100局完全一致。没有任何规则轨迹发生变化。随后实际V2.4与旧audit逐seed/逐队核对endpoint counts、全部五项实际reward与Liu-style反事实total，全部通过。

## 7. 轨迹指标完整对比

每格R/B，均为每局均值，胜负以Red为口径。表中轨迹项要求严格一致，collision项则按本轮要求改变。

| 指标 | V2.3 | V2.4 | 结果 |
|---|---:|---:|---|
| W/D/L | 53/6/41 | 53/6/41 | 一致 |
| alive R/B | 2.32/1.87 | 2.32/1.87 | 一致 |
| weapon kills R/B | 2.52/2.01 | 2.52/2.01 | 一致 |
| weapon losses R/B | 2.01/2.52 | 2.01/2.52 | 一致 |
| OOB R/B | 0.67/0.61 | 0.67/0.61 | 一致 |
| attack windows R/B | 2.59/2.06 | 2.59/2.06 | 一致 |
| Radar contact steps R/B | 214.85/214.85 | 214.85/214.85 | 一致 |
| episode length | 233.48 | 233.48 | 一致 |
| weapon combat episode rate | 100% | 100% | 一致 |
| timeout | 45% | 45% | 一致 |
| draw | 6% | 6% | 一致 |
| collision events R/B | 21.73/21.04 swept | 9.43/8.20 endpoint | 预期改变 |
| collision episode rate R/B/combined | 100%/100%/100% | 99%/99%/99% | 检测语义改变 |
| collision reward R/B | −108.65/−105.20 | −1.886/−1.640 | 预期改变 |
| total return R/B | −114.55/−124.90 | −7.786/−21.340 | 预期改变 |

V2.4累计events Red943、Blue820，精确等于旧audit的endpoint参与者累计。碰撞回合率下降不是物理轨迹变化，而是部分回合只有swept近距交会。

死亡审计保持：Red总268=weapon201+OOB67（75%/25%）；Blue总313=weapon252+OOB61（80.5112%/19.4888%）；合计581=453+128（weapon77.9690%、OOB22.0310%）。不把collision算入死亡。

## 8. Reward完整均值与逐局核账

| Component | V2.3 Red | V2.4 Red | V2.3 Blue | V2.4 Blue |
|---|---:|---:|---:|---:|
| kill | 25.20 | 25.20 | 20.10 | 20.10 |
| weapon loss | −20.10 | −20.10 | −25.20 | −25.20 |
| collision | −108.65 | −1.886 | −105.20 | −1.640 |
| OOB | −13.40 | −13.40 | −12.20 | −12.20 |
| terminal | 2.40 | 2.40 | −2.40 | −2.40 |
| total | −114.55 | −7.786 | −124.90 | −21.340 |

不是“差不多”：100/100个seed，两队endpoint计数精确相同；每队每局collision reward/total与历史Liu-style结果误差<1e−9。全体最大绝对差5.684341886080802e−14，来自浮点累加顺序。非collision四项保持不变。

生产episode_summary只存Red return；Blue使用真实每步team_rewards/team_reward_components累计，不把Red return误当Blue。完整逐局证据位于validation.json的episode_checks，含两队actual_team_rewards与最大误差。

旧audit的Liu-style share可直接用于同轨迹V2.4：pooled collision absolute share R/B约2.38%/2.10%，平均episode share约2.40%/2.29%。只说明奖励账本中已成为辅助项，不代表已证明学习性能改善。

## 9. MAPPO smoke

沿用configs/smoke.yaml，不改其参数：seed11、CPU、2个真实worker进程、rollout64、hidden64、epochs2、minibatch32；只覆盖output到新V2.4目录。实际worker PID41184/41185。现有训练默认仍为CUDA/16workers，本轮没有修改。

实际1280 env steps、10 updates、14个完整episode，采样/update正常。所有loss、entropy、KL、梯度与CSV数值有限；Actor参数L2变化0.34711045026779175；final checkpoint保存并通过MAPPO.load加载，Actor/Critic state_dict与文件逐tensor相等；载入global_step=1280。奖励五项求和误差=0。

| PPO metric | Min | Max | Last |
|---|---:|---:|---:|
| actor_loss | -0.0096782569 | 0.018431703 | 0.0019794613 |
| critic_loss | 47.267148 | 232.22484 | 96.521968 |
| entropy | 1.8396943 | 1.8478218 | 1.8478218 |
| approx_kl | 0.000022778980 | 0.00027190878 | 0.000063754040 |
| clip_fraction | 0.0000000 | 0.0000000 | 0.0000000 |
| actor_grad_norm | 0.81693142 | 1.3884232 | 0.89807343 |
| critic_grad_norm | 10.360705 | 28.955240 | 21.082884 |
| explained_variance | -0.043417096 | 0.068383932 | 0.053416908 |

末次汇总14个episode：return −78.7571428571、W/D/L=0/0/14。只作为Reward改变后的采样/学习/存取链路验证，不是策略能力评估，不宣称胜率改善。受新reward影响，短训学习轨迹本就可以不同于V2.3 smoke；严格保持不变的验收对象是固定Greedy双方100局，而非训练中的更新后策略。

## 10. 当前剩余问题与50k建议

- 45% timeout、6% draw、22.03% OOB死亡占比未变，本轮没有处理。
- 原有限转率boundary不能保证不OOB，Greedy仍可能长期直追/绕转；Blue未变。
- endpoint不检测两端都分离的step中途穿越；这次明确采用该语义，不声称物理上阻止穿越。
- collision仍逐步重复计数，没有cooldown；小权重降低主导性，不等于安全问题已消失。
- 当前只完成确定性正确性和短smoke，MAPPO的可学习性、训练稳定性和泛化仍需下一步验证。
- collision reward口径已变，旧checkpoint若继续训练不再是同一实验目标；不要把旧return与新return直接当策略性能改善。

建议下一步经用户检查后进入50k短程验证，不调整其它参数、不升级对手。是否继续正式2M应等50k结果，不自动决定。本轮到此停止，没有50k/2M。

## 11. 输出与复现

runs/v2_4_endpoint_collision/：

- validation.json：100局基础统计、逐step/逐局验收结果、两队实际reward、历史reference及audit hash。
- smoke_validation.json：Actor变化、checkpoint加载、PPO范围、reward accounting。
- scope_validation.json：源码/配置/历史报告前后SHA256。
- smoke/smoke.log、episodes.csv、updates.csv、checkpoints/、tensorboard/：实际短训产物。

复现（WSL Ubuntu / conda uav）：
```bash
pytest -o addopts='' -q
python scripts/validate_v2_4.py --output runs/v2_4_recheck/validation.json --smoke
```

请选择未使用的新输出目录。--smoke只在100局验收成功后运行1280步，任何轨迹或反事实不一致均assert停止，不会继续训练。无需安装新框架，也不改生产类。
