# V2.2 Rear-Pursuit Blue：局部修改与实际验证

日期：2026-10-16。V2.2仅是本次Blue-policy验证编号，环境物理、Reward和MAPPO仍保持V2.1。

结论：指定Blue规则已准确实现，79项测试通过，短接口smoke通过；行为验证不建议进入50k。直接追击Red胜率从100%降至67%，Blue能产生真实窗口/击杀，但规则双方的weapon combat明显退化、碰撞事件明显增加，并有持续转向而难以建立合法几何的情况。本轮未执行50k或2M，也没有为改善结果追加规则或调参。

## 1. 修改文件及冻结核对

实际修改：

- src/uav_mappo_2d/opponents.py：替换交战heading目标，新增rear_point几何函数和必要退化回退；control/boundary_heading/nearest_target保持原样。
- src/uav_mappo_2d/config.py、configs/base.yaml：仅删除旧Blue距离配置；无新增Blue空间尺度。smoke.yaml原本不含该字段，未改动。
- tests/test_opponents.py、tests/test_checkpoints.py：新几何、全距离rear追踪、退化回退及旧配置不存在的验证。
- scripts/sanity_check.py：将第四模式标签明确改为direct-pursuit-vs-rule；Red direct-pursuit函数保持原样。
- scripts/behavioral_audit.py：只读记录每个Blue决策的目标/heading/rear point/ATA/aspect/真实武器窗口、击毁和OOB，并重建碰撞pair诊断；不参与控制。
- README.md、docs/environment_spec.md：当前Blue定义、来源修正、implementation convention和相对文献引用。
- docs/validation_v2_2_rear_pursuit.md：本报告。

经源码与AST核对，env.py、models.py、mappo.py、parallel_env.py、train.py、evaluate.py、rendering.py、smoke.yaml、旧validation_v2_1_5v5.md均与修改前完全一致；Blue.control/boundary_heading/nearest_target及Red direct-pursuit函数的AST相同。证据见runs/v2_2_rear_pursuit/frozen_audit.json和implementation_audit.json。

原validation_v2_1_5v5.md原样保留为历史记录，未覆盖其数据或将旧50k作为新策略训练证据。历史原文中的旧策略名称仅表示旧实验，不能据此理解当前策略。

## 2. 新Blue完整伪代码与公式

```text
actions = zeros(5,2)
detected = env.team_detected_enemy_mask(team)
for each alive UAV i:
    desired_heading = boundary_heading(...)   # 原nominal one-step prediction
    if desired_heading is None:
        j = nearest currently detected enemy (distance, enemy_id tie-break)
        if no j:
            continue                         # [0,0], straight ingress
        rear_point = p_j - attack_range*[cos(h_j),sin(h_j)]
        delta = rear_point-p_i
        desired_heading = h_j if norm(delta)<1e-6 else atan2(delta.y,delta.x)
    omega_norm = clip(wrap(desired_heading-h_i)/(radians(max_turn_rate_degrees)*dt),-1,1)
    actions[i] = [0,omega_norm]
return actions
```

优先级仅Boundary Return > Rear-Aspect Pursuit > Straight Ingress。无速度策略、独立尾距、assignment、target memory、patrol、collision avoidance、左右让行、head-on特判、threat score/FSM或shaping。每架每步独立选最近detected敌机，允许多机同目标。死机动作零。

rear_point=p_target-attack_range*e_target。默认attack_range1000m，目标heading0时p_target=(0,0)得到(-1000,0)，heading=pi/2时得到(0,-1000)。只在norm(delta)<1e-6回退target heading；不是新增战术模式。

边界逻辑不变：当前速度/heading名义预测下一步将越过±5000时，desired=wrap(current_heading+pi)，仍经10°/s有限转率控制，不直接改物理状态，无margin/safe point/中心返回/teleport。Blue仍可能OOB。

## 3. 文献支持原则与实现约定

本轮用PDF技能重新查看项目实际PDF封面与相关整页：
- 孙世彬：《基于多智能体强化学习的大规模无人机集群博弈对抗研究》，东南大学硕士论文，2025，第三章3.4.2，PDF第43页/印刷页35。交战阶段提出尽量保持探测目标，并绕到敌机后方有效攻击。
- 实际项目材料为[基于多智能体强化学习的大规模无人机集群博弈对抗研究_孙世彬 (1).pdf](../基于多智能体强化学习的大规模无人机集群博弈对抗研究_孙世彬%20(1).pdf)。作者为孙世彬，不能混成孙笠晋；后者《基于多智能体强化学习的无人机对抗技术研究》是独立论文，仅保留为冻结二维建模来源之一。
- 未随项目保存的论文在正式规范中只列作者/题名/章节或表格，不创建本机绝对文件路径引用。当前README/environment_spec已清理此类引用。

| Item | 文献支持原则 | 当前实现/implementation convention |
|---|---|---|
| 可探测目标追踪 | 二维空战中基于探测目标进行追踪 | 队级detected集合、每步最近目标/等距小ID，是项目选择 |
| 交战目标 | 孙世彬固定规则交战阶段的绕后攻击思想 | 不照搬其突进/巡航阶段，不增加原文阶段FSM |
| rear-point控制 | 原文未公开本项目所需具体连续低层公式 | p_target-attack_range*heading_vector，完全是implementation convention |
| 距离尺度 | 沿用当前Weapon已有attack_range | 不新增700m或其它尾距配置；1000m不是论文公布的rear距离 |
| 退化点 | 无额外战术依据 | 1e-6数值容差、target heading回退 |
| Boundary Return | 简单返回战场 | 保留V2.1连续有限转率适配，不宣称能保证不越界 |

将攻击距离作为尾后接近点的距离尺度，是本项目为了将“绕到敌机后方”转化为连续控制目标而采用的实现约定，不是参考文献公开的控制公式。

## 4. 冻结的环境与Reward

5v5、dt1s、最多300步、10km地图；初速95–105m/s、速度50–150、加速度±10m/s²、转率±10°/s。Radar3000m确定性圆形、队内共享、无记忆；Actor68D/Critic60D不变。Weapon1000m、ATA±30°、aspect±45°、最近合法目标、同快照同步击毁且victim去重。collision50m swept不致死、每step每机event，仍可当步开火；OOB先死亡后不参与collision/weapon。

Reward仍为+10 kill、-10 weapon loss、-5每机collision event、-20 OOB、terminal±20/0。无potential。Gaussian Actor、float buffer、team GAE、PPO全部不变，正式默认CUDA/16进程未修改。本轮未改变任何这些数值。

## 5. Pytest实际结果

79 passed in 5.06s。覆盖：
- 删除旧字段，未添加尾距/旧策略机制；nearest detected、等距ID、多机可同目标、hidden/no-contact。
- 两个精确rear-point方向、attack_range尺度联动、1500/1000/500m均使用rear point。
- action指向rear point而非目标本体/远离目标，a_norm=0；rear点数值退化回退finite。
- boundary优先级、严格预测边界、未直接修改状态、仍可真实OOB。
- 原冻结环境、Reward、Gaussian、PPO、checkpoint与真实多进程测试继续通过。

## 6. 短smoke（配置接口清理验证）

原smoke配置：1280环境步，CPU/2workers，seed11、64宽网络、10次PPO update，12个完成回合。未修改smoke超参数。PID38890/38891。
Actor参数相对相同seed初始值L2变化0.399702；全部CSV/PPO有限，五分量reward核账误差0；final checkpoint实际重新加载成功。只做链路验证，不把短训练胜率作为策略评价。

训练完成回合W/D/L=0/0/12、return=-111.667、kill R/B=0/1.25、OOB=3.75/.25、collision events=.833/31.583、length103.833。

| PPO指标 | 最后一次update |
|---|---:|
| actor_loss | -0.003205410 |
| critic_loss | 458.8565 |
| entropy | 1.841154 |
| approx_kl | 0.0001514067 |
| clip_fraction | 0.000000 |
| actor_grad_norm | 0.9789449 |
| critic_grad_norm | 44.03042 |
| explained_variance | 0.03678119 |

## 7. 四组各100局sanity完整结果

各模式固定seed200000–200099，与旧V2.1相同；random RNG也相同。rule双方独立新RuleBasedOpponent，random/passive定义不变，direct-pursuit Red函数源码不变（仅模式名从pursuit-vs-rule明确为direct-pursuit-vs-rule）。未调参数、未训练规则。

除局数/比例外均为每回合均值，R/B=Red/Blue，W/D/L从Red视角：

| 指标 | rule-vs-rule | random-vs-rule | passive-vs-rule | direct-pursuit-vs-rule |
|---|---:|---:|---:|---:|
| W/D/L局数 | 8/78/14 | 1/0/99 | 0/0/100 | 67/21/12 |
| return | -282.900 | -118.500 | -128.000 | -119.050 |
| alive R/B | 4.800/4.870 | 0.010/4.920 | 0.000/5.000 | 4.550/2.870 |
| weapon kills R/B | 0.120/0.190 | 0.010/1.030 | 0.000/0.000 | 2.130/0.450 |
| weapon losses R/B | 0.190/0.120 | 1.030/0.010 | 0.000/0.000 | 0.450/2.130 |
| OOB R/B | 0.010/0.010 | 3.960/0.070 | 5.000/0.000 | 0.000/0.000 |
| collision events R/B | 56.160/48.580 | 1.900/17.660 | 1.600/5.740 | 29.370/39.980 |
| attack windows R/B | 0.120/0.240 | 0.010/1.090 | 0.000/0.000 | 2.180/0.460 |
| radar contact steps R/B | 281.880/281.880 | 85.530/85.530 | 69.940/69.940 | 264.590/264.590 |
| episode length | 300.000 | 106.360 | 89.090 | 282.710 |
| kill_reward | 1.200 | 0.100 | 0.000 | 21.300 |
| weapon_loss_reward | -1.900 | -10.300 | 0.000 | -4.500 |
| collision_reward | -280.800 | -9.500 | -8.000 | -146.850 |
| oob_reward | -0.200 | -79.200 | -100.000 | 0.000 |
| terminal_reward | -1.200 | -19.600 | -20.000 | 11.000 |
| collision episode rate red | 100.00% | 55.00% | 54.00% | 100.00% |
| collision episode rate blue | 100.00% | 84.00% | 68.00% | 100.00% |
| collision episode rate combined | 100.00% | 87.00% | 71.00% | 100.00% |
| weapon combat episode rate | 25.00% | 72.00% | 0.00% | 87.00% |

真正death_total仅weapon_loss+OOB；碰撞是事件，不进入分母。零死亡时占比定义0。

| 模式 | 范围 | death total | weapon | OOB | weapon death share | OOB share |
|---|---|---:|---:|---:|---:|---:|
| rule-vs-rule | red | 20 | 19 | 1 | 95.00% | 5.00% |
| rule-vs-rule | blue | 13 | 12 | 1 | 92.31% | 7.69% |
| rule-vs-rule | combined | 33 | 31 | 2 | 93.94% | 6.06% |
| random-vs-rule | red | 499 | 103 | 396 | 20.64% | 79.36% |
| random-vs-rule | blue | 8 | 1 | 7 | 12.50% | 87.50% |
| random-vs-rule | combined | 507 | 104 | 403 | 20.51% | 79.49% |
| passive-vs-rule | red | 500 | 0 | 500 | 0.00% | 100.00% |
| passive-vs-rule | blue | 0 | 0 | 0 | 0.00% | 0.00% |
| passive-vs-rule | combined | 500 | 0 | 500 | 0.00% | 100.00% |
| direct-pursuit-vs-rule | red | 45 | 45 | 0 | 100.00% | 0.00% |
| direct-pursuit-vs-rule | blue | 213 | 213 | 0 | 100.00% | 0.00% |
| direct-pursuit-vs-rule | combined | 258 | 258 | 0 | 100.00% | 0.00% |

## 8. 与旧V2.1对比（历史策略 vs 当前策略）

下面的旧策略名称仅表示历史实验，不描述当前Blue。使用相同100个初始seed；旧数据直接读取runs/v2_1_validation/sanity.json，未覆盖历史验证文档。

| 指标 | rule旧V2.1 | rule新Rear | direct-pursuit旧V2.1 | direct-pursuit新Rear |
|---|---:|---:|---:|---:|
| W/D/L局数 | 54/0/46 | 8/78/14 | 100/0/0 | 67/21/12 |
| return | -76.900 | -282.900 | 56.150 | -119.050 |
| alive R/B | 1.810/1.580 | 4.800/4.870 | 4.920/0.000 | 4.550/2.870 |
| weapon kills R/B | 2.020/1.680 | 0.120/0.190 | 5.000/0.080 | 2.130/0.450 |
| weapon losses R/B | 1.680/2.020 | 0.190/0.120 | 0.080/5.000 | 0.450/2.130 |
| OOB R/B | 1.510/1.400 | 0.010/0.010 | 0.000/0.000 | 0.000/0.000 |
| collision events R/B | 10.340/7.790 | 56.160/48.580 | 2.610/1.300 | 29.370/39.980 |
| attack windows R/B | 2.020/1.680 | 0.120/0.240 | 5.270/0.080 | 2.180/0.460 |
| radar contact R/B | 191.030/191.030 | 281.880/281.880 | 45.040/45.040 | 264.590/264.590 |
| episode length | 213.720 | 300.000 | 63.160 | 282.710 |
| collision episode rate合并 | 95.00% | 100.00% | 71.00% | 100.00% |
| weapon death share合并 | 55.98% | 93.94% | 100.00% | 100.00% |
| weapon combat episode rate | 100.00% | 25.00% | 100.00% | 87.00% |

direct-pursuit重点：Red胜率100%→67%，kill5.00→2.13；Blue kill .08→.45、window .08→.46，说明不再是机械式全胜。但新Blue并未整体通过：规则双方合计kill3.70→.31/局、combat回合100%→25%、平局0→78%，而且规则双方100局全部跑到300步。

碰撞恶化不是胜率能掩盖的：rule新collision R/B=56.16/48.58，旧10.34/7.79；direct新29.37/39.98，旧2.61/1.30。Blue OOB没有恶化（rule1.40→.01，direct仍0），但减少OOB不能抵消交战停滞和碰撞重复惩罚。

## 9. 各20局只读轨迹审计

两组均seed200000–200019，是100局sanity子集。逐个Blue活机/step记录CSV，而非只抽取成功案例。字段包含target id/distance、Blue和target heading、rear point/rear distance、ATA/aspect前后、合法窗口、实际自动武器目标及其几何、victim确认、OOB与weapon loss。所有记录a_norm=0。

| 指标 | rule-vs-rule 20局 | direct-pursuit-vs-rule 20局 |
|---|---:|---:|
| straight agent-steps | 1955 | 1955 |
| rear-pursuit agent-steps | 27572 | 19965 |
| 目标ID变化次数 | 1703 | 1364 |
| 最长连续同目标rear-pursuit步数 | 279 | 261 |
| 最长活机无任何武器窗口步数 | 300 | 300 |
| ATA绝对值改善步数 | 15201 | 11546 |
| aspect绝对值改善步数 | 13511 | 8914 |
| 两角同时改善步数 | 2749 | 1667 |
| 所选目标形成窗口步数 | 1 | 1 |
| 所有合法pair窗口数 | 5 | 6 |
| Blue真实weapon kills | 4 | 6 |
| Blue OOB deaths | 0 | 0 |
| Blue collision events | 1435 | 751 |
| 同队collision pairs | 865 | 264 |
| 敌我collision pairs | 277 | 410 |
| 有Blue kill/窗口的回合 | 4/4 | 5/5 |

collision pair仅诊断，不作为reward；每step重建pair后对参与机去重，断言与环境collision_events增量完全一致。rule子集中同队865对、敌我277对，说明多机同目标/相近机动造成的同队重复接近是主要来源；direct子集敌我410对、同队264对。pair累计是step-pair次数，不是不同物理pair的数量。

后半程step_before≥150的rear-pursuit决策状态：

| 指标 | rule | direct-pursuit |
|---|---:|---:|
| 记录数 | 14630 | 9031 |
| ATA超过30°比例 | 92.96% | 94.91% |
| aspect超过45°比例 | 79.90% | 83.41% |
| 目标距离超过1000m比例 | 52.00% | 41.35% |
| 最大转率饱和比例 | 83.64% | 95.32% |
| 对任意目标实际开火步数 | 3 | 1 |
| 到rear point距离中位数(m) | 1304.768 | 1286.242 |

这两组后半程对各自“当前所选目标”的window均为0；对任意其它合法目标自动开火分别有3/1个step。不能把“指向rear point”直接等同于“ATA/aspect已经合适”。

## 10. 具体轨迹、窗口与击杀例证

1. rule seed200000 step19前B2选择detected R3，目标距3159.770m（可由队友共享，非越界读隐藏目标）；rear point=(-2327.236,-1647.016)，距rear4158.824m。动作[0,-1]，本步后ATA -11.139°→-1.612°、aspect177.140°→166.667°。动作确实朝rear point并局部改善几何，但离武器合法窗口仍远。
2. rule seed200010，第88步B4所选R0：距离992.985→978.050m，ATA -40.866°→-29.956°，aspect -32.806°→-41.896°，最终满足全部几何并击毁R0。这证明存在完整链路，但该20局对所选目标的此类窗口仅1个step。
3. rule seed200005第203步B3虽然选R2进行机动，自动武器实际击毁R0：944.992m、ATA -11.256°、aspect41.387°。Weapon仍按所有合法敌机选最近者，不要求等于Blue规则选定的机动目标；审计将两种target及其角度分开记录，未错误用R2的不合法角度解释R0击杀。
4. direct-pursuit seed200001第133步B3自动击毁R1：998.914m、ATA -13.067°、aspect25.904°。其机动target为R0，与自动开火target不同，同样是冻结Weapon语义。
5. 子集Blue OOB为0；全100局rule只1架Blue OOB，其余重点组为0。没有通过重设计boundary降低OOB；boundary函数AST与旧版相同，差异来自新的交战轨迹。

## 11. 行为解释与仍存在的问题

- 当前规则解决了旧近距直接背离与rear-aspect武器的结构性不匹配之一，direct-pursuit不再100/100取胜。但不能由此断言新Blue整体有效。
- rear point是移动的几何目标，当前位置/heading随目标运动不断变化；以恒速、有限转率单纯朝这个点转向，不保证自身机头同时对准目标，也不保证进入1000m和45°aspect。未增加速度策略或heading匹配层，因而没有额外机制保证收敛。
- 观测到了长达279/261步对同一目标持续rear-pursuit、后半程高比例最大转率、坏ATA/aspect和几乎无有效窗口，符合持续绕转/相对机动而难以形成攻击的退化。不能据有限数据证明rear point在数学上不可达，但不能忽略实际长期无窗口。
- 多机允许选同一最近敌机，会汇聚于相似接近点；同队碰撞pair占rule诊断多数。碰撞不致死，所以可能连续许多步计事件；诊断核账显示不是去重bug。
- rule Red collision reward -280.8/局，占return -282.9的大部分；direct Red虽67胜/100局，return仍-119.05，因为collision reward -146.85。return不能直接当胜率。
- 高weapon death share不等于交战充分：rule占比93.94%，但总weapon deaths仅31架/100局、实际weapon combat仅25局，是低死亡分母下的比例。
- passive对规则100局无weapon kill，全靠Red OOB决定胜负；这同样说明新规则并非在所有基本对照中都能有效建立攻击窗口。

## 12. 是否建议进入50k

不建议。虽然Blue存在真实kill且direct-pursuit机械全胜有所缓解，但规则双方武器交战显著稀疏、78%平局、碰撞事件增幅明显，并存在持续无窗口。未满足“weapon combat正常且OOB/collision无新灾难性问题”的综合条件。

按要求，本轮只做pytest、1280步接口smoke、4×100 sanity与各20局只读审计；没有自动运行50k或正式2M。不修改rear距离、Weapon角度/距离、turn rate、speed、Reward、collision或boundary，也没有进一步添加战术/工程规则。后续采用何种固定Blue基准应由用户决定，当前不把失败解释为需要立即加复杂控制层。

## 13. 原始证据

均位于runs/v2_2_rear_pursuit/：

- pytest.log：79测试。
- smoke.log、smoke/checkpoints/final.pt、smoke/episodes.csv、updates.csv、tensorboard/。
- sanity.json：4×100完整统计。
- behavioral_audit.json：各20局summary、动作/几何/击毁例子及collision诊断。
- rule-vs-rule_blue_steps.csv（29527行）、direct-pursuit-vs-rule_blue_steps.csv（21920行）：全部Blue活机决策时刻。
- trajectory_summary.json：角度/距离分布、后半程饱和/窗口统计和样本。
- implementation_audit.json、frozen_audit.json：短smoke、checkpoint加载及冻结代码一致性。

历史比较源为runs/v2_1_validation/sanity.json；历史validation_v2_1_5v5.md内容核对完全相同，未覆盖。正式文献引用无本机绝对路径；运行脚本路径不作为论文引用。
