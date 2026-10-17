# V2.5：邓岩速度范围与基础概率武器模型

V2.5把当前二维连续5v5环境的速度与基础武器模型对齐到本次提供的邓岩《基于深度强化学习的无人机集群空战决策研究》明确参数。它不是论文环境的逐行复现。这里依据本次提供的论文设计实施，没有新增未经核对的论文引文或lambda/gamma数值。

## A. 论文明确参数

| 字段 | 正式值 |
|---|---:|
| speed_min | 200 m/s |
| speed_max | 400 m/s |
| attack_range | 1000 m |
| attack_angle_degrees | 90°，本次明确按半角解释，inclusive |
| hit_probability | 0.70 |
| radar_range | 3000 m |

EnvironmentConfig默认值与`configs/base.yaml`一致，完整参数见[environment_spec.md](environment_spec.md)。

## B. 项目实现约定

| 项目 | 当前实现 |
|---|---|
| 初始速度 | 每架严格300 m/s，initial_speed_min=initial_speed_max=300 |
| dt / 最大加速度 / 最大转率 | 1s / 10 m/s² / 10°/s，保持原值 |
| 地图 | 10000×10000m，坐标±5000m |
| 出生 | 红x[-3800,-3200]、蓝x[3200,3800]；两队独立采样；y[-3000,3000]，同队最小y间隔500m |
| 初始航向 | 红围绕0、蓝围绕pi，独立±5°扰动 |
| episode | 最长300 step，原终止/胜负语义 |
| 自动攻击 | 每attacker每step最多选择一个合法目标、尝试一次 |
| 目标选择 | 合法目标中最小(distance,target ID)，与Greedy移动目标选择独立 |

论文未给出可确认的二维具体初速，300仅因为是200–400的中点而被选用，不是论文初速。论文运动模型以调整系数控制速度与航向变化，没有可确认的lambda/gamma具体值，因此没有改加速度、转率或dt。

```text
v_next = clip(v + action_accel * 10 * dt, 200, 400)
h_next = wrap(h + action_turn * radians(10) * dt)
p_next = p + v_next * dt * [cos(h_next), sin(h_next)]
```

Actor绝对速度和global state继续使用`2*(v-speed_min)/(speed_max-speed_min)-1`，200/300/400分别映射-1/0/+1；相对速度分母自动变成200。OBS_DIM=68、STATE_DIM=60、ACTION_DIM=2保持不变。

## 攻击资格、随机数与同步结算

```text
alpha = wrap_angle(atan2(target_y-own_y, target_x-own_x) - attacker_heading)
valid = attacker_alive AND target_alive
        AND distance <= attack_range
        AND abs(alpha) <= radians(attack_angle_degrees)
hit = env.np_random.random() < hit_probability
```

±90°是inclusive半角（保留1e-12弧度数值边界容差），迎头几何也可合法。目标自身航向不影响资格；`_compute_pair_geometry`仍返回distance/alpha/beta，beta仍用于观测与诊断，但`_attack_valid`不读取beta。

删除正式配置的`attack_ata_degrees`与`attack_aspect_degrees`，新增`attack_angle_degrees`和`hit_probability`，没有隐式兼容旧gate。校验0<speed_min<speed_max、初速有序且在速度范围内、attack_range>0、0<angle≤180、0≤probability≤1。

武器随机试验仅来自Gymnasium环境自己的`np_random`，不用全局random/np.random。没有合法目标就不抽攻击随机数；有目标每个attacker一次，miss不改选第二个目标。相同reset seed、初始状态和动作序列严格重现攻击结果。

step顺序保持action validation → kinematics → OOB → endpoint non-lethal collision → attack → step_count++ → termination/winner → reward/info。OOB机排除；碰撞参与者可同step攻击。按team 0/1和升序attacker ID遍历同一个post-OOB/post-collision/pre-weapon快照，先完成全部试验、收集victims集合，再同步删除。即使目标已被前一个攻击者命中，后续攻击者仍执行自己的试验；支持mutual kill。同目标多次命中只产生一次death、一次combat kill和一次weapon loss。

`attack_window_count`表示该快照所有满足距离与alpha条件的attacker-target pair数量。它是几何机会，不是尝试、命中或击杀次数；miss也计window。不能用window/kill或kill/window估计命中率，多目标窗口和重复命中去重都会破坏这种解释。

## C. 暂未采用的论文参数或设计

- **ammo=6**：论文未给出当前自动攻击接口下的fire action、firing timing、ammunition consumption timing、cooldown/reload和miss后的再次发射规则。没有依据时不创造这些规则，后续单独设计；本轮不添加弹药状态或动作。
- **radar angle=90°**：仍为V2.4的360°range-only探测、team_detected_enemy_mask与队内即时共享。仅radar_range=3000与论文一致；探测角尚未对齐。探测可用性、Greedy获取目标及观测mask是独立语义，后续单独处理。

## 冻结内容与工具

Reward数值及归属完全不变：kill +10、weapon loss -10、OOB -20、collision participant-event -0.2、terminal ±20/0。只有实际weapon death产生kill/loss反馈；miss无penalty，无attack/window reward或任何shaping。轨迹改变会改变事件次数，但不改变奖励定义。

RuleBasedOpponent源文件未改：nearest detected target、direct pursuit、acceleration=0，boundary return和有限转率control也不改。Greedy默认自然以300m/s恒速飞行。MAPPO/MADDPG/OU/Actor/Critic与正式超参数未调，地图、spawn、collision、episode长度未调。

`behavioral_audit.py`重建几何window与最近目标的攻击尝试；目标weapon death只从真实alive变化确认，成功攻击者ID未知，不猜测。`replay_html.py`从状态变化重建死亡并与info核对；HTML按配置显示攻击扇区。`sanity_check.py`与`validate_v2_5.py`使用真实环境计数。V2.3 collision审计与V2.4历史验证入口明确提示切换历史checkout，不在当前配置运行旧断言；历史数学测试保留。

MADDPG继续保存实际EnvironmentConfig的checkpoint/config_snapshot。MAPPO保留既有权重/optimizer checkpoint格式（原格式不嵌入环境配置）；本次smoke另存smoke_config_snapshot.json，评估/回放显式用V2.5配置。旧权重形状相容不代表历史实验属于V2.5。

验证结果见[validation_v2_5_deng_combat.md](validation_v2_5_deng_combat.md)。V2.3/V2.4历史报告、运行数据和实验数字保留原样。本轮不启动正式RL训练。
