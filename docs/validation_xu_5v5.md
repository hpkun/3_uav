# Xu-5v5基准重构验收报告

代码按本轮明确要求完成，完整测试、smoke、三组各100局规则sanity和50k Course1均已实际执行。数值与接口验收通过，但行为验收有明确缺口：规则sanity和50k短训练均未产生武器击杀，Strategy3仍以碰撞损失为主。不能将本报告解读为“已恢复论文作战效果”或“已验证可学习攻击”。

此前continuous 5v5的500k结果、旧3v3训练结果属于不同环境，不作为Xu baseline的有效训练结果，checkpoint不兼容且未迁移。未运行正式2M/2M/6M课程。

## 1. 修改文件

- src/uav_mappo_2d/config.py：Xu参数、任务奖励、blue_strategy，删除旧continuous/OOB/potential配置。
- env.py：离散动作、旧状态位移、2km雷达、600m武器、避碰mask、七分量reward，无OOB死亡。
- models.py、mappo.py：masked Categorical Actor、int64 rollout动作、采样mask存储/重算；其余PPO逻辑保留。
- opponents.py：重建Strategy1/2/3，删除tail pursuit、全局最优匹配和边界恢复。
- parallel_env.py：整数(E,5)动作，(E,5,10)mask，课程蓝方和terminal reset数据。
- train.py、evaluate.py：掩码传递、奖励/碰撞/出界诊断、--init-checkpoint和策略CLI。
- rendering.py：仅裁剪绘图像素，不改变物理位置。
- configs/base.yaml、smoke.yaml，新增xu_course1.yaml、xu_course2.yaml、xu_course3.yaml。
- scripts/sanity_check.py：三种对称规则sanity，mask合法性和完整reward JSON。
- 重建tests/test_environment.py、test_opponents.py、test_mappo.py、test_parallel_env.py；新增conftest.py、test_rewards.py、test_curriculum.py；删除旧语义test_radar_collision_reward.py。
- README.md、docs/environment_spec.md、pyproject.toml更新；旧validation_fixed_5v5.md标记为历史记录；新增本报告。

## 2. 参数、动作和shape

| 项目 | 最终值 |
|---|---|
| 场景 | 固定5v5，二维同高度同质UAV |
| 名义地图 | 10000m×10000m，中心0，坐标±5000m |
| 回合 | 300steps；全灭可提前终止 |
| 初始/最小/最大速度 | 100 / 50 / 150m/s |
| 初始heading | 红0/蓝pi，无heading jitter |
| 出生project convention | x=±3500m；y=[-3000,-1500,0,1500,3000]；位置各坐标独立±200m |
| dt project convention | 1s，论文未给出具体数值 |
| 雷达 / 武器 | 2000 / 600m |
| 前向/尾后半角 | 15° / 50° |
| 碰撞 / buffer / collision辅助距离 | 150 / 450 / 600m |
| Actor/Critic | 68–128–128–10 / 60–128–128–1 |
| 全env obs/state/action/reward | (2,5,68) / (60,) / (2,5)整数 / (2,5) |
| red active/action masks | (5,) / (5,10)bool |
| parallel obs/state/action/action mask | (E,5,68) / (E,60) / (E,5) / (E,5,10) |
| rollout actions/masks | (T,E,5)int64 / (T,E,5,10)bool |

10-action mapping：

| action | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| Δv m/s | -1 | -.5 | 0 | .5 | 1 | 0 | 0 | 0 | 0 | 0 |
| Δheading deg | 0 | 0 | 0 | 0 | 0 | -5 | -2 | 0 | 2 | 5 |

2/7两个物理no-op保留。先用v_t/heading_t移动，再按动作改变下一状态，速度clip，heading wrap。动作非法或违反mask时直接报错。

## 3. 雷达、武器、碰撞和mask

存活敌机距任一活队友≤2000m时全队共享；无记忆，离开覆盖立即隐藏。Actor隐藏slot为0，友机始终已知，Critic全局状态不隐藏。68D/60D数值编码是项目约定，不是论文精确公布维度。

weapon条件：d≤600、abs(alpha)≤15°、abs(beta)≤50°。自动选择最近合法目标，同距按ID。同步命中、多攻击者同目标去重；尾追可攻击，迎头不可；collision victim不可当步攻击。

swept collision求两条一步相对线性轨迹的连续最小距离，≤150m则双方损毁；同队/跨队有效，多pair去重。该算法是数值implementation detail。

buffer触发只看最近活机：>450m十动作全开放；≤450m且alpha≥约0只开放{5,6,7}；alpha<0只开放{7,8,9}；接近0固定右转。dead只开放2且active=0。规则选择最大角动作5/9。Categorical invalid probability为0，deterministic argmax和PPO重算均使用mask。buffer保存采样时mask，而非next-state mask；worker自动reset后供下次动作的数据全部来自reset。

## 4. 完整reward

```text
R_team = kill_reward + own_loss_reward + terminal_reward + timeout_reward
       + distance_reward + attack_angle_reward + collision_avoidance_reward
kill_reward = 20 * weapon_kills
own_loss_reward = -20 * (own_weapon_deaths + own_collision_deaths)
terminal_reward = 消灭敌方:+100 / 己方全灭:-100 / 同时全灭:0
timeout_reward = 双方尚存且300步时：数量优势+80 / 劣势-80 / 相等0
distance_reward = mean((d_prev_nearest_detected - d_curr_nearest_detected)/1000)
                  前后均有contact且d_prev>1500m，否则该agent为0
attack_angle_reward = mean(theta_min_prev - theta_min_curr)，radians
                      前后均有contact，否则该agent为0
collision_avoidance_reward = mean((d_curr_nearest - d_prev_same_aircraft)/1000)
                            当前最近活机距离<600m，否则该agent为0
```

辅助mean的分母为本队最终存活agent数，不满足条件的存活agent贡献0。本队无存活则辅助0。该聚合和每架自身损失都记team−20为shared-team adaptation。任务项相加，例如最后武器击杀产生kill+20和terminal+100。env正向尾追测试实际得到红+120、蓝−120。没有OOB/potential/旧±1武器分量。

## 5. Blue策略和课程

Strategy1保持初始方向和100m/s，避碰后逐步回正。Strategy2选择最近detected敌机并直接追当前位置，无contact回初始方向。Strategy3最近优先，目标容量min(alive_friends,2+dead_enemies)，超容量保留最近追击者，较远者按next-nearest重试；无可选敌机朝最远活队友。所有策略均优先避碰，所有tie确定性处理，没有tail point、permutation最优分配或boundary recovery。

课程用法：

```bash
uav2d-train --config configs/xu_course1.yaml
uav2d-train --config configs/xu_course2.yaml \
  --init-checkpoint runs/xu_course1_seed7/checkpoints/final.pt
uav2d-train --config configs/xu_course3.yaml \
  --init-checkpoint runs/xu_course2_seed7/checkpoints/final.pt
```

预算2M/2M/6M，默认CUDA、16环境、rollout128。--init-checkpoint只加载Actor/Critic，重置optimizer/global_step/episode/best；--resume恢复这些断点字段。两者互斥，测试验证权重和Adam状态区别。

## 6. 论文对齐边界

完整Item / Xu thesis / Implementation / Status / Notes表见environment_spec.md，覆盖场景、速度、动作、雷达、武器、回合、碰撞、buffer、奖励、策略、课程、观测、dt、spawn、swept和OOB。

论文明确的绝对尺度已实现；dt=1、精确spawn、68D/60D、同步结算、最近目标和各tie为约定。individual辅助奖励取team mean、自身损失team penalty为接口适配。PPO现有超参数和MLP不等于论文含GRU/attention/信用分配的完整方法。

原文第30页图3.5以区域及其重合描述buffer，附件则明确要求两机中心距≤450m触发。本实现按附件执行，不能把这两种几何解释不加说明地称为完全一致。原文左右转措辞与正负角有不一致，本实现采用任务明确给出的signed alpha规则。

## 7. Pytest与smoke

实际命令均经WSL Ubuntu进入conda uav：

```bash
pytest -q
uav2d-train --config configs/smoke.yaml
uav2d-evaluate --config configs/smoke.yaml \
  --checkpoint runs/smoke_xu_5v5/checkpoints/final.pt --episodes 20 --seed-start 300000
```

81项测试全部通过。覆盖场景、10动作及两个no-op、旧状态位移、雷达边界/共享/消失、全局状态、武器角度与距离边界、同步击杀/去重、swept/多pair/同队跨队碰撞、mask边界方向/最近机/死亡动作、categorical概率/采样/argmax/重算mask、奖励和分量核对、无OOB死亡、三策略、三课程、独立进程/reset、PPO有限值和checkpoint加载。

Smoke：1280env steps、2worker（33605/33606）、CPU、10次PPO update，checkpoint保存成功，后续评估加载成功。完整训练回合4个：0胜1平3负，return−136.418，平均长度244。最后actor_loss=.0069、critic_loss=201.879、entropy=2.2116、EV=.007，全部finite。

Smoke确定性20局：0胜20平0负；红蓝alive=2.3/2.3，weapon kill=0/0，collision=2.7/2.7，outside=2.3/2.3，长度260.45，return−55.081611。

## 8. 三种规则各100局sanity

seeds=200000–200099，均双方相同策略且各自独立controller。原始完整结果在runs/xu_validation/rule_sanity.json。

| 策略 | 红胜/平/蓝胜 | 红蓝alive | 红蓝weapon kill | 红蓝collision | 红蓝当前outside | mean length | mean red return |
|---|---|---|---|---|---|---|---|
| S1 vs S1 | 0/100/0 | 2.09/2.09 | 0/0 | 2.91/2.91 | 2.09/2.09 | 286.78 | -59.488026 |
| S2 vs S2 | 0/100/0 | 0/0 | 0/0 | 5.00/5.00 | 0/0 | 35.71 | -98.747936 |
| S3 vs S3 | 25/44/31 | 1.01/1.13 | 0/0 | 3.99/3.87 | .32/.30 | 143.55 | -85.609907 |

S1、S2、S3 buffer-agent steps分别2470、1474、4765，合法动作数分别也为2470、1474、4765，即全部符合约束。真实weapon kill在三组sanity均未出现。

| 红方累计reward均值 | S1 | S2 | S3 |
|---|---|---|---|
| kill | 0 | 0 | 0 |
| own loss | -58.2 | -100 | -79.8 |
| terminal | 0 | 0 | -6.0 |
| timeout | 0 | 0 | 0 |
| distance | .582245 | 1.435947 | 1.053047 |
| attack angle | -1.576460 | .448726 | -.071700 |
| collision avoidance | -.293811 | -.632610 | -.791254 |

## 9. 碰撞主导的逐步复核

Strategy3仍collision dominant，已按要求先核对450m约束，未调整dc/de。seed200000首次碰撞发生在第23步，是蓝方两机（flattened IDs6和8）相撞：

| step | 最近距离m（动作前） | 允许动作 | 规则动作 |
|---|---|---|---|
| 20 | 887.084 | 全10个 | 2 |
| 21 | 687.138 | 全10个 | 2 |
| 22 | 487.230 | 全10个 | 2 |
| 23 | 287.439 | {7,8,9} | 9（+5°） |

第23步mask正确收缩，规则正确选最大转向；但按(3.1)本步仍用旧heading位移，转向只影响下一状态，两机在本步进入150m碰撞区。原始记录在runs/xu_validation/avoidance_trace.json。

这说明“约束正确执行”不等于“dt=1和450m阈值足以安全避碰”。论文没有公开dt，且区域重合到中心距离的解释也需谨慎。按照本轮明确定义，不能通过擅自改dt、增大buffer或改为新heading移动来掩盖结果。

## 10. Course1 50k可学习性短验证

```bash
uav2d-train --config configs/xu_course1.yaml --steps 50000 \
  --output runs/xu_course1_learnability_seed7
```

实际CUDA、16独立worker，完成50000env steps和25次PPO update；checkpoint global_step核验为50000。TensorBoard首末记录约29.49s（不含完整启动开销）。原始日志runs/xu_validation/learnability.log，完整汇总learnability_summary.json。

| 指标 | 全部160个完整回合 | 最近100回合 |
|---|---|---|
| 红胜/平/红负 | 0/150/10 | 0/96/4 |
| 红蓝alive | 4.05/4.175 | 4.14/4.22 |
| 红蓝weapon kill | 0/0 | 0/0 |
| 红蓝collision | .95/.825 | .86/.78 |
| 红蓝outside | 4.05/4.175 | 4.14/4.22 |
| mean length | 300 | 300 |
| mean red return | -26.038255 | -22.536583 |
| kill reward | 0 | 0 |
| own loss reward | -19 | -17.2 |
| terminal reward | 0 | 0 |
| timeout reward | -5 | -3.2 |
| distance reward | -.252733 | -.393385 |
| attack angle reward | -1.614282 | -1.593182 |
| collision avoidance reward | -.171240 | -.150016 |

4个完整回合return为正，但全部训练回合双方weapon kill总数均为0。Actor相对同seed初始化的权重L2变化3.370633，说明网络实际更新，并非未训练。25次update全部PPO指标finite：

| 最后一次update | 值 |
|---|---|
| actor loss | -.004815 |
| critic loss | .001692 |
| entropy | 2.227623 |
| approximate KL | .004459 |
| clip fraction | .045704 |
| explained variance | .889292 |

最后partial rollout可能不含终止奖励，所以不能凭最后critic loss较小就判断任务已学会。20局独立确定性评估（seeds400000–400019）：0胜20平0负，alive3.9/3.9，weapon0/0，collision1.1/1.1，outside3.8/3.9，长度300，return−30.301206。

## 11. 验收结论与仍存在的偏差

通过：所有规定接口、离散PPO更新、mask一致性、保存/加载、权重初始化与resume区分、规则/奖励专项测试和规定的实际运行。武器在标准尾追构造中能产生真实击毁与+20/+100，几何结算不是关闭状态。

未获行为证据：三种规则sanity和50k训练均没有weapon kill；Strategy3碰撞仍占全部损失，尚未恢复预期的基本作战过程。不能据短训练胜率评价算法性能，也不能声称学会攻击。实现忠实于本轮参数/时序要求，但行为层面的目标尚未验证成功。

已知局限：dt=1的运动时序与450m中心阈值不保证有效避碰；出生精确分布未公开；无OOB死亡时直线离场后双方可能长时间无contact，辅助信号消失，300步平局较多；shared-team聚合不同于原文individual reward；原文奖励分情况到相加式实现有明确约定；当前MLP/PPO配置不等于论文完整方法。上述均保留并公开记录，没有调参修饰sanity结果。
