# V2.3 只读 Collision Audit

## 1. 结论与范围

本轮未修改正式环境、Reward、Greedy Blue、MAPPO/PPO、配置或训练参数；未训练、未调参、未运行random/passive、50k或2M。只新增外部审计脚本、数学测试、本报告及runs内诊断数据。没有monkey patch，没有把burst/集合等审计状态传入环境或动作。

主要发现：

- 2173个swept pair-step中，Endpoint 892（41.05%），Swept-only 1281（58.95%）。
- RB占71.88%，同队RR+BB占28.12%；高碰撞不主要来自同队。
- 4277个participant-events对应2502个逐UAV连续burst。2–3步burst贡献56.25%的events，≥4步贡献14.40%，不是仅少数超长burst造成全局高均值。
- Current碰撞绝对分量pooled占比Red58.37%、Blue57.96%，为整体回报分量中的主导项；极端同队长尾也确实存在。
- 仅固定轨迹重算：Liu-style mean return R/B=−7.786/−21.340，Gao-base=−28.900/−36.550；Current=−114.550/−124.900。这不预测改奖励后训练的策略效果，也不自动构成方案推荐。

下一步优先明确“敌我高速近距交会是否属于训练中要惩罚的collision”，然后单独验证检测/阵营口径，保持−5不动以隔离因果；之后再独立讨论系数尺度。不优先用cooldown解释全部问题：短burst和间隔多步的反复交会才是总体主要部分。本轮不实际修改。

## 2. 新增文件、执行和不可变性

新增：

- scripts/collision_audit.py：外部几何、去重集合、burst、三套离线reward和交叉核账。
- tests/test_collision_audit.py：只测审计数学及其与未修改环境的一致性。
- docs/collision_audit_v2_3.md：本报告。

实际执行于WSL Ubuntu、conda uav、项目目录：
```bash
pytest -o addopts='' -q
python scripts/collision_audit.py
```

pytest：99 passed in 6.42s（原84项+新增15项）。包含Endpoint、Swept-only、无碰撞、RR/BB/RB、participant去重、same/cross重叠、Endpoint/Swept-only participant重叠、burst、OOB排除、武器受害者保留、三套重算、零分母、输入不被修改。

所有src/uav_mappo_2d/*.py及configs/*.yaml执行前后SHA256完全一致；当前Git差异只新增上述审计文件。JSON保存全部前后hash。没有改env.py/config.py/opponents.py/models.py/mappo.py/train.py/evaluate.py/parallel_env.py/base.yaml，也没有改原sanity或behavioral audit。

## 3. 基础复现与关键核验

两次运行均只用当前RuleBasedOpponent双方、configs/base.yaml、同一100个seed200000–200099。先原始rollout建立独立基准并核对旧sanity，再对同100seed执行外部审计；不是200个独立样本。每回合逐步hash动作、位置、航向、速度、alive、完整info（含奖励/终局统计）及RNG状态。100/100回合完整hash一致，100/100终局summary一致。旧sanity只保存100局聚合，全部聚合项精确匹配；另与旧behavioral_audit前20局summary逐项核对20/20一致。

| 指标 | 复现结果 |
|---|---:|
| W/D/L（Red口径） | 53/6/41 |
| weapon kills R/B | 2.52/2.01 |
| weapon losses R/B | 2.01/2.52 |
| OOB R/B | 0.67/0.61 |
| collision participant-events R/B | 21.73/21.04 |
| alive R/B | 2.32/1.87 |
| attack windows R/B | 2.59/2.06 |
| Radar contact steps R/B | 214.85/214.85 |
| episode length | 233.48 |
| weapon combat episode rate | 100% |
| timeout / draw | 45% / 6% |
| current return R/B | −114.55/−124.90 |

核验结果：

- audited_steps=23348；collision increment mismatch=0。
- 逐步unique involved UAV数与env.collision_events增量完全一致。
- 23348步pair互斥分类核账全部通过。
- Current collision reward和total return最大绝对核账误差=0。
- 正式summary只累计Red reward；Blue以每步实际team_reward_components及team_rewards累加交叉核验，未臆造Blue episode_summary字段。

OOB顺序严格按movement→OOB→collision→weapon。valid=old_alive AND abs(postmovement_position)<=map_size/2；不是step结束后的alive。128个本步OOB死亡者被排除；453个本步随后被武器击毁者在pre-weapon mask中被保留（不代表它们全都有collision）。每步断言OOB增量及post-step alive是该mask子集。测试也单独覆盖“本步碰撞后被武器击毁”场景。

## 4. 几何与计数定义

逐pair复现当前r0、相对dr、denominator>1e−12、tau裁剪[0,1]、线段相对距离判定。和env一致使用平方距离<=50²，输出距离为其平方根。有效UAV只取post-OOB/pre-weapon mask。

pair count是pair-step：同一实体pair跨多步会多次出现，不等于独立事故/独立相遇数。本批2173个pair-step对应745个不同的episode+实体pair，但同一实体pair仍可多次分离后重遇，745也不是独立物理事故数。

Endpoint与Swept-only在pair层面互斥；participant层面不互斥：同一UAV可以同一步参与一个Endpoint pair和另一个Swept-only pair。same/cross participant也同理。当前reward按每步unique involved UAV计罚，不能按pair数乘−5。

## 5. Pair分类与阵营二维表

| Category | Pair-step count | Share of current swept |
|---|---:|---:|
| Endpoint | 892 | 41.05% |
| Swept-only | 1281 | 58.95% |
| Total | 2173 | 100% |

| Category | Endpoint | Swept-only | Total | Share of current swept |
|---|---:|---:|---:|---:|
| RR | 232 | 86 | 318 | 14.63% |
| BB | 176 | 117 | 293 | 13.48% |
| RB | 484 | 1078 | 1562 | 71.88% |
| Total | 892 | 1281 | 2173 | 100% |

同队合计611（28.12%）。RB Swept-only单项1078，占全部pair-step49.61%。Endpoint中RB484/892，仍占54.26%。

特别注意“Swept-only”不全是本步中途穿入又穿出：1281中634次步初已<=50m、步末离开；647次步初步末均>50m、线段中间<=50m。因而仅647次能称作两端均在外的中途穿越，不能把所有1281次都称为新的一次相撞。

## 6. Participant events与重叠

以下为100局累计；每队每step对UAV去重。

| Metric | Red | Blue | Combined |
|---|---:|---:|---:|
| Current swept participant | 2173 | 2104 | 4277 |
| Endpoint participant | 943 | 820 | 1763 |
| Swept-only pair participant（可重叠） | 1245 | 1297 | 2542 |
| Same-team endpoint participant | 460 | 337 | 797 |
| Same-team swept participant | 628 | 552 | 1180 |
| Cross-team swept participant | 1551 | 1556 | 3107 |
| Same/cross intersection | 6 | 4 | 10 |
| Same/cross union | 2173 | 2104 | 4277 |
| Endpoint/Swept-only intersection | 15 | 13 | 28 |
| Exclusive swept-only participant（Current−Endpoint） | 1230 | 1284 | 2514 |

集合核账：Red628+1551−6=2173；Blue552+1556−4=2104。Endpoint+Swept-only pair participants还需减重叠：Red943+1245−15=2173，Blue820+1297−13=2104。

collision_participant_sets.csv输出每队每步same_set、cross_set、both_set（intersection）、all_set（union），以及endpoint_set、swept_only_set及两者intersection。空集合也输出。按exclusive定义，2514/4277=58.78%的当前participant-events在改成Endpoint检测后会消失；不能误用2542/4277，因为后者有28个与Endpoint重叠的events。

## 7. 逐UAV burst分布

burst为同一局同一UAV的连续1序列，不跨0、死亡、episode连接；不会写入环境。连续burst不保证每步遇到相同pair，故也不能直接当物理事故计数。

| Team | burst count | mean length | median | max | P90 | P95 | bursts/episode | events/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Red | 1220 | 1.781 | 1 | 56 | 2 | 3 | 12.2 | 21.73 |
| Blue | 1282 | 1.641 | 2 | 9 | 2 | 3 | 12.82 | 21.04 |
| Combined | 2502 | 1.709 | 1 | 56 | 2 | 3 | 25.02 | 42.77 |

| Team | Length | Burst count | Share of bursts | Participant-events | Share of events |
|---|---|---:|---:|---:|---:|
| Red | 1 | 617 | 50.57% | 617 | 28.39% |
| Red | 2-3 | 572 | 46.89% | 1178 | 54.21% |
| Red | >=4 | 31 | 2.54% | 378 | 17.40% |
| Blue | 1 | 638 | 49.77% | 638 | 30.32% |
| Blue | 2-3 | 595 | 46.41% | 1228 | 58.37% |
| Blue | >=4 | 49 | 3.82% | 238 | 11.31% |
| Combined | 1 | 1255 | 50.16% | 1255 | 29.34% |
| Combined | 2-3 | 1167 | 46.64% | 2406 | 56.25% |
| Combined | >=4 | 80 | 3.20% | 616 | 14.40% |

合计70.66%的events位于length>=2的burst，但“连续第二步及之后”的events为4277−2502=1775，占41.50%。前者包含每个burst首步，不能都说成重复计罚增量。>=4步只有80段、616events；极端56步存在但不是总体唯一来源。

## 8. Team-level collision-active burst

任意一架本队UAV碰撞则team-active=1，按连续step提取。它不是participant-event序列，长度累计是team-active steps。连续active也可能由不同UAV/不同pair接力形成，不能单凭它断言整队始终扎堆。

| Team | burst count | bursts/episode | mean length | max length | active steps/episode |
|---|---:|---:|---:|---:|---:|
| Red | 926 | 9.26 | 1.855 | 56 | 17.18 |
| Blue | 931 | 9.31 | 1.791 | 12 | 16.67 |

JSON中Combined team_burst是Red/Blue段的合并样本，不是对两队active取OR再提段。

## 9. 每局分布

std为总体标准差ddof=0；百分位为numpy默认linear插值，故可非整数。Swept-only为“参与至少一个Swept-only pair”的去重数，与Endpoint有交集。Combined max_burst是该局10架UAV中的最大长度，不是两队max相加。RB endpoint pairs为全局pair数，Red/Blue列相同不是两个独立样本。

| Team | Metric | Mean | Median | Std | P25 | P75 | P90 | Max |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Red | swept | 21.73 | 17 | 17.949 | 12 | 25 | 33.3 | 121 |
| Red | endpoint | 9.43 | 6 | 15.824 | 3 | 9.25 | 13.3 | 114 |
| Red | swept_only | 12.45 | 11 | 5.977 | 8 | 16 | 21 | 39 |
| Red | same_endpoint | 4.6 | 0 | 15.714 | 0 | 2 | 6.2 | 112 |
| Red | cross_team_endpoint_pairs | 4.84 | 4 | 3.218 | 2 | 6.25 | 10 | 16 |
| Red | burst_count | 12.2 | 11 | 5.568 | 8 | 15 | 18.1 | 33 |
| Red | max_burst_length | 3.34 | 2 | 6.029 | 2 | 2 | 3.1 | 56 |
| Blue | swept | 21.04 | 16 | 16.633 | 11 | 24 | 35.1 | 101 |
| Blue | endpoint | 8.2 | 5 | 10.288 | 3 | 9 | 14.1 | 62 |
| Blue | swept_only | 12.97 | 11 | 7.549 | 8 | 16 | 22 | 47 |
| Blue | same_endpoint | 3.37 | 0 | 9.384 | 0 | 2 | 8 | 54 |
| Blue | cross_team_endpoint_pairs | 4.84 | 4 | 3.218 | 2 | 6.25 | 10 | 16 |
| Blue | burst_count | 12.82 | 11 | 7.145 | 8 | 16 | 21.1 | 41 |
| Blue | max_burst_length | 2.42 | 2 | 1.115 | 2 | 2 | 4 | 9 |
| Combined | swept | 42.77 | 34 | 26.158 | 25.75 | 52 | 78.1 | 132 |
| Combined | endpoint | 17.63 | 12 | 18.793 | 6 | 20.5 | 40.2 | 116 |
| Combined | swept_only | 25.42 | 22 | 12.131 | 16 | 32 | 40 | 70 |
| Combined | same_endpoint | 7.97 | 2 | 17.608 | 0 | 4.5 | 24.6 | 112 |
| Combined | cross_team_endpoint_pairs | 4.84 | 4 | 3.218 | 2 | 6.25 | 10 | 16 |
| Combined | burst_count | 25.02 | 22 | 11.435 | 16 | 32 | 39.1 | 61 |
| Combined | max_burst_length | 3.68 | 2 | 6.05 | 2 | 3 | 5 | 56 |

原始每局总计见collision_audit.json的episodes字段；不是只保存这些均值。

## 10. 三套离线Reward重算

每局完整轨迹结束后，使用该轨迹实际kill/weapon loss/OOB/terminal，不重新运行新reward环境：

- Current：−5 × swept unique participant-events。
- Liu-style diagnostic mapping：−0.2 × endpoint unique participant-events（任意阵营pair）。
- Gao-base diagnostic：−5 × same-team endpoint unique participant-events（只RR/BB）。

全部仍用项目50m阈值，不换论文归一化尺度。两套文献命名按本轮指定诊断映射；不是完整物理尺度复现，不是高雷完整改进算法，也不是最终推荐。没有读取这些离线值控制动作或学习。

### 10.1 每局collision reward分布与total均值

| Team | Scheme | Mean collision | Median | Std | Min | Max | Mean total |
|---|---|---:|---:|---:|---:|---:|---:|
| Red | Current | -108.65 | -85 | 89.746 | -605 | -25 | -114.55 |
| Red | Liu-style | -1.886 | -1.2 | 3.165 | -22.8 | 0 | -7.786 |
| Red | Gao-base | -23 | 0 | 78.568 | -560 | 0 | -28.9 |
| Blue | Current | -105.2 | -80 | 83.165 | -505 | -25 | -124.9 |
| Blue | Liu-style | -1.64 | -1 | 2.058 | -12.4 | 0 | -21.34 |
| Blue | Gao-base | -16.85 | 0 | 46.918 | -270 | 0 | -36.55 |

### 10.2 Red完整分项均值

| Component | Current | Liu-style | Gao-base |
|---|---:|---:|---:|
| kill_reward | 25.2 | 25.2 | 25.2 |
| weapon_loss_reward | -20.1 | -20.1 | -20.1 |
| collision_reward | -108.65 | -1.886 | -23 |
| oob_reward | -13.4 | -13.4 | -13.4 |
| terminal_reward | 2.4 | 2.4 | 2.4 |
| total | -114.55 | -7.786 | -28.9 |

### 10.3 Blue完整分项均值

| Component | Current | Liu-style | Gao-base |
|---|---:|---:|---:|
| kill_reward | 20.1 | 20.1 | 20.1 |
| weapon_loss_reward | -25.2 | -25.2 | -25.2 |
| collision_reward | -105.2 | -1.64 | -16.85 |
| oob_reward | -12.2 | -12.2 | -12.2 |
| terminal_reward | -2.4 | -2.4 | -2.4 |
| total | -124.9 | -21.34 | -36.55 |

非collision四项三种口径完全相同。Current逐回合两队collision与实际step组件核账、total与实际team_rewards累计核账误差0；Red另外与正式episode_summary核账误差0。

Liu-style同时改变检测和系数，不能将全部改善归因于去掉swept-only。代数分解：若只改Endpoint仍保留−5，collision均值R/B为−47.15/−41.00、total为−53.05/−60.70；从此再改为−0.2，collision再缩小25倍到−1.886/−1.640。这只是同轨迹计数的代数说明，没有加跑环境或另选参数。

### 10.4 Collision absolute-component share

每episode share=abs(collision)/sum(abs(五项实际分量))，零分母取0。Pooled先跨episode累计每项绝对值再求比；episode mean先各局求share再平均。没有使用collision/total_return。特别是terminal pooled用的是每局abs值，不是abs(terminal均值)。

| Team | Scheme | Pooled absolute-component share | Mean episode share |
|---|---|---:|---:|
| Red | Current | 58.37% | 53.26% |
| Red | Liu-style | 2.38% | 2.40% |
| Red | Gao-base | 22.89% | 10.54% |
| Blue | Current | 57.96% | 52.57% |
| Blue | Liu-style | 2.10% | 2.29% |
| Blue | Gao-base | 18.09% | 9.71% |

Current在pooled和平均episode口径均超过50%，是奖励分量主导项。但这不是PPO梯度归因，也不证明修改后一定更容易学。Liu降至约2%，可能过小或合适都需另行实验；不能只看total变高就断言更好。Gao平均episode share明显小于pooled，体现少数同队长burst拉高总量。

## 11. 自动提取的典型案例

A按双方current participant-events合计选最大；B按Swept-only pair-step占比最大选；C/D按RB或RR+BB pair-step数；E按单UAV最长burst。并列用最小seed，D/E重复引用同一seed已注明。

| Case | Seed | Length | Kills R/B | OOB R/B | Current events R/B | Endpoint events R/B | Swept-only pair events R/B | RR/BB/RB pair-steps | Longest burst R/B |
|---|---:|---:|---|---|---|---|---|---|---|
| A | 200081 | 300 | 0/3 | 1/0 | 31/101 | 8/62 | 23/47 | 0/43/31 | 2/7 |
| B | 200030 | 300 | 2/0 | 1/1 | 10/10 | 0/0 | 10/10 | 0/0/10 | 1/1 |
| C | 200001 | 273 | 4/0 | 1/1 | 36/36 | 12/12 | 24/24 | 0/0/36 | 2/2 |
| D | 200054 | 197 | 5/0 | 0/0 | 121/10 | 114/2 | 8/8 | 56/0/10 | 56/2 |
| E | 200054 | 197 | 5/0 | 0/0 | 121/10 | 114/2 | 8/8 | 56/0/10 | 56/2 |

B为100%Swept-only，但每UAV最长burst只有1步；同一R4/B3 pair在step34、70、106多次相遇，说明并非所有重复相遇都能用“连续多步cooldown”解释。D/E的R1/R2从step142持续到197形成56步同队Endpoint段；两架贡献112个events，Red最终5:0武器击杀获胜却return=−535，collision项−605，说明局部安全长尾可压倒作战回报。

每例最多5个代表pair如下；step为1-based完成步；距离单位m。Swept-only中的tau=0常是步初已近、步末离开，不是数值错误。

### Case A：seed 200081

| Step | First | Second | Type | Endpoint? | d_previous | d_endpoint | d_swept | tau |
|---:|---|---|---|---|---:|---:|---:|---:|
| 34 | Red0 | Blue0 | RB | false | 99.771 | 101.001 | 4.49e-14 | 0.497 |
| 34 | Red3 | Blue1 | RB | false | 142.756 | 58.048 | 1.78e-14 | 0.711 |
| 36 | Red1 | Blue4 | RB | false | 58.164 | 140.017 | 2.42e-14 | 0.293 |
| 53 | Red3 | Blue4 | RB | true | 169.928 | 31.687 | 4.26e-14 | 0.843 |
| 54 | Red3 | Blue4 | RB | false | 31.687 | 232.887 | 31.687 | 0 |

### Case B：seed 200030

| Step | First | Second | Type | Endpoint? | d_previous | d_endpoint | d_swept | tau |
|---:|---|---|---|---|---:|---:|---:|---:|
| 34 | Red1 | Blue1 | RB | false | 146.322 | 55.056 | 1.10e-13 | 0.727 |
| 34 | Red4 | Blue3 | RB | false | 140.468 | 66.259 | 1.80e-13 | 0.679 |
| 35 | Red3 | Blue0 | RB | false | 151.632 | 54.137 | 7.65e-14 | 0.737 |
| 70 | Red4 | Blue3 | RB | false | 140.468 | 66.259 | 6.78e-6 | 0.679 |
| 106 | Red4 | Blue3 | RB | false | 140.468 | 66.259 | 1.36e-5 | 0.679 |

### Case C：seed 200001

| Step | First | Second | Type | Endpoint? | d_previous | d_endpoint | d_swept | tau |
|---:|---|---|---|---|---:|---:|---:|---:|
| 34 | Red0 | Blue1 | RB | false | 106.65 | 95.984 | 1.59e-14 | 0.526 |
| 35 | Red1 | Blue4 | RB | true | 157.454 | 42.895 | 1.60e-13 | 0.786 |
| 36 | Red1 | Blue4 | RB | false | 42.895 | 242.707 | 42.895 | 0 |
| 36 | Red4 | Blue0 | RB | true | 195.873 | 46.566 | 45.771 | 0.957 |
| 37 | Red4 | Blue0 | RB | false | 46.566 | 204.863 | 46.566 | 0 |

### Case D：seed 200054

| Step | First | Second | Type | Endpoint? | d_previous | d_endpoint | d_swept | tau |
|---:|---|---|---|---|---:|---:|---:|---:|
| 142 | Red1 | Red2 | RR | true | 52.257 | 49.695 | 49.695 | 1 |
| 143 | Red1 | Red2 | RR | true | 49.695 | 47.146 | 47.146 | 1 |
| 144 | Red1 | Red2 | RR | true | 47.146 | 44.607 | 44.607 | 1 |
| 145 | Red1 | Red2 | RR | true | 44.607 | 42.074 | 42.074 | 1 |
| 146 | Red1 | Red2 | RR | true | 42.074 | 39.545 | 39.545 | 1 |

### Case E：seed 200054

| Step | First | Second | Type | Endpoint? | d_previous | d_endpoint | d_swept | tau |
|---:|---|---|---|---|---:|---:|---:|---:|
| 142 | Red1 | Red2 | RR | true | 52.257 | 49.695 | 49.695 | 1 |
| 143 | Red1 | Red2 | RR | true | 49.695 | 47.146 | 47.146 | 1 |
| 144 | Red1 | Red2 | RR | true | 47.146 | 44.607 | 44.607 | 1 |
| 145 | Red1 | Red2 | RR | true | 44.607 | 42.074 | 42.074 | 1 |
| 146 | Red1 | Red2 | RR | true | 42.074 | 39.545 | 39.545 | 1 |

## 12. 最重要结论及下一步建议（未实施）

1. 当前−5计罚没有审计实现错误：每步参与者去重、OOB顺序、pre-weapon存活者、episode核账和原行为全部对齐。
2. Swept-only占多数，且一部分是上一步末仍近、本步离开的延续；不能把pair-step当独立物理碰撞。
3. 敌我交会为主要pair来源，RR/BB仍有重要长尾。仅对同队Endpoint计罚会显著改变任务约束，不是小修。
4. 连续2–3步贡献主要events；56步极端存在，但不能用它概括100局，也不能说单加cooldown就能解决全部。
5. Current碰撞绝对分量占约58%，安全目标确实主导整体reward分量；离线Liu/Gao的高return只是换账本，非新策略成绩。
6. 最值得下一步优先审查的是collision训练目标及RB swept交会的计罚口径。建议若要改，先固定−5仅做单变量检测/阵营口径对照，再单独讨论权重。没有证据支持本轮直接删除RB、改−0.2或加入cooldown。
7. 若研究目标确实要求敌我也保持50m安全，则当前swept的检测本身有意义，重点应转向权重与作战目标平衡；Endpoint会漏掉647次两端均远的中途近距穿越，不能因为return较高就认定检测更正确。

不自动选择最终方案、不改collision_distance/penalty/swept、没有cooldown/shield/avoidance/debounce，没有新环境状态。下一轮需要先由用户确认期望的安全语义。

## 13. 原始输出

目录：runs/v2_3_greedy_shooter/，本轮新增：

- collision_audit.json：配置hash、复现、全统计、每局数据、基准summary与轨迹hash、案例。
- collision_pairs.csv：2173行真实pair-step及距离/tau/双方ID。
- collision_participant_sets.csv：46696行每队每步集合（含空集合）。
- collision_bursts.csv：2502行逐UAV连续段。
- collision_team_bursts.csv：1857行team-active连续段。
- episode_reward_recompute.csv：600行（100局×2队×3诊断口径）。

旧sanity.json、behavioral_audit.json、轨迹CSV、训练checkpoint均未覆盖。重跑会覆盖上述本轮输出，复现时建议用--output选择新目录。
