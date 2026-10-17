# V2.5 验收报告

执行日期：2026-10-17（Asia/Shanghai）。所有运行命令均从PowerShell调用Windows conda `uav`，解释器为`D:\conda_envs\envs_dirs\uav\python.exe`。没有启动正式RL训练。完整参数来源与未采用机制见[environment_v2_5_deng_combat.md](environment_v2_5_deng_combat.md)。

## 1. 修改与新增文件

| 文件 | 修改 |
|---|---|
| src/uav_mappo_2d/config.py | 速度/初速默认值、新攻击字段与校验；Reward/算法配置未改 |
| src/uav_mappo_2d/env.py | V2.5 class说明、alpha-only weapon gate、一次seeded随机试验 |
| configs/base.yaml | 正式V2.5速度和武器参数 |
| src/uav_mappo_2d/train.py、evaluate.py | 仅CLI版本描述 |
| src/uav_mappo_2d/train_maddpg.py | 仅模块版本说明 |
| scripts/behavioral_audit.py | 新几何gate、端点碰撞诊断、区分尝试/实际目标死亡，成功attacker ID未知 |
| scripts/replay_html.py、replay_template.html | V2.5版本、按配置显示攻击扇区、窗口说明；沿用真实alive变化与info核账 |
| scripts/sanity_check.py | 当前版本及几何window统计说明；策略逻辑未改 |
| scripts/validate_v2_5.py（新增） | 100局Greedy、相同seed逐步重放、事件/Reward核账、CUDA MAPPO smoke与HTML |
| scripts/smoke_maddpg.py | V2.5独立输出、奖励/配置快照核对、smoke worker数量选项、CUDA自动选择 |
| scripts/collision_audit.py、validate_v2_4.py | 明确历史入口，拒绝把历史验收断言套在V2.5；原历史数学/正文保留 |
| tests/test_environment.py、test_rewards.py、test_opponents.py、test_collision_audit.py | 更新适用速度fixture；必杀事件测试显式使用p=1；删除旧aspect/ATA断言 |
| tests/test_environment_v2_5.py（新增） | 配置校验、初速、clipping/归一化、距离/角度边界、aspect独立、hit/miss/Reward、无重试、同步去重、seed与禁止全局random |
| tests/test_v2_5_tools.py（新增） | 审计aspect独立、全miss时有window/attempt但无kill、死亡确认与HTML扇区 |
| docs/environment_spec.md | 当前正式完整规范 |
| docs/environment_v2_5_deng_combat.md（新增） | A论文参数/B项目约定/C暂未采用设计 |
| docs/html_replay.md | 当前回放版本和概率攻击解释 |
| docs/maddpg_baseline.md | 仅开头加当前V2.5指引，原V2.4实验数字保留 |
| docs/validation_v2_5_deng_combat.md（新增） | 本验收报告 |
| README.md、pyproject.toml | 当前环境说明与项目description版本 |

V2.3/V2.4历史validation文档与已有runs数据未修改。临时编辑/源码核查脚本与本次pytest临时目录在完成后清理，保留原`tmp/pdfs`。

## 2. 正式参数与语义

| 参数 | V2.5 |
|---|---:|
| speed_min / speed_max | 200 / 400 m/s |
| initial_speed_min / initial_speed_max | 300 / 300 m/s |
| attack_range / attack_angle_degrees | 1000 m / 90°半角，inclusive |
| hit_probability / radar_range | 0.70 / 3000 m |
| dt / max_acceleration / max_turn_rate_degrees | 1s / 10m/s² / 10°/s，未改 |
| OBS_DIM / STATE_DIM / ACTION_DIM | 68 / 60 / 2，未改 |

初速300只是200–400的中点约定，论文未明确给出该初速。运动仍为连续加速度/转率控制、先更新速度航向再移动；200/300/400在obs/state中映射-1/0/+1，相对速度分母改由配置自动得到200。

`alpha=wrap_angle(LOS_heading-attacker_heading)`；攻击资格仅为alive且`distance<=1000 AND abs(alpha)<=90°`。`_attack_valid`不再使用beta；geometry返回值和观测中的beta保留。旧`attack_ata_degrees`、`attack_aspect_degrees`已从src/config/scripts执行逻辑移除，只在历史文档、删除说明与缺席断言中出现。

每attacker每step按(distance,target ID)选一个最近合法目标，用`env.np_random.random()<.70`尝试一次；无合法目标不抽攻击随机数，miss不换目标重试。没有全局random攻击源。

OOB→endpoint collision→attack的顺序不变；collision不致死也不阻止同step攻击。所有攻击使用post-OOB/post-collision/pre-weapon快照，完成全部试验后统一删除victims。后续攻击者不会因前一命中失去目标；同victim多次命中只计一次death/combat kill/weapon loss，支持mutual kill。

`attack_window_count`累计该快照所有几何合法attacker-target pair；miss也累计。不是发射、命中、击杀次数，不能用它与kill的比值作为命中率。审计只重建几何窗口和最近目标尝试；目标weapon death从真实状态变化确认，成功攻击者ID未知，不猜测。

未实现ammo=6：论文未明确当前自动攻击下的fire action、发射/消耗时机、cooldown/reload及miss后重发规则。未加入radar angle：保留V2.4的360°range-only、team_detected_enemy_mask和共享，只有radar_range与论文一致。本轮不同时改变观测可用性和Greedy目标获取。

Reward定义/归属完全保持：+10 kill、-10 weapon loss、-20 OOB、-0.2 collision participant-event、terminal ±20/0，无miss penalty、attack reward或shaping。Greedy源文件完全未改，nearest detected/direct pursuit/acceleration=0及boundary/control不变，默认自然恒速300m/s。

## 3. 完整测试与源码范围审计

最终命令：

```powershell
conda run --no-capture-output -n uav pytest -q --basetemp tmp/pytest_v25_final -o addopts=
```

**205 passed in 25.27s**，无失败或跳过。`--basetemp`仅绕开本机原pytest-of-HPK目录的Windows权限问题；`-o addopts=`取消项目中重复的-q以显示总数。记录：`runs/v2_5_deng_combat/pytest.txt`。首次运行前为uav补齐缺失gymnasium/tensorboard（另安装pygame），并以editable方式安装当前项目，之后所有验证均使用uav。

新增两组V2.5专项测试，连同原有仍适用的测试覆盖几何边界、不同target heading、概率阈值（random=.70为miss）、无window不抽样、miss无kill/loss reward、最近目标/ID tie-break、重复命中去重、同快照反击、相同seed重放、不同seed不得依赖全局random，以及审计/HTML适配。既有端点碰撞/OOB precedence/非致死与parallel env、算法、checkpoint测试均通过。

`source_scope_audit.json`对照干净起始HEAD：opponents/models/mappo/maddpg/maddpg_models/maddpg_config/parallel_env/evaluate_maddpg/rendering、两个正式MADDPG配置及smoke配置的规范化源码SHA256不变。AST核对env仅_attack_valid/_resolve_attacks及class说明改变；step/Reward、运动积分、Radar、spawn、collision、归一化函数均未改。train/evaluate仅CLI描述；train_maddpg仅模块docstring。没有自行调整地图、Reward、turn rate、spawn、episode steps、OU、网络、算法公式或正式超参数。

## 4. Greedy-vs-Greedy 100局

命令：`python scripts/validate_v2_5.py`，seed 200000–200099。双方同一个无状态Greedy实现。原始逐局summary及完整指标：`runs/v2_5_deng_combat/validation.json`。

每个seed另建独立同seed环境，重放完全相同动作。3244个step的obs/state/物理状态/Reward/武器结果/RNG状态完全一致；真实状态变化重建的weapon/OOB死亡与环境计数一致。奖励累计与五分量核账最大绝对误差0。

**Red W/D/L = 46 / 3 / 51。**

| 指标（每局均值） | Red | Blue |
|---|---:|---:|
| Return | -9.714 | -6.502 |
| Alive | 0.76 | 0.82 |
| Combat kills | 3.83 | 3.78 |
| Weapon losses | 3.78 | 3.83 |
| OOB | 0.46 | 0.35 |
| Collision participant-events | 0.07 | 0.01 |
| Geometric attack windows | 7.58 | 7.66 |
| Radar contact steps | 19.55 | 19.55 |

平均episode length **32.44**步。长度min/p25/median/p75/max = **11/13/30/36/300**。Red/Blue累计kill为383/378、weapon loss为378/383、OOB为46/35、collision为7/1、window为758/766、contact为1955/1955。

| Red奖励分量均值 | 数值 |
|---|---:|
| kill_reward | 38.3 |
| weapon_loss_reward | -37.8 |
| collision_reward | -0.014 |
| oob_reward | -9.2 |
| terminal_reward | -1.0 |
| sum | -9.714 |

现象判断：

- 46比51的胜局、近似对称的alive/kills/windows中，未见明显结构性单边偏置；100局不证明严格无偏。
- 无接敌回合0，无武器交战回合0；不是几乎没有接敌。
- 达到300步2局（2%），不是几乎全部超时。
- ≤20步终止37局（37%），值得关注节奏明显偏快，但不是几乎全部很早终止；本轮不自行修正。
- OOB合计81架，占1000架初始飞机8.1%，占842次总死亡9.61995%；未见OOB主导死亡的极端情况，仍如实保留边界损失。
- 本轮同时改变速度和武器几何/概率，不能从这100局单独归因于速度。未为了这些现象修改地图、turn rate、spawn或Reward。

## 5. MAPPO CUDA smoke

命令：`python scripts/validate_v2_5.py --smoke-only`。GPU为NVIDIA GeForce RTX 5060 Laptop GPU，PyTorch 2.11.0+cu128，CUDA可用。

沿用`configs/smoke.yaml`轻量profile：seed11、hidden64、epochs2、batch32、rollout64、2真实worker；仅选择CUDA和独立输出目录，正式base未改。

| 检查 | 结果 |
|---|---|
| env steps / updates / completed episodes | 1280 / 10 / 45 |
| Actor / Critic参数L2变化 | 0.338223934 / 1.043125391 |
| loss/梯度等指标 | 全部finite |
| 最后actor / critic loss | 0.002526687 / 320.085007 |
| checkpoint save/load | Actor/Critic逐tensor精确重载，optimizer加载正常 |
| Reward accounting | 最大误差0 |
| 配置 | smoke_config_snapshot.json准确记录V2.5；原MAPPO checkpoint不嵌入环境配置，格式保持原样 |
| rollout / evaluate / replay | 全部完成；回放30 frames |

产物：`runs/v2_5_deng_combat/mappo_smoke/`的checkpoint、episodes.csv、updates.csv、TensorBoard、smoke_config_snapshot.json与smoke_validation.json。4局holdout均loss，仅作链路验证，不评价性能。

## 6. MADDPG CUDA smoke

第一次沿用16worker在本机Windows触发WinError1455（页面文件不足），未完成更新，保留失败输出目录以区分。停止该次启动后，采用smoke专用2worker重跑；未修改Windows页面文件或正式16worker配置。

```powershell
python scripts/smoke_maddpg.py --device cuda --parallel-envs 2 --output runs/v2_5_deng_combat/maddpg_smoke_2workers
```

沿用原smoke的临时batch32/learning_starts128/buffer4096，正式configs/maddpg.yaml与pretanh配置未改。seed7、2真实worker、2560步。

| 检查 | 结果 |
|---|---|
| env steps / update rounds / completed episodes | 2560 / 24 / 100 |
| 五个Actors参数L2变化范围 | 6.119229–6.777500 |
| 五个Critics参数L2变化范围 | 4.782859–5.138904 |
| target Actors / target Critics变化范围 | 0.922016–1.015019 / 0.663155–0.740909 |
| Actor公式核查 | 120次，最大误差0 |
| soft-update公式最大误差 | 2.9802322e-8 |
| mean actor loss范围 | -0.120139–3.674290 |
| mean critic loss范围 | 10.344948–104.127335 |
| loss/梯度/Q/TD指标 | 全部finite |
| checkpoint save/load | 全网络逐tensor精确重载，optimizer加载正常 |
| checkpoint/config snapshot | environment精确为V2.5配置 |
| Reward accounting | 最大误差0 |

产物：`runs/v2_5_deng_combat/maddpg_smoke_2workers/`，详见smoke_validation.json与config_snapshot.json。4局holdout均loss，现有smoke还记录动作饱和；仅验证链路，不评价性能或据此调参。

## 7. 当前工具与命令行回归

- MAPPO正式evaluate CLI：2局CUDA，checkpoint正常，输出`mappo_cli_evaluation.json`。
- MADDPG正式evaluate CLI：2局CUDA，checkpoint正常，输出`maddpg_cli_evaluation.json`。
- `replay_html.py`实际CLI：使用匹配hidden64的configs/smoke.yaml加载MAPPO checkpoint，30 frames，状态/事件/Reward核账通过，输入source/config/checkpoint哈希不变。当前扇区HTML为`runs/v2_5_deng_combat/replay_current.html`。
- behavioral_audit实际执行两个模式各2局：helper动作差0，最大控制核查误差2.93e-8；Blue 14个几何window、13次尝试、9个真实kill，未把这些数量解释为命中率；成功attacker ID始终未知。
- 另有全miss审计测试：有window与attempt，无weapon death，不误报kill。
- parallel env、joint replay、checkpoint、HTML与两类算法测试包含在205项完整pytest中。

本轮只完成环境升级、测试、规则基线与短smoke。没有启动正式50k/2M训练，没有因为结果而调地图、Reward、turn rate、spawn或算法参数。
