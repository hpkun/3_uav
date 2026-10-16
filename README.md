# V2.4 Endpoint Collision：Greedy-Shooter Blue

固定连续二维5v5环境。本轮只将正式collision从swept改为endpoint center-distance，penalty从5.0改为0.2。Greedy Blue、动力学、初始化、Radar/共享、68D Actor/60D Critic、Weapon、OOB、其余Reward、Gaussian Actor/MAPPO和所有PPO超参数保持不变。

Blue只有三层：Boundary Return > Greedy Direct Pursuit > Straight Flight。每机每步独立选择当前team detected敌机中最近者，等距小ID；期望heading直接指向目标当前位置，a_norm恒0。无目标[0,0]；死亡机零动作。无目标记忆、分配/去冲突、虚拟目标点、逃逸、避碰、Boids、FSM、patrol或调速。

文献依据：于欣萌《基于强化学习的无人机集群对抗方法研究》，第3.2.2节(2)、式(3.17)–(3.19)，PDF第40–41页/印刷页30–31。论文直接给出探测内最近敌机、沿目标当前位置方向逼近、允许多机同目标的规则。原文是二维导航/控制向量；本项目采用冻结team radar共享、目标bearing→有限转率控制、a_norm=0和原boundary return，属于implementation adaptation，不是论文低层公式完全复现。

完整参数与来源见[环境规范](docs/environment_spec.md)；当前测试、100局逐步轨迹对照、逐局counterfactual核账和smoke见[V2.4验收](docs/validation_v2_4_endpoint_collision.md)。V2.1/V2.2/[V2.3验收](docs/validation_v2_3_greedy_shooter.md)及[V2.3只读碰撞审计](docs/collision_audit_v2_3.md)原样保留为历史。本轮未运行50k或2M。

## 冻结环境

10km方形地图、dt1s、最多300步；速度50–150m/s、初速95–105m/s；连续加速度±10m/s²、转率±10°/s，先更新速度/heading再移动。3km确定性Radar，队内即时共享、无记忆。

武器1000m、ATA半角30°、target aspect半角45°，自动同步、最近合法目标且victim去重。collision在movement后、OOB后、weapon前，对仍alive的RR/BB/RB全部pair计算当前中心距离，≤50m触发；每机每步最多一个event、不致死、不阻止本步weapon。Reward为+10 kill、-10 weapon loss、-0.2 collision participant-event、-20 OOB、terminal±20/0，无potential。

刘莹春《基于强化学习的无人机集群协同打击策略研究》第3.2.3节式(3.11)（正文28页/PDF39页）提供当前距离触发小额-0.2安全惩罚的依据。原文为归一化距离<0.5；本项目50m、inclusive ≤、shared-team按本队参与者累计、双阵营全部pair及非致死时序属于implementation adaptation，不是完整复现论文。

V2.3采用swept/-5，其只读审计中58.95%的pair-step为swept-only，71.88%为RB，碰撞绝对奖励分量占约58%。V2.4将其恢复为辅助安全项。历史swept诊断不进入当前正式Reward、observation或state，也没有新增cooldown或避碰。

边界函数完全不变：当前速度/heading名义预测下一步越界时，仅朝反向heading有限转向；不加margin或安全层，仍可能OOB。

## 运行（WSL + conda uav）

Windows终端：

```powershell
wsl -d Ubuntu
```

WSL：

```bash
source /home/hpk/anaconda3/etc/profile.d/conda.sh
conda activate uav
cd /mnt/c/Users/HPK/Desktop/3_uav
pytest -q
python scripts/validate_v2_4.py --smoke
```

V2.4验收只跑Greedy-vs-Greedy的100个固定seed200000–200099。validate_v2_4.py从记录的V2.3提交c33ed9e读取并验证与历史审计SHA256一致的环境源码，在隔离模块中做对照，不修改或patch生产类；该提交不可用时明确停止。逐步核对动作/物理状态/obs/state/RNG，逐局核对历史Liu-style反事实；通过后--smoke才运行1280步短训。结果在runs/v2_4_endpoint_collision/，复跑请用--output选择新目录，避免覆盖或追加旧smoke。

原sanity四模式接口保留；本轮不额外运行random/passive。scripts/collision_audit.py和scripts/behavioral_audit.py是保留的V2.3 swept审计工具，不能用它们的旧断言验证V2.4生产语义。其历史数据/报告不覆盖；当前使用validate_v2_4.py。

正式采样仍为CUDA learner、16个独立持久环境进程、rollout128，未修改训练参数；短smoke沿用CPU/2workers/64宽配置。默认配置输出目录命名保留V2.1，后续实验用--output单独指定Blue版本目录，不覆盖历史数据。评估、resume/init-checkpoint接口不变，不添加课程或Categorical迁移。

日志保留双方alive、weapon kill/loss、OOB、collision events、window、contact、length、五reward与八PPO指标；collision events现在指endpoint unique participants。45%达到300步和OOB等问题本轮不改；是否进入50k由用户审查V2.4验收后决定，不自动长训。

## 单文件 HTML 录像（只读）

完整用法和时间对齐说明见[HTML回放文档](docs/html_replay.md)。以下在WSL的conda uav中执行；生成文件可直接双击离线播放，无CDN/图片帧/JSON旁文件，不改环境或checkpoint，也不自动打开浏览器。

```bash
PYTHONPATH=src python scripts/replay_html.py \
  --config configs/base.yaml \
  --checkpoint runs/v2_4_mappo_2m_seed7/checkpoints/best.pt \
  --seed 3000000 --device cuda \
  --output runs/v2_4_mappo_2m_seed7/replays/best_seed3000000.html
```

默认deterministic，与正式evaluate一致；加`--stochastic`则按seed采样。省略`--output`默认生成`runs/replays/<checkpoint_stem>_seed<seed>.html`。已存在的输出会拒绝覆盖，请改文件名。原GIF功能不变。
