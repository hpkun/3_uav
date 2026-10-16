# V2.3 Greedy-Shooter Blue

固定连续二维5v5环境，仅更换正式固定Blue为Nearest-Target Direct-Pursuit Baseline Opponent。环境动力学、初始化、Radar/共享、68D Actor/60D Critic、Weapon、collision/OOB、Reward、Gaussian Actor/MAPPO和所有PPO超参数保持不变。

Blue只有三层：Boundary Return > Greedy Direct Pursuit > Straight Flight。每机每步独立选择当前team detected敌机中最近者，等距小ID；期望heading直接指向目标当前位置，a_norm恒0。无目标[0,0]；死亡机零动作。无目标记忆、分配/去冲突、虚拟目标点、逃逸、避碰、Boids、FSM、patrol或调速。

文献依据：于欣萌《基于强化学习的无人机集群对抗方法研究》，第3.2.2节(2)、式(3.17)–(3.19)，PDF第40–41页/印刷页30–31。论文直接给出探测内最近敌机、沿目标当前位置方向逼近、允许多机同目标的规则。原文是二维导航/控制向量；本项目采用冻结team radar共享、目标bearing→有限转率控制、a_norm=0和原boundary return，属于implementation adaptation，不是论文低层公式完全复现。

完整参数与来源见[环境规范](docs/environment_spec.md)；84项测试、smoke、4×100 sanity、三代对比和轨迹审计见[V2.3验收](docs/validation_v2_3_greedy_shooter.md)。V2.1/V2.2验证原样保留为历史。本轮未运行50k或2M。

## 冻结环境

10km方形地图、dt1s、最多300步；速度50–150m/s、初速95–105m/s；连续加速度±10m/s²、转率±10°/s，先更新速度/heading再移动。3km确定性Radar，队内即时共享、无记忆。

武器1000m、ATA半角30°、target aspect半角45°，自动同步、最近合法目标且victim去重。50m swept collision仅计每步每机安全事件、不致死；OOB仍死亡。Reward为+10 kill、-10 weapon loss、-5 collision event、-20 OOB、terminal±20/0，无potential。

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
uav2d-train --config configs/smoke.yaml --output runs/v2_3_greedy_shooter/smoke
python scripts/sanity_check.py --mode all --episodes 100 \
  --seed-start 200000 --output runs/v2_3_greedy_shooter/sanity.json
python scripts/behavioral_audit.py --episodes 20 \
  --output runs/v2_3_greedy_shooter/behavioral_audit.json
```

四模式：greedy-vs-greedy、random-vs-greedy、passive-vs-greedy、direct-pursuit-vs-greedy。最后一组仅检查原helper与正式规则一致性；本轮和greedy-vs-greedy的100局结果完全相同，逐步动作差为0。timeout与draw分别统计。

正式采样仍为CUDA learner、16个独立持久环境进程、rollout128，未修改训练参数；短smoke沿用CPU/2workers/64宽配置。默认配置输出目录命名保留V2.1，后续实验用--output单独指定Blue版本目录，不覆盖历史数据。评估、resume/init-checkpoint接口不变，不添加课程或Categorical迁移。

日志保留双方alive、weapon kill/loss、OOB、collision events、window、contact、length、五reward与八PPO指标。当前Greedy双方100局均有weapon combat，但45%达到300步、碰撞惩罚仍主导return；是否进入50k由用户审查后决定，不能仅凭胜率启动长训。
