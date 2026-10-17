# V2.5：连续二维5v5空战

当前正式环境将速度与基础武器模型对齐到本次提供的邓岩《基于深度强化学习的无人机集群空战决策研究》参数：速度200–400m/s、攻击距离1000m、前向攻击半角90°、命中概率70%、探测距离3000m。初速固定300m/s是项目取中点的约定，不来自论文。

来源分类与适配边界见[V2.5设计说明](docs/environment_v2_5_deng_combat.md)，全部参数和接口见[当前环境规范](docs/environment_spec.md)，测试及100局结果见[V2.5验收](docs/validation_v2_5_deng_combat.md)。V2.3/V2.4报告与实验数据保留为历史，不改写历史数字。V2.5不是论文环境的逐行复现。

武器资格为`distance <= 1000 AND abs(wrap(LOS_heading-attacker_heading)) <= 90°`，边界inclusive；不检查target aspect。每个攻击者每step按(distance,target ID)选一个最近合法目标，用`env.np_random.random() < .70`试验一次。所有试验共用pre-weapon快照，再同步结算去重victims。attack windows计全部几何合法pair，包含miss；它不等于尝试、命中或击杀次数，不能拿window/kill推算命中率。

10km地图、dt1s、300步、加速度±10m/s²、转率±10°/s、spawn、endpoint非致死碰撞、OOB顺序保持原样。Reward仍为kill +10、weapon loss -10、OOB -20、collision participant-event -0.2、terminal ±20/0，无miss penalty或shaping。OBS/STATE/ACTION仍为68/60/2。

Greedy Blue仍是Boundary Return > nearest detected Direct Pursuit > Straight Flight，acceleration=0，自然以初速300m/s飞行。探测仍为V2.4的360°range-only与即时team sharing。没有实现ammo=6、radar angle、fire action或cooldown，理由见设计说明。MAPPO/MADDPG/OU/网络与正式算法超参数未改。

## PowerShell / conda uav

```powershell
conda activate uav
cd C:\Users\HPK\Desktop\3_uav
python -m pip install -e '.[dev]'
pytest -q
python scripts/validate_v2_5.py
python scripts/validate_v2_5.py --smoke-only
python scripts/smoke_maddpg.py --parallel-envs 2
```

先做100局规则验收，再做短smoke。CUDA可用时使用CUDA；MAPPO沿用1280步轻量profile，MADDPG沿用2560步smoke临时batch32/learning_starts128/buffer4096，Windows资源受限时用2worker。正式base仍16worker，算法正式配置文件未改；smoke不评价性能，不启动50k或2M。输出在`runs/v2_5_deng_combat/`；复跑选新目录，避免覆盖。Windows旧pytest临时目录权限不足时可加`--basetemp tmp/pytest_v25_new`。

`sanity_check.py`四模式接口保留；当前行为审计用`behavioral_audit.py`。历史`collision_audit.py`、`validate_v2_4.py`需使用相应历史checkout，入口会拒绝在当前代码直接套用旧验收。

## 回放与算法

[单文件HTML回放说明](docs/html_replay.md)。使用本次MAPPO smoke checkpoint（hidden64）生成回放：

```powershell
python scripts/replay_html.py --config configs/smoke.yaml --checkpoint runs/v2_5_deng_combat/mappo_smoke/checkpoints/final.pt --seed 3000000 --device cuda --output runs/v2_5_deng_combat/replay_current.html
```

正式hidden128 checkpoint使用base配置。smoke已自动生成`mappo_smoke/replay_seed3000000.html`；回放从真实状态变化/info确认weapon death，不猜成功攻击者ID，选中飞机可显示攻击扇区。旧权重形状相容不代表旧实验属于V2.5。

独立MADDPG保持5个Actor/5个centralized Critic及target、joint replay和OU；说明及历史短程结果见[MADDPG文档](docs/maddpg_baseline.md)。其中V2.4实验数字是历史记录；当前环境以V2.5规范为准，MAPPO/MADDPG checkpoint格式仍不兼容。
