# 文献约束下的二维连续5v5多无人机对抗基线 V2.1

基于多篇二维无人机集群对抗文献共同建模范式构建的连续5v5基线，不完全复现任何单篇论文。红方vanilla continuous MAPPO；蓝方唯一、无状态的最近detected目标chase/escape规则。

V2.1冻结V2动力学/初始化/Radar/Weapon/68D Actor/60D Critic/Gaussian网络和PPO。核心变化：collision为非致死安全事件；纯五分量event reward；删除三种Blue、assignment、tail point、patrol、避碰控制和potential shaping。

完整参数、文献来源与implementation convention见[环境规范](docs/environment_spec.md)；75项测试、1280步smoke、4×100 sanity和50k学习性验证见[V2.1验收报告](docs/validation_v2_1_5v5.md)。50k仅初步学习性证据，不能据此评价最终算法性能；2M未运行。

## 定义速览

10km方形地图、dt1s、最多300步；速度50–150m/s、初速95–105m/s；动作[a_norm,omega_norm]同时控制±10m/s²和±10°/s，先更新速度/航向再移动。3km确定性Radar，任一活友机发现即即时共享、无记忆。1km自动武器，ATA半角30°、target aspect半角45°，满足才击毁。

50m swept collision只计每步参与的独立UAV事件，不修改alive，也不阻止当步weapon。仅weapon与OOB导致死亡。Reward为+10 kill、-10 weapon loss、-5每机collision event、-20 OOB、terminal±20/0；没有shaping。

Blue仅Boundary > Chase/Escape > Straight，a_norm恒0。blue_escape_distance:null在运行时取attack_range=1000m，属于implementation convention而不是论文公布的阈值。边界仅预测当前速度/航向下一步会越界时朝反向航向转，受有限转率限制仍可能OOB；不加margin或shield。

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
# 仅尚未安装本项目时需要
pip install -e '.[dev]'
pytest -q
uav2d-train --config configs/smoke.yaml
uav2d-evaluate --config configs/smoke.yaml \
  --checkpoint runs/smoke_v2_1_5v5/checkpoints/final.pt \
  --episodes 20 --seed-start 300000
python scripts/sanity_check.py --mode all --episodes 100 \
  --seed-start 200000 --output runs/v2_1_validation/sanity.json
python scripts/behavioral_audit.py --episodes 20
```

默认正式配置仍为CUDA、16个独立持久环境进程、rollout128。环境在CPU各worker执行，learner批量推理/更新在CUDA。动作先发全部worker再收集，不是假并行。device=cuda要求可用CUDA；只有显式--device auto才自动选择CUDA/CPU，启动行显示实际device/PID。

```bash
# sanity满足武器交战且OOB不明显主导时，才运行50k学习性验证
uav2d-train --config configs/base.yaml --steps 50000 --output runs/v2_1_learnability_seed7
# 下列是未来正式命令，本轮未运行，不代表已批准长训
uav2d-train --config configs/base.yaml
```

2M为全部环境合计2,000,000步，不是每环境2M或2M回合；最后horizon缩短，steps须能被parallel_envs整除。支持--steps/--output/--device/--seed，已删除--blue-strategy。

```bash
uav2d-evaluate --config configs/base.yaml \
  --checkpoint runs/v2_1_5v5_seed7/checkpoints/final.pt \
  --episodes 500 --output runs/v2_1_5v5_seed7/evaluation.json
uav2d-train --config configs/base.yaml --resume PATH_TO_V2_1_CHECKPOINT
# 仅权重初始化，重置optimizer/计数
uav2d-train --config configs/base.yaml --init-checkpoint PATH_TO_COMPATIBLE_CHECKPOINT
tensorboard --logdir runs/v2_1_5v5_seed7/tensorboard
```

resume与init-checkpoint互斥，不自动添加课程。smoke网络64宽，正式128宽，不能混载。旧Categorical checkpoint不迁移。旧V2 continuous形状相容但任务已改变，不将旧训练数据当V2.1证据；本轮从零训练。

输出：episodes.csv逐回合、updates.csv每次更新最近100回合均值及八PPO指标、tensorboard/、checkpoints/。日志明确使用collision_events，不是collision loss；五reward可直接核账return。评估默认确定性tanh(mean)，--gif可保存首局动画，原图标/轨迹保持。

无导弹、弹药、概率命中、fire action、attention/GRU/GNN、自博弈、复杂战术或课程。旧验证文档和实验数据保留并标记历史。
