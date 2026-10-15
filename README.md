# 通用二维连续 5v5 多无人机对抗基线 V2

Fixed 5v5 Continuous 2D UAV Combat Baseline V2。固定红蓝各5架、二维定高，红方共享策略continuous MAPPO，蓝方简单固定规则。这是多篇二维多无人机对抗文献共同范式下构建的通用baseline，不是某一篇论文复现。

当前接口：68D局部Actor、60D全局Critic、单机连续动作[a_norm, omega_norm]∈[-1,1]²。10 km地图、1 s步长、最多300步；3 km雷达及即时队内共享；1 km自动武器、ATA半角30°、target aspect半角45°；50 m swept collision；真实出界损毁。先更新速度/航向，再移动。

完整定义见[环境规范](docs/environment_spec.md)，实际行为数据和已知问题见[V2验收报告](docs/validation_v2_5v5.md)。本轮规则对抗仍由碰撞主导，不推荐直接进入正式长训。旧Xu和旧continuous数据不作为V2证据。

## 运行

在Windows终端执行：

```powershell
wsl -d Ubuntu
```

进入WSL后：

```bash
source /home/hpk/anaconda3/etc/profile.d/conda.sh
conda activate uav
cd /mnt/c/Users/HPK/Desktop/3_uav
# 仅首次尚未安装本项目时需要：
pip install -e '.[dev]'
pytest -q
uav2d-train --config configs/smoke.yaml
uav2d-evaluate --config configs/smoke.yaml \
  --checkpoint runs/smoke_v2_5v5/checkpoints/final.pt \
  --episodes 20 --seed-start 300000 \
  --output runs/v2_validation/smoke_evaluation.json
python scripts/sanity_check.py --mode all --episodes 100 \
  --seed-start 200000 --output runs/v2_validation/sanity.json
python scripts/behavioral_audit.py --episodes 20
```

正式配置默认CUDA learner、16个独立持久子进程环境、每环境128步rollout、总计2M环境步。父进程批量推理，向全部worker发送后再收集；各worker独立推进env与蓝方策略，是真正多进程，不是顺序调用16个env。环境NumPy计算在CPU，learner在CUDA；CUDA不可用会告警回退CPU，启动行显示实际device和worker PID。

以下只提供正式命令，本轮未执行；须先解决验收报告中的行为问题：

```bash
uav2d-train --config configs/base.yaml
```

2M是所有环境合计2,000,000步，不是每环境2M，也不是2M回合。最后一轮缩短horizon，步数不向上取整；请求步数必须能被parallel_envs整除。可用--steps、--output、--device、--seed、--blue-strategy覆盖配置，默认蓝方Strategy2。

```bash
# 同一V2结构断点续训，保留optimizer/计数/best评估
uav2d-train --config configs/base.yaml --resume runs/v2_5v5_seed7/checkpoints/final.pt
# 仅兼容V2权重初始化，重置optimizer和计数，不自动构成课程
uav2d-train --config configs/base.yaml --init-checkpoint PATH_TO_V2_CHECKPOINT
uav2d-evaluate --config configs/base.yaml \
  --checkpoint runs/v2_5v5_seed7/checkpoints/best.pt \
  --episodes 500 --output runs/v2_5v5_seed7/evaluation.json
tensorboard --logdir runs/v2_5v5_seed7/tensorboard
```

resume与init-checkpoint互斥。旧Categorical checkpoint不兼容，不提供迁移。smoke网络宽64，仅适配自身配置；正式网络宽128，不能直接使用smoke权重初始化。

## 日志与范围

输出目录包含episodes.csv（逐回合）、updates.csv（每次PPO更新，最近100回合汇总及八项PPO指标）、tensorboard/、checkpoints/。控制台显示step、完成回合数、滑动平均return/WDL、双方alive/weapon kill与loss/OOB/collision/window/contact、长度、五reward及八PPO指标。评估使用确定性tanh(mean)，支持--gif保存首回合动画，沿用原无人机图标和轨迹渲染。

不包含导弹、弹药、概率命中、fire action、GRU/attention/GNN、自博弈、复杂战术FSM或课程学习。三份旧Xu课程配置已删除，历史文档顶部明确标为HISTORICAL / DEPRECATED，实验数据保留。
