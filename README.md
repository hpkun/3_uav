# Xu-thesis-aligned 5v5 baseline

基于徐金辉论文第3章环境定义和第4章课程框架构建的二维同高度、固定5v5对齐基线。论文未公开的实现细节采用明确记录的项目约定；本项目只实现vanilla shared-team MLP MAPPO，不实现注意力、GRU、沙普利或威胁度信用分配。

完整参数、论文对齐表和实现选择见 [环境规范](docs/environment_spec.md)，实际行为、测试和50k短训练结果见 [Xu验收报告](docs/validation_xu_5v5.md)。此前continuous 5v5、500k或旧3v3训练结果均不作为当前基线的有效结果；旧checkpoint不兼容，不提供迁移。

| 场景 | 当前定义 |
|---|---|
| 名义地图 / 回合 | 10km×10km / 最多300steps |
| 初始 / 最小 / 最大速度 | 100 / 50 / 150 m/s |
| 初始heading | 红0，蓝pi，无航向扰动 |
| 离散动作 | 0–4：速度增量[-1,-.5,0,.5,1]；5–9：角增量[-5,-2,0,2,5]° |
| 位移时序 | 使用v_t和heading_t移动，再应用动作到v_(t+1)/heading_(t+1) |
| 雷达 | 2km确定性探测，队内完美共享，无记忆 |
| 自动武器 | 600m，前向半角15°，目标尾后半角50° |
| 碰撞 / 避碰 / 辅助奖励 | 150m swept collision / 最近机450m动作约束 / 最近机<600m距离差奖励 |
| Actor / Critic | 68D局部固定slot / 60D全局状态，均为项目编码选择 |
| action / action mask | 整个env(2,5)整数 / 红方(5,10)bool |
| Actor / Critic网络 | 68–128–128–10 Categorical / 60–128–128–1 |

位置分布沿用x=±3500m、y=[-3000,-1500,0,1500,3000]m及各坐标±200m独立扰动；精确出生区间和dt=1s是project convention。名义地图没有OOB死亡、惩罚、反射、teleport或物理clamp，归一化坐标可以超过1，渲染仅裁剪绘图坐标。

当最近存活飞机（任意阵营）距离≤450m，动作仅允许朝远离方向转向的三个steering actions：alpha≥约0时{5,6,7}，alpha<0时{7,8,9}。规则双方均优先选±5°避碰。死亡实体仅允许canonical no-op 2，active mask为0。MAPPO在采样和PPO重算时使用同一份rollout action mask。

团队reward为以下七项之和，辅助项先逐存活agent计算再求team mean：

```text
kill_reward = +20 * weapon_kills
own_loss_reward = -20 * (own_weapon_deaths + own_collision_deaths)
terminal_reward = 对方全灭:+100 / 己方全灭:-100 / 双方同时全灭:0
timeout_reward = 双方仍有存活且到300步时，数量优势:+80 / 劣势:-80 / 平:0
distance_reward = mean((d_prev_nearest_detected-d_curr_nearest_detected)/1000)
                  仅前后都有contact且d_prev>1500m
attack_angle_reward = mean(min_abs_alpha_prev-min_abs_alpha_curr)，radians
collision_avoidance_reward = mean((d_curr_nearest-d_prev_same_aircraft)/1000)
                            仅当前最近机距离<600m
```

任务奖励采用相加实现，最后一架武器击杀可同时产生+20和+100。损失和辅助项的shared-team aggregation、单位转换、同时全灭处理等均在环境规范中明确标为适配/约定。

## WSL运行

```bash
wsl -d Ubuntu
source /home/hpk/anaconda3/etc/profile.d/conda.sh
conda activate uav
cd /mnt/c/Users/HPK/Desktop/3_uav
python -m pip install -e . --no-deps
pytest -q
uav2d-train --config configs/smoke.yaml
uav2d-evaluate --config configs/smoke.yaml \
  --checkpoint runs/smoke_xu_5v5/checkpoints/final.pt --episodes 20
python scripts/sanity_check.py --mode all --episodes 100 \
  --output runs/xu_validation/rule_sanity.json
```

Smoke为1280env steps、2进程、10次PPO update、CPU，验证离散动作、掩码、保存与加载链路。

## 三阶段课程

```bash
# Course1：Strategy1，2M interactions
uav2d-train --config configs/xu_course1.yaml

# Course2：Strategy2，新增2M；只加载权重，重置optimizer、step和评估记录
uav2d-train --config configs/xu_course2.yaml \
  --init-checkpoint runs/xu_course1_seed7/checkpoints/final.pt

# Course3：Strategy3，新增6M
uav2d-train --config configs/xu_course3.yaml \
  --init-checkpoint runs/xu_course2_seed7/checkpoints/final.pt
```

`--resume path`保留原有优化器、global_step、episode counter、best evaluation的断点恢复语义；`--init-checkpoint path`只加载Actor/Critic权重，两者互斥。课程配置默认CUDA、16独立进程、rollout128、共享Actor和team GAE。PPO超参数沿用项目，不声称复现论文完整算法超参数表。

`base.yaml`为Course1环境的展开配置。可用`--blue-strategy 1|2|3`覆盖训练或评估蓝方；定期评估使用与该课程相同的策略。独立测试示例：

```bash
uav2d-evaluate --config configs/xu_course3.yaml \
  --checkpoint runs/xu_course3_seed7/checkpoints/best.pt \
  --episodes 500 --seed-start 2000000
tensorboard --logdir runs/xu_course1_seed7/tensorboard
```

终端输出最近100个完整回合的return、W/D/L、武器击杀、碰撞、存活、outside diagnostic、七项reward和PPO指标。CSV包含return/total_return及各分量，TensorBoard记录每个回合和每次update，EV使用更新后的critic。

## 当前验证边界

81项测试通过，三种规则各100局和50k Course1短训练已执行，但尚未证实可学习攻击：本次规则sanity和50k训练均没有武器击杀。Strategy3仍有较多碰撞，450m动作约束已核对严格生效；dt=1s和旧航向位移使某些迎头情况在转向生效前进入碰撞范围。另有较多存活飞机飞出名义地图。请依据验收报告的数据决定后续实验，不将“数值链路正常”等同于“论文行为效果已恢复”。
