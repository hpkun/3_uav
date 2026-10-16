# 固定5v5连续二维对抗：Rear-Pursuit Blue

基于V2.1冻结环境，仅将固定Blue更换为nearest-detected-target rear-aspect pursuit。红方vanilla continuous MAPPO；Blue唯一、无状态、恒速，仅通过有限转率控制航向。不是任何单篇论文的完全复现。

优先级：Boundary Return > Rear-Aspect Pursuit > Straight Ingress。每步每机选最近detected敌机（等距小ID），朝其当前航向后方attack_range处接近，无目标[0,0]。不增加可调尾距、assignment、target memory、patrol、避碰控制或新reward shaping。

```text
p_rear = p_target - attack_range*[cos(h_target),sin(h_target)]
desired_heading = atan2(p_rear.y-own_y,p_rear.x-own_x)
# 若距rear point<1e-6，仅回退target heading
action = [0, clip(wrap(desired_heading-own_heading)/(max_turn_rate*dt),-1,1)]
```

“绕到敌机后方”的战术思想参考孙世彬《基于多智能体强化学习的大规模无人机集群博弈对抗研究》第三章固定规则交战阶段；rear-point公式和以attack_range作为接近点距离尺度均为implementation convention，不是论文公开控制公式。作者/题名、项目PDF相对文件名、页码及其他冻结环境来源见[环境规范](docs/environment_spec.md)。

本轮实际测试与行为数据见[Rear-Pursuit验收报告](docs/validation_v2_2_rear_pursuit.md)；[V2.1历史验收](docs/validation_v2_1_5v5.md)原样保留。新策略的闭环效果必须根据本轮sanity判断，不依据历史50k结果；本轮不运行50k或2M。

## 冻结环境

固定5v5、dt1s、最多300步、10km方形地图；速度50–150m/s，初速95–105m/s；连续加速度±10m/s²、转率±10°/s，先更新速度/heading再移动。3km确定性Radar，队内即时共享、无记忆。Actor68D局部固定slot，Critic60D真实全局。

武器1000m、ATA半角30°、target aspect半角45°，自动、同步、最近合法目标且victim去重。50m swept collision只计每步每机安全事件、不致死、可同step weapon；OOB仍死亡。Reward保持+10 kill、-10 weapon loss、-5每机collision event、-20 OOB、terminal±20/0，不含shaping。

边界控制完全保留：当前速度/heading名义预测下一步越界时，仅朝反向heading转向，仍受10°/s限制，可能OOB；不加margin、中心返回或安全层。

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
# Blue配置字段清理后的短接口smoke，不代表学习性训练
uav2d-train --config configs/smoke.yaml --output runs/v2_2_rear_pursuit/smoke
python scripts/sanity_check.py --mode all --episodes 100 \
  --seed-start 200000 --output runs/v2_2_rear_pursuit/sanity.json
python scripts/behavioral_audit.py --episodes 20 \
  --output runs/v2_2_rear_pursuit/behavioral_audit.json
```

四模式：rule-vs-rule、random-vs-rule、passive-vs-rule、direct-pursuit-vs-rule。最后一项仅重命名报告标签，Red simple direct-pursuit函数保持原样。

采样仍为16个独立持久进程、learner默认CUDA、rollout128，所有MAPPO/PPO设置不变。smoke沿用CPU/2workers/64宽配置；不是正式128宽网络。配置的默认输出目录命名保留V2.1，后续实验必须用--output指定新目录以区分Blue版本；不要覆盖历史数据。

uav2d-evaluate仍支持--config、--checkpoint、--episodes、--seed-start、--output和--gif；评估为确定性tanh(mean)。普通resume/init-checkpoint语义不变，无课程或旧Categorical迁移。日志保留双方alive、weapon kill/loss、OOB、collision events、windows、contact、length、五reward及八PPO指标。
