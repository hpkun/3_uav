# V2.4 独立 Vanilla MADDPG Baseline

## 1. 范围与结果

本轮新增独立MADDPG，不改名/复用MAPPO学习器，不共享Actor/Critic，不实现单team Critic。冻结的V2.4环境、Greedy Blue、Radar/Weapon/OOB/Reward/endpoint collision、68D局部观测、60D state和原MAPPO均不变。

已完成算法、专项测试和2560步CUDA/16进程smoke，真实24次MADDPG update。没有运行正式2M、参数搜索或性能优化实验。全部原有源码/配置/pyproject的前后SHA256一致，证据见runs/v2_4_maddpg_smoke_seed7/frozen_scope.json。

## 2. 参考原文与官方代码

- [Lowe et al., 2017, Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275)：§4.1、式(5)–(6)、附录Algorithm 1、实验参数。依据为独立局部Actor、各agent centralized Q、replay、deterministic policy gradient、target networks和soft update；两层64-unit ReLU及参考超参数也有原文/配套实现依据。
- [OpenAI trainer/maddpg.py](https://github.com/openai/maddpg/blob/master/maddpg/trainer/maddpg.py)：Q MSE、仅替换当前agent action的policy loss、1e−3正则项、1e6 replay、batch_size*max_episode_len预热、每100步更新、polyak=.99。
- [OpenAI experiments/train.py](https://github.com/openai/maddpg/blob/master/experiments/train.py)：两层64-unit MLP、Adam lr=.01、gamma=.95、batch1024。
- [OpenAI tf_util.py](https://github.com/openai/maddpg/blob/master/maddpg/common/tf_util.py)：minimize_and_clip逐参数张量clip_by_norm，而非整个网络全局norm裁剪。
- [OpenAI distributions.py](https://github.com/openai/maddpg/blob/master/maddpg/common/distributions.py)：官方有动作分布封装，因此本实现的deterministic bounded tanh/外置OU并非对TF代码逐行复现。

特别区分：官方对某个agent更新时，以同一sample index从各buffer取对齐joint transition；不同agent各自update可以抽不同batch。本实现按本轮要求，一轮五个agent共用一个joint batch，并先更新全部critics再更新全部actors。这是明确的更新组织适配，不声称与官方执行顺序/随机batch耦合逐项完全等同。

## 3. 文件

| 新增文件 | 职责 |
|---|---|
| src/uav_mappo_2d/maddpg_models.py | Actor/Critic独立MLP |
| src/uav_mappo_2d/maddpg.py | 五agent学习器、joint ring replay、OU、target/optimizer/save/load |
| src/uav_mappo_2d/maddpg_config.py | 算法独立配置与校验 |
| src/uav_mappo_2d/train_maddpg.py | 复用原真实进程环境，独立训练/CSV/TensorBoard/checkpoint |
| src/uav_mappo_2d/evaluate_maddpg.py | 独立加载MADDPG，确定性评估 |
| configs/maddpg.yaml | official-reference算法profile，不复制环境配置 |
| scripts/smoke_maddpg.py | 短程真实学习/存取/target公式验证 |
| tests/test_maddpg.py | 专项数学/参数隔离/采样/调度/终止等测试 |
| docs/maddpg_baseline.md | 本文档 |

README仅增加入口。不改env.py/opponents.py/parallel_env.py/config.py/mappo.py/models.py/train.py/evaluate.py/rendering.py、base.yaml、smoke.yaml或pyproject.toml。

复用的只是原ParallelUAVEnvs和evaluate_policy的环境rollout/指标聚合。MADDPG提供相同的deterministic policy_actions接口，评估helper没有实例化MAPPO网络，也不改变其实现/语义。没有为审计或MADDPG增加生产环境/原MAPPO API。

## 4. 网络与参数量

| 类型 | 数量 | 每个结构 | 每个参数量 | 本类合计 |
|---|---:|---|---:|---:|
| Red Actor | 5 | 68→64 ReLU→64 ReLU→2 tanh | 8706 | 43530 |
| Red Critic | 5 | 70→64 ReLU→64 ReLU→1 | 8769 | 43845 |
| target Actor | 5 | 与对应Actor相同 | 8706 | 43530 |
| target Critic | 5 | 与对应Critic相同 | 8769 | 43845 |

Online合计87375；含targets总计174750。每个网络单独对象、独立parameter storage；每个online网络独立Adam optimizer。初始化Xavier uniform、bias0；target初始化为对应online exact copy，target requires_grad=False。

Actor i只读取自己的68D o_i，没有global state/其他agent完整obs/额外Blue hidden input。

Critic i读取60D global_state与5个Red动作共10D，即70D。state已包含10机物理状态；Blue固定无状态Greedy，由状态决定动作，不单独拼接Blue action。这是当前任务的CTDE implementation adaptation，而不是声称官方MPE恰有60D/70D接口。

## 5. Joint Replay

每个transition保存：
```text
observations       [5,68]
states             [60]
actions            [5,2]
rewards            scalar Red team reward
next_observations  [5,68]
next_states        [60]
dones              scalar episode done
active_masks       [5]
next_active_masks  [5]
```

一个ring buffer，默认capacity1,000,000；uniform with replacement采样，所有字段共用同一个索引向量，不会跨agent错配；没有PER/n-step。batch_size=1024指joint transitions，不是把1024个完整300步episode拼进一个minibatch。

Float32数组总容量约3.06GiB（不含进程/运行时开销）；smoke容量单独缩小。reward不乘5，保持shared team scalar。Replay float32与现有ParallelUAVEnvs的float32 reward逐值一致；episode精确核账用环境info中的真实分量，不能把float32存储舍入误差当作改Reward。

Joint buffer替代对齐多buffer是存储适配；一轮所有agent共用batch是本轮明确选择，参见第2节差异说明。

## 6. 损失、mask、target与更新顺序

记m_i为当前active，m'_i为next active，d为episode done，r为当前Red shared team reward。

目标动作：
```text
a'_j = target_actor_j(o'_j) * next_active_j
y_i  = stop_gradient[r + gamma*(1-done)*next_active_i
                    * target_critic_i(s', a'_joint)]
```

Critic i仅在m_i=1样本上计算MSE：
```text
L_Q_i = mean_active_i((Q_i(s, replay_joint_actions) - y_i)^2)
```

Actor i将replay_joint_actions.detach().clone()中的第i项替换为当前actor_i(o_i)，其它四项仍为replay实际动作并detach：
```text
L_actor_i = -mean_active_i(Q_i(s, a_replay_except_i, actor_i(o_i)))
            + 0.001*mean_active_i(actor_i(o_i)^2)
```

actor阶段冻结critic参数requires_grad，但保留Q对action的梯度，结束恢复。没有current-active样本时安全skip该agent的actor/critic优化，CSV的agent_i_updated=0明确标识；对应loss数值0只作空值占位，不应解释为完美拟合。

一次共同batch：先算全部target → 更新5 critics → 更新5 actors → soft-update全部targets：
```text
target_new = (1-tau)*target_old + tau*online_new
tau = 0.01
```

当前已死亡：最终action=[0,0]，不参与自己的actor/critic loss。当前transition中死亡：仍得到r，但next_active_i=0禁止未来bootstrap。episode done时即使vector返回下一episode reset obs/state/mask，也由(1-done)=0截断；target数值不依赖新episode的Q。

梯度裁剪按官方TF逐参数张量norm≤0.5实现；日志mean_*_grad_norm是裁剪前整个网络总norm，因此可大于0.5，不代表单张量裁剪失效。没有误称为PyTorch全网络global-norm≤0.5。

官方regularizer作用于其动作分布flat parameters，本实现按本轮要求作用于bounded deterministic action平方；这是动作接口适配，不声称两者数值等价。

## 7. OU探索

训练：clip(actor_i(o_i)+OU_noise,−1,1)，再乘active mask，死机最终仍为0。Evaluation只用Actor，不加OU。

OU state shape=[parallel_envs,5,2]，每个环境/agent/dimension独立增量，dt=1：
```text
x <- x + theta*(mu-x) + sigma*N(0,1)
```

默认mu=0、theta=.15、sigma=.20，均可配置；这些是本bounded-action PyTorch实现的探索参数，不称为官方TF默认。episode reset只重置相应environment的OU state。没有噪声退火、随机warmup策略或额外战术。

## 8. Official-reference profile与并行计步

| 参数 | 正式profile |
|---|---:|
| hidden_size | 64 |
| actor/critic learning rate | .01 / .01 |
| gamma | .95 |
| tau | .01（polyak=.99） |
| buffer_capacity | 1,000,000 |
| batch_size | 1024 |
| update_interval_steps | 100 individual transitions |
| max_grad_norm | .5，逐参数张量 |
| actor_regularization | .001 |
| learning_starts | null → batch_size*episode_steps=307200 |

algorithm YAML只定义以上算法/探索项。环境/Reward始终读取base.yaml。warmup不是307200个vector loop，也不是每个agent各存307200次。

复用原16个持久独立环境进程，global_step每次+=16。threshold=100时实际跨过点为112、208、304…；不能写成每100个vector loop才update。一次跨过多个threshold则补齐次数；未预热时消耗/跳过历史threshold，不在warmup结束后积压几千次补更新。正式profile第一次在global_step307200满足门槛，之前loss标注warmup/N/A，replay持续增长属于预期。

## 9. 训练、评估与checkpoint

正式命令如下，仅提供，本轮没有执行：
```bash
# Windows先 wsl -d Ubuntu；以下在WSL项目目录
source /home/hpk/anaconda3/etc/profile.d/conda.sh
conda activate uav
cd /mnt/c/Users/HPK/Desktop/3_uav
PYTHONPATH=src python -m uav_mappo_2d.train_maddpg \
  --config configs/base.yaml --algo-config configs/maddpg.yaml \
  --steps 2000000 --output runs/v2_4_maddpg_2m_seed7 \
  --device cuda --seed 7
```

新训练拒绝使用非空output目录，避免追加旧实验。没有实现假resume：checkpoint不保存replay、OU状态或完整环境/RNG，因此本轮不提供断点精确恢复CLI。

Checkpoint包含algorithm='maddpg'、global_step、update_count、5套online/target Actor/Critic、10个optimizer状态、MADDPG config、environment/reward/render快照、best evaluation metadata。load显式拒绝MAPPO格式，网络和算法配置须匹配。best.pt、step_xxxxxxxxx.pt、final.pt分别保存；best按与MAPPO完全一致的(win_rate,survival_difference)字典序选择。达到训练最后步会再做一次确定性evaluation并保存final，短程运行也能得到best。

评估命令：
```bash
PYTHONPATH=src python -m uav_mappo_2d.evaluate_maddpg \
  --config configs/base.yaml \
  --checkpoint runs/v2_4_maddpg_2m_seed7/checkpoints/best.pt \
  --episodes 500 --seed-start 3000000 --device cuda \
  --output runs/v2_4_maddpg_2m_seed7/evaluation_holdout.json
```

eval算法配置从MADDPG checkpoint读取，Actor无噪声，Blue仍Greedy。训练内best选择用seed+1000000，与MAPPO相同；正式holdout用新seed区间。指标复用当前MAPPO evaluate的原环境rollout聚合，因此return、W/D/L、alive、kills/losses、OOB、collision、windows、contact、length、death share等口径一致。这里并未增加value/PPO等不适用于MADDPG的指标。

当前replay_html.py专为MAPPO设计，不兼容MADDPG checkpoint；本轮不为此改动旧录像或MAPPO接口。

## 10. 日志

episodes.csv字段与MAPPO保持一致。updates.csv每实际update一行，保存global_step、update编号、replay_size、noise_sigma、mean_actor_loss、mean_critic_loss、mean_q、mean_target_q、mean_td_abs、mean_actor_grad_norm、mean_critic_grad_norm，以及每agent actor/critic loss和updated标志。

mean_q为critic优化前的replay Q_i均值；mean_target_q明确指TD目标y_i均值，不是单独的raw future Q'_i。两者均对各agent有效样本分别均值后，再跨实际参与更新的agent平均；不是所有agent/sample混在一起的加权均值。

终端显示step/总步数与百分比、replay、update数、return、双方行为、Reward分解、length、算法损失/Q/TD/梯度，W/D/L不是唯一指标。未完成任何episode时episode_metrics明确标注N/A，而不是把空窗口0均值误认为真实存活数。

TensorBoard记录episode/*、evaluation/*及maddpg/actor_loss_mean、critic_loss_mean、q_mean、target_q_mean、td_abs_mean、actor_grad_norm、critic_grad_norm、replay_size、noise_sigma和每agent loss/updated。

## 11. 测试

最终pytest结果：150 passed in 5.91s（包含原125项测试）。专项覆盖：

- 20个网络对象和所有parameter storage独立；每个target初始exact copy。
- 更新Actor0只改变Actor0，有真实dQ/da梯度，Critic及其它Actor不变。
- 更新Critic0只改变Critic0；Actor阶段critic梯度冻结并恢复。
- 当前active mask、全dead安全skip、target joint dead动作0、done/death禁止bootstrap。
- 一轮targets→5critics→5actors→soft顺序，同一个joint batch。
- Actor loss正则数学、TD人工期望、逐tensor tau公式、逐参数0.5裁剪。
- ring overwrite、共同索引的全部transition字段对齐、sample副本。
- OU独立/可复现/环境选择性reset，动作有界、dead0、只用自身obs。
- per-transition调度、跨threshold补齐、官方warmup不积压旧更新。
- 真实ParallelUAVEnvs autoreset terminal transition以及新episode obs不bootstrap。
- Save/load全部网络/optimizer、deterministic输出及global_step一致、拒绝MAPPO。
- 独立训练CSV/多种checkpoint/TensorBoard/评估链路，best规则与MAPPO一致。

## 12. 实际2560步smoke

命令：
```bash
PYTHONPATH=src python scripts/smoke_maddpg.py
```

正式环境全部保留（300步、V2.4全部Reward与物理），16真实worker/CUDA。算法仅临时改batch_size32、learning_starts128、buffer_capacity4096；hidden64、lr .01、gamma .95、tau .01、update_interval100、OU参数均保持参考值。短训日志/保存/评估频率缩小只为检查接口，不写回正式配置。

实际global_step2560、replay2560、9个完整训练episode、24次MADDPG update；首个update在208，每个agent各发生24次actor/critic优化。动作全部合法，max|action|=1；奖励核账最大误差0。16个不同worker PID42534–42549。训练不足完整episode的在途transition也进入replay，故2560不等于9个已完成episode步数之和。

| 跨agent平均指标 | Min | Max |
|---|---:|---:|
| mean_actor_loss | -0.229527 | 0.924363 |
| mean_critic_loss | 0.006463 | 29.916102 |
| mean_q | -0.943828 | 0.211702 |
| mean_target_q | -1.819699 | 0.182519 |
| mean_td_abs | 0.06112 | 2.496347 |
| mean_actor_grad_norm | 0.005641 | 0.269176 |
| mean_critic_grad_norm | 0.232042 | 14.023591 |

| Agent | Actor loss min/max | Critic loss min/max | Actor/Critic update次数 |
|---|---:|---:|---:|
| 0 | -0.736408 / 1.581413 | 0.012685 / 42.603065 | 24 / 24 |
| 1 | -0.671175 / 0.825824 | 0.003556 / 65.265167 | 24 / 24 |
| 2 | -0.780912 / 1.555495 | 0.006636 / 34.624485 | 24 / 24 |
| 3 | -1.512776 / 0.46535 | 0.004656 / 29.900459 | 24 / 24 |
| 4 | -0.092684 / 1.275234 | 0.004783 / 37.079456 | 24 / 24 |

实际初始到final的参数L2变化：

| Agent | Actor | Critic | target Actor | target Critic |
|---|---:|---:|---:|---:|
| 0 | 5.576597 | 4.353097 | 0.83448 | 0.597181 |
| 1 | 5.802644 | 4.512106 | 0.92082 | 0.621218 |
| 2 | 6.083739 | 4.165352 | 0.800593 | 0.586553 |
| 3 | 5.755468 | 4.808696 | 0.774021 | 0.659298 |
| 4 | 5.906869 | 4.689148 | 0.911329 | 0.641704 |

5个Actor、5个Critic、全部target均真实变化，非仅采样而未训练；target本次观察到较小变化不等于数学上保证其L2始终小于online。逐tensor tau公式单测通过，另在加载后的诊断副本上执行一次soft_update，对照(1−tau)*old+tau*online最大差1.4901161193847656e−8（float32），未改checkpoint文件。tau=.01。

checkpoint save/load全部online/target网络逐tensor一致，optimizer状态已保存/恢复。独立evaluate_maddpg CLI也从final.pt成功完成4局，结果与smoke中的独立holdout完全一致。

### 确定性评估联通结果

训练内在1280/2560步各做4局验证，用作best选择；载入final后又在seed3000000–3000003做4局。后者：

- return −29.5；alive R/B=2.75/4.50。
- weapon kills R/B=.50/2.25；weapon losses R/B=2.25/.50。
- OOB R/B=0/0；collision R/B=10/8.5。
- attack windows R/B=.50/2.25；contact R/B=265/265。
- mean length300；outcome W/D/L=1/0/3。

这里只证明deterministic evaluation可完整运行；没有据此评价算法质量、收敛或优于MAPPO。4个测试episode全达300步也不在本轮调整环境或策略。

## 13. 原始结果

runs/v2_4_maddpg_smoke_seed7/：
- config_snapshot.json、training_summary.json。
- episodes.csv、updates.csv、TensorBoard事件。
- checkpoints/best.pt、step_000001280.pt、step_000002560.pt、final.pt。
- smoke_validation.json、evaluation_cli.json、frozen_scope.json。

后续先检查实现再决定正式训练；本轮未运行2M。没有TD3/twin/delayed update、PER、RNN、attention、parameter sharing、communication、BC、curriculum、self-play或n-step等扩展。
