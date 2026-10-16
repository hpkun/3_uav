# V2.4 HTML 空战录像

只读运行一个Red MAPPO checkpoint对当前Greedy Blue的完整episode，将数值轨迹、CSS和JavaScript嵌入一个HTML。双击即可离线播放，不需要服务器、网络、CDN、图片帧或JSON旁文件。只新增录像工具，不改变环境、Reward、MAPPO/PPO、checkpoint或原GIF功能。

## 1. 生成录像

Windows终端先进入WSL：

```powershell
wsl -d Ubuntu
```

在WSL中执行：

```bash
source /home/hpk/anaconda3/etc/profile.d/conda.sh
conda activate uav
cd /mnt/c/Users/HPK/Desktop/3_uav
PYTHONPATH=src python scripts/replay_html.py \
  --config configs/base.yaml \
  --checkpoint runs/v2_4_mappo_2m_seed7/checkpoints/best.pt \
  --seed 3000000 \
  --device cuda \
  --output runs/v2_4_mappo_2m_seed7/replays/best_seed3000000.html
```

省略`--output`时只输出`runs/replays/<checkpoint_stem>_seed<seed>.html`。换checkpoint只改`--checkpoint`，例如换成`final.pt`；换episode只改`--seed`，同时建议改输出文件名。输出文件必须以`.html`结尾，已存在则拒绝覆盖，请选新文件名。

| 参数 | 含义 / 默认值 |
|---|---|
| `--config` | 项目配置，默认`configs/base.yaml`；网络宽度须与checkpoint相符 |
| `--checkpoint` | 必填，当前连续MAPPO checkpoint |
| `--seed` | episode和采样seed，默认3000000 |
| `--device` | cuda / cpu / auto，省略跟随配置；指定CUDA时不静默退回CPU |
| `--output` | 可选HTML输出路径 |
| `--stochastic` | 启用带seed随机采样；默认不启用 |
| `--help` | 查看帮助 |

默认使用与`evaluate.py`相同的`policy.policy_actions(..., deterministic=True)`和active mask，输出tanh(mean)，不是另写一套控制器。Blue始终为当前`RuleBasedOpponent()`。

`--stochastic`从同一Actor分布采样；脚本显式设置Python、NumPy、Torch及CUDA seed。同checkpoint/配置/seed/模式在相同软件与设备条件下可复现；不承诺不同设备、不同PyTorch版本的逐bit一致。随机录像和正式确定性评估要分别标注，不能混为同一实验。

## 2. 浏览器控制与战场

- 固定10km方形战场，x/y均为[-5000,5000]m，等比例；世界坐标y向上。
- 红色R0–R4、蓝色B0–B4；箭头实时按heading旋转。
- 轨迹只绘制step 0至当前步，死亡端点记录一次，随后停止。死亡位置显示weapon `×`或`OOB`，不把坐标钳回地图内。
- Play/Pause、上一帧、下一帧、Timeline；速度0.25x/0.5x/1x/2x/4x/8x，1x表示每dt秒推进一帧。Space切换播放，左右键逐帧；输入控件获得焦点时保留原键盘行为。
- Show trails开关。Selected UAV支持None/R0–R4/B0–B4；仅对选中的存活飞机显示Radar绿色虚线圈和Attack橙色虚线圈。
- 圆圈半径读取录像配置（默认3000m/1000m），只是距离示意：攻击圆不代表ATA/aspect已合法，也不改变探测或武器判定。
- 支持浏览器缩放与窄屏布局、系统明暗主题；不需要自动打开浏览器。

## 3. 状态面板和时间对齐

顶部显示checkpoint文件、从payload读取的global_step、seed、模式、V2.4、已记录episode最终result和length。不会根据文件名猜训练步数；缺失global_step显示unknown。result以Red为口径，win/loss不代表Blue视角。

右侧显示当前步与时间、双方alive，以及真实info中的累计weapon kills/losses、OOB、endpoint collision events、attack windows和Radar contact steps。当前帧显示Red本步reward、五项分量和累计return；没有自行定义新Reward。

时间约定非常重要：

- `frames[t]`是状态`s_t`，step=0是初始状态。
- 当前帧的reward、死亡事件和增量，属于进入该帧的`t-1 → t`；初始帧为零。
- 当前帧的Critic `V(s_t)`、Actor实际动作、pre-tanh mean属于即将执行的`t → t+1`。
- 终局帧没有下一步动作或pre-action value，显式记录null并显示N/A；没有伪造bootstrap value。
- Selected Red Agent独立选择R0–R4，显示a_norm、omega_norm、mu_accel、mu_turn、std_accel、std_turn；死亡机显示inactive。dead动作在实际命令数组中为0，mu为null，不能把它们当有效决策。
- std读取Actor分布，是全局可学习参数，非state-dependent；同时显示checkpoint原始log_std及clamp后的有效log_std。

录像总帧数严格为episode_length+1。完整trace内嵌`<script type="application/json">`，包含metadata、checkpoint路径/实际global_step/SHA256、完整配置、initial_state、frames、episode_summary。没有额外JSON输出；如需科研复查，可直接读取HTML内的这段JSON。

## 4. 事件身份与碰撞

Event Log按时间显示至当前步的真实weapon kill/loss、OOB、collision participant和attack-window增量，并列出死亡/碰撞飞机ID；不提前显示未来事件。战场在事件帧显示轻量击毁/碰撞圆圈。

死亡分类不猜ID：old_alive且移动后越界是OOB；old_alive、未越界而step后dead是weapon。当前碰撞用old_alive和移动后坐标重建post-OOB/pre-weapon集合，再按endpoint≤50m及unique participants识别；因此本步稍后被武器击毁的飞机仍计入碰撞。每步将诊断身份数与实际info增量断言核对，不影响env行为。

attack_window是合法目标窗口累计，不是发射次数；本工具不把窗口数标作命中率，也不猜击杀攻击者身份。

## 5. 只读边界

`replay_html.py`只加载checkpoint，不调用update/save/训练。源码、配置和checkpoint执行前后SHA256核对；不修改`src/uav_mappo_2d/`、`configs/base.yaml`、`rendering.py`、`evaluate.py`、旧checkpoint或训练CSV。不新增依赖，不调用GIF renderer、不编码PNG/base64帧。

Python脚本读取项目内的`scripts/replay_template.html`作为模板，因此移动工具时保留这两个源文件。生成后的HTML已包含模板及全部数据，移动/拷贝它时不需要带模板或项目文件。

## 6. 本轮验证

在WSL Ubuntu、conda uav下执行完整pytest，最终125 passed in 5.44s。涵盖坐标、自包含/无CDN、帧序列、死亡分类、endpoint参与者、轨迹终止、deterministic与seeded stochastic复现、正式evaluate语义一致、有限Actor/Critic、控制项、checkpoint payload元数据及拒绝覆盖测试。测试使用人工数据和临时未训练小网络，不依赖真实2M文件。JavaScript语法及播放/逐帧/时间轴/8倍速/键盘/范围圈/死亡轨迹逻辑也经Node运行检查，未打开浏览器。

实际录像（不是策略审计或挑选成功案例）：

| 项目 | 结果 |
|---|---|
| checkpoint | runs/v2_4_mappo_2m_seed7/checkpoints/best.pt |
| payload global_step | 1800192 |
| seed / device / mode | 3000000 / CUDA / deterministic |
| 输出 | runs/v2_4_mappo_2m_seed7/replays/best_seed3000000.html |
| HTML大小 | 287602 bytes（约281 KiB） |
| episode length / frames | 116 / 117 |
| Red result / return | loss / -70.4 |
| weapon kills R/B | 0 / 5 |
| OOB R/B | 0 / 0 |

这是单个seed的录像结果，不用于宣称收敛、泛化、优于或总体胜率。本轮不实现RL审计、不重新训练，也不自动打开浏览器。
