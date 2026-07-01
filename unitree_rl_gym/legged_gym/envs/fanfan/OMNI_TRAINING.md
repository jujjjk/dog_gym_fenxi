# Fanfan 单策略全向训练与部署

## A. 当前基线审计

- 前进基线为52维观测：机身线速度3、角速度3、重力3、`vx/vy/yaw`命令3、
  关节位置12、关节速度12、上一动作12、步态相位2、heading上下文2。
- 原基线命令只有正 `vx`，`vy=0`；`backward_velocity=-10`、
  `lateral_velocity=-2` 会主动排斥后退和横移。
- 基类 `_reward_tracking_lin_vel` 已同时跟踪X/Y；command scale也同时包含X/Y。
- 原基线使用absolute-heading模式，不等价于ROS `/cmd_vel.angular.z`。V1–V4均关闭
  heading模式并直接采样/输入yaw-rate。
- 12维动作经过`tanh`，再按hip/front/rear缩放并叠加步态参考；PD为
  hip `60/0.6`、thigh/calf `70/0.8`，URDF限扭17 Nm，策略频率50 Hz。
- 通用`deploy_real.py`是47维TorchScript人形部署器，不兼容fanfan 52维ONNX，
  也没有ROS2 `/cmd_vel`。

适合一个ONNX。主流legged locomotion使用速度条件策略；低速前后、横移、yaw是同一
动力学技能空间，拆模型会引入切换冲击和状态管理。只有在高速特殊步态或硬件模式完全
不同且单策略验证失败时才考虑拆分。

## B. V1–V4训练路线

四个任务均保持52维和同一网络结构，可逐级resume。下列`max_iterations`是本次追加量。

### V1 前进 + 直接yaw-rate转弯

范围：`vx=[0.05,0.28], vy=0, yaw=[-0.5,0.5]`。增强angular tracking，关闭
heading reward，把无条件yaw-rate惩罚降至轻量。由现有最佳checkpoint启动：

```bash
python legged_gym/scripts/train.py --task=fanfan_omni_v1 --num_envs=1024 \
  --resume --load_run Jun27_18-41-11_ --checkpoint 850 \
  --max_iterations 300 --run_name omni_v1 --headless
```

### V2 增加低速后退

范围：`vx=[-0.08,0.28], vy=0, yaw=[-0.5,0.5]`，`backward_velocity=0`。

```bash
python legged_gym/scripts/train.py --task=fanfan_omni_v2 --num_envs=1024 \
  --resume --load_run <V1_RUN> --checkpoint <V1_CKPT> \
  --max_iterations 300 --run_name omni_v2 --headless
```

### V3 增加小幅横移

范围：`vx=[-0.08,0.25], vy=[-0.04,0.04], yaw=[-0.5,0.5]`，
`lateral_velocity=0`；XY联合tracking保持启用。

```bash
python legged_gym/scripts/train.py --task=fanfan_omni_v3 --num_envs=1024 \
  --resume --load_run <V2_RUN> --checkpoint <V2_CKPT> \
  --max_iterations 400 --run_name omni_v3 --headless
```

### V4 低速全向整合

最终范围：`vx=[-0.12,0.30], vy=[-0.10,0.10], yaw=[-0.7,0.7]`。每次resume后
按300/600 iteration分三段从V3范围逐步扩展，避免初期激进组合。

```bash
python legged_gym/scripts/train.py --task=fanfan_omni_v4 --num_envs=1024 \
  --resume --load_run <V3_RUN> --checkpoint <V3_CKPT> \
  --max_iterations 900 --run_name omni_v4 --headless
```

随机命令play：

```bash
python legged_gym/scripts/play.py --task=fanfan_omni_v4 \
  --load_run <V4_RUN> --checkpoint <V4_CKPT> --num_envs=64
```

## C. 导出和Sim2Sim

```bash
PYTHONPATH=/home/nszb/gym/mujoko/export_deps \
/home/nszb/gym/unitree-rl/bin/python /home/nszb/gym/mujoko/export_onnx.py \
  logs/rough_fanfan/<V4_RUN>/model_<V4_CKPT>.pt \
  /home/nszb/gym/mujoko/models/fanfan_omni.onnx \
  --gym-root /home/nszb/gym/unitree_rl_gym \
  --config-class legged_gym.envs.fanfan.fanfan_omni_config:FanfanOmniV4Cfg

/home/nszb/gym/mujoko/.venv/bin/python /home/nszb/gym/mujoko/prepare_model.py \
  /home/nszb/gym/mujoko/assets/fanfan.urdf \
  /home/nszb/gym/mujoko/models/fanfan_scene.xml \
  --policy /home/nszb/gym/mujoko/models/fanfan_omni.onnx

/home/nszb/gym/mujoko/.venv/bin/python /home/nszb/gym/mujoko/test_omni_matrix.py
```

V2重点命令：`--command -0.04 0 0`、`-0.08 0 0`、`-0.08 0 0.2`。
V3重点命令：`0 0.03 0`、`0 -0.03 0`、`0.08 0.03 0`、`0.08 -0.03 0`。
V4矩阵脚本覆盖前进、后退、左右横移、左右原地转、前进左右转、后退转弯和斜移。

## D. ROS2安全部署

`deploy/deploy_real/fanfan_omni_ros2.py`订阅：

- `/cmd_vel` (`Twist`)
- `/joint_states`、`/imu/data`、`/odom`
- `/fanfan/stand_ready`、`/fanfan/emergency_stop`

并输出`/fanfan/policy_joint_targets`。RS01底层适配器必须按metadata关节顺序消费该目标，
节点本身不猜测电机ID或总线协议。初期限幅为`vx[-0.06,0.18]`、
`vy[-0.04,0.04]`、`yaw[-0.35,0.35]`；每20 ms斜坡分别为0.01、0.005、0.03。
保护包括反馈超时拒绝启动、stand-ready、急停锁存、roll/pitch、目标变化率、q error和
基于PD的raw torque guard。

推荐实机顺序：站立零命令 → 前进0.05 → yaw 0.1 → 后退-0.03 → 横移0.02 →
前进转弯 → 最后才测试后退转弯及斜移。每次CSV保存command、实测速度、roll/pitch、
关节q/dq/target、raw/applied torque、保护状态、推理延迟和反馈时间戳。

## E. 最终产物

最终只导出`fanfan_omni.onnx`。部署端将`Twist.linear.x/y`和`angular.z`直接放入
观测的三个command槽位；观测仍是52维，因为V1–V4保留网络兼容性，direct-yaw模式下
最后两维使用中性heading编码`[0,1]`。
