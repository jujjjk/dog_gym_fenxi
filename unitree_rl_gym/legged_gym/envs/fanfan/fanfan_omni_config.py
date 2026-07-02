"""Incremental direct velocity-command tasks for one fanfan omni policy."""
from legged_gym.envs.fanfan.fanfan_config import FanfanRoughCfg, FanfanRoughCfgPPO


class _DirectYawCfg(FanfanRoughCfg):
    class commands(FanfanRoughCfg.commands):
        heading_command = False
        observe_heading_error = True
        resampling_time = 6.0
        pure_yaw_probability = 0.20
        stand_probability = 0.10
        pure_lateral_probability = 0.0

    class rewards(FanfanRoughCfg.rewards):
        class scales(FanfanRoughCfg.rewards.scales):
            tracking_ang_vel = 4.0
            heading_tracking = 0.0
            yaw_rate = -0.20


class FanfanOmniV1Cfg(_DirectYawCfg):
    class commands(_DirectYawCfg.commands):
        class ranges(_DirectYawCfg.commands.ranges):
            lin_vel_x = [0.05, 0.28]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [-0.50, 0.50]


class FanfanOmniV2Cfg(_DirectYawCfg):
    class commands(_DirectYawCfg.commands):
        class ranges(_DirectYawCfg.commands.ranges):
            lin_vel_x = [-0.08, 0.28]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [-0.50, 0.50]

    class rewards(_DirectYawCfg.rewards):
        class scales(_DirectYawCfg.rewards.scales):
            backward_velocity = 0.0


class FanfanOmniV3Cfg(FanfanOmniV2Cfg):
    class commands(FanfanOmniV2Cfg.commands):
        pure_lateral_probability = 0.25
        pure_sagittal_probability = 0.20
        class ranges(FanfanOmniV2Cfg.commands.ranges):
            lin_vel_x = [-0.08, 0.25]
            lin_vel_y = [-0.06, 0.06]
            ang_vel_yaw = [-0.50, 0.50]

    class rewards(FanfanOmniV2Cfg.rewards):
        class scales(FanfanOmniV2Cfg.rewards.scales):
            lateral_velocity = 0.0
            tracking_lateral_vel = 4.0
            tracking_longitudinal_vel = 4.0
            lateral_hip_common_mode = 0.0
            lateral_yaw_error = -20.0
            translation_yaw_error = -10.0


class FanfanOmniV4Cfg(FanfanOmniV3Cfg):
    class control(FanfanOmniV3Cfg.control):
        hip_action_scale = 0.08
        # Balance lateral impulse: the original 0.20 rear scale made the rear
        # legs outrun the front pair and produced visible fishtailing.
        rear_action_scale = 0.17

    class commands(FanfanOmniV3Cfg.commands):
        # Give straight forward/backward commands enough replay to eliminate
        # accumulated cross-track drift without forgetting lateral motion.
        pure_sagittal_probability = 0.30
        resampling_time = 20.0
        omni_curriculum = True
        omni_curriculum_stages = [
            {"until_iteration": 100, "lin_vel_x": [-0.10, 0.28],
             "lin_vel_y": [-0.08, 0.08], "ang_vel_yaw": [-0.60, 0.60]},
            {"until_iteration": 1.0e12, "lin_vel_x": [-0.12, 0.30],
             "lin_vel_y": [-0.10, 0.10], "ang_vel_yaw": [-0.70, 0.70]},
        ]

        class ranges(FanfanOmniV3Cfg.commands.ranges):
            lin_vel_x = [-0.12, 0.30]
            lin_vel_y = [-0.10, 0.10]
            ang_vel_yaw = [-0.70, 0.70]

    class rewards(FanfanOmniV3Cfg.rewards):
        lateral_tracking_sigma = 0.00010
        class scales(FanfanOmniV3Cfg.rewards.scales):
            heading_tracking = 10.0


class _OmniPPO(FanfanRoughCfgPPO):
    class runner(FanfanRoughCfgPPO.runner):
        # Keep one experiment root so CLI resume can cross V1 -> V2 -> V3 -> V4.
        experiment_name = "rough_fanfan"


class FanfanOmniV1CfgPPO(_OmniPPO):
    pass
class FanfanOmniV2CfgPPO(_OmniPPO):
    pass
class FanfanOmniV3CfgPPO(_OmniPPO):
    pass
class FanfanOmniV4CfgPPO(_OmniPPO):
    pass
