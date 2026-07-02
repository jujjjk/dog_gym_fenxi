"""Forward-only, path-locked policy initialized from a proven Fanfan gait."""
from legged_gym.envs.fanfan.fanfan_omni_config import FanfanOmniV4Cfg, FanfanOmniV4CfgPPO


class FanfanStraightV2Cfg(FanfanOmniV4Cfg):
    class commands(FanfanOmniV4Cfg.commands):
        heading_command = False
        observe_heading_error = True
        resampling_time = 20.0
        omni_curriculum = False
        pure_yaw_probability = 0.0
        stand_probability = 0.0
        pure_lateral_probability = 0.0
        pure_sagittal_probability = 1.0

        class ranges(FanfanOmniV4Cfg.commands.ranges):
            lin_vel_x = [0.10, 0.20]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [0.0, 0.0]

    class rewards(FanfanOmniV4Cfg.rewards):
        only_positive_rewards = False
        tracking_sigma = 0.01
        lateral_tracking_sigma = 0.00010
        longitudinal_tracking_sigma = 0.001

        class scales(FanfanOmniV4Cfg.rewards.scales):
            tracking_lin_vel = 10.0
            tracking_lateral_vel = 8.0
            tracking_longitudinal_vel = 8.0
            tracking_ang_vel = 4.0
            heading_tracking = 10.0
            backward_velocity = -20.0
            lateral_velocity = -20.0
            yaw_rate = -2.0
            translation_yaw_error = -10.0
            lateral_yaw_error = 0.0
            forward_progress = 8.0
            straight_cross_track = -50.0
            straight_heading_error = -20.0


class FanfanStraightV2CfgPPO(FanfanOmniV4CfgPPO):
    class algorithm(FanfanOmniV4CfgPPO.algorithm):
        entropy_coef = 0.002

    class runner(FanfanOmniV4CfgPPO.runner):
        experiment_name = "rough_fanfan_straight_v2"
        run_name = "straight_v2"
