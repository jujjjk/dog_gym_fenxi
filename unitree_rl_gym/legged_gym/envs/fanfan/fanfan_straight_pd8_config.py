"""Straight-walk continuation using the requested RS01 PD8 controller."""
from legged_gym.envs.fanfan.fanfan_straight_v2_config import (
    FanfanStraightV2Cfg,
    FanfanStraightV2CfgPPO,
)


class FanfanStraightPD8Cfg(FanfanStraightV2Cfg):
    class control(FanfanStraightV2Cfg.control):
        stiffness = {"hip": 60.0, "thigh": 70.0, "calf": 70.0}
        damping = {"hip": 1.2, "thigh": 1.6, "calf": 1.6}
        pd_position_error_limits = {
            "hip": 0.133,
            "thigh": 0.137,
            "calf": 0.183,
        }


class FanfanStraightPD8CfgPPO(FanfanStraightV2CfgPPO):
    class runner(FanfanStraightV2CfgPPO.runner):
        experiment_name = "rough_fanfan_straight_pd8"
        run_name = "straight_pd8"
