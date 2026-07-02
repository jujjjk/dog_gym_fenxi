from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

from legged_gym.envs.go2.go2_config import GO2RoughCfg, GO2RoughCfgPPO
from legged_gym.envs.fanfan.fanfan_config import FanfanRoughCfg, FanfanRoughCfgPPO
from legged_gym.envs.fanfan.fanfan_env import FanfanRobot
from legged_gym.envs.fanfan.fanfan_omni_config import (
    FanfanOmniV1Cfg, FanfanOmniV1CfgPPO,
    FanfanOmniV2Cfg, FanfanOmniV2CfgPPO,
    FanfanOmniV3Cfg, FanfanOmniV3CfgPPO,
    FanfanOmniV4Cfg, FanfanOmniV4CfgPPO,
)
from legged_gym.envs.fanfan_rouhe.fanfan_config import FanfanRouheRoughCfg, FanfanRouheRoughCfgPPO
from legged_gym.envs.fanfan_rouhe.fanfan_env import FanfanRouheRobot
from legged_gym.envs.h1.h1_config import H1RoughCfg, H1RoughCfgPPO
from legged_gym.envs.h1.h1_env import H1Robot
from legged_gym.envs.h1_2.h1_2_config import H1_2RoughCfg, H1_2RoughCfgPPO
from legged_gym.envs.h1_2.h1_2_env import H1_2Robot
from legged_gym.envs.g1.g1_config import G1RoughCfg, G1RoughCfgPPO
from legged_gym.envs.g1.g1_env import G1Robot
from .base.legged_robot import LeggedRobot

from legged_gym.utils.task_registry import task_registry

task_registry.register( "go2", LeggedRobot, GO2RoughCfg(), GO2RoughCfgPPO())
task_registry.register( "fanfan", FanfanRobot, FanfanRoughCfg(), FanfanRoughCfgPPO())
task_registry.register("fanfan_omni_v1", FanfanRobot, FanfanOmniV1Cfg(), FanfanOmniV1CfgPPO())
task_registry.register("fanfan_omni_v2", FanfanRobot, FanfanOmniV2Cfg(), FanfanOmniV2CfgPPO())
task_registry.register("fanfan_omni_v3", FanfanRobot, FanfanOmniV3Cfg(), FanfanOmniV3CfgPPO())
task_registry.register("fanfan_omni_v4", FanfanRobot, FanfanOmniV4Cfg(), FanfanOmniV4CfgPPO())
task_registry.register( "fanfan_rouhe", FanfanRouheRobot, FanfanRouheRoughCfg(), FanfanRouheRoughCfgPPO())
task_registry.register( "h1", H1Robot, H1RoughCfg(), H1RoughCfgPPO())
task_registry.register( "h1_2", H1_2Robot, H1_2RoughCfg(), H1_2RoughCfgPPO())
task_registry.register( "g1", G1Robot, G1RoughCfg(), G1RoughCfgPPO())

# --- fanfan_rouhe_straight: straight-walk correction task ---
from legged_gym.envs.fanfan_rouhe_straight.fanfan_config import (
    FanfanRouheStraightCfg,
    FanfanRouheStraightCfgPPO,
)
from legged_gym.envs.fanfan_rouhe_straight.fanfan_env import FanfanRouheStraightRobot

task_registry.register(
    "fanfan_rouhe_straight",
    FanfanRouheStraightRobot,
    FanfanRouheStraightCfg(),
    FanfanRouheStraightCfgPPO(),
)

# Forward-only V2: initialized from a proven gait and trained with path locking.
from legged_gym.envs.fanfan.fanfan_straight_v2_config import (
    FanfanStraightV2Cfg, FanfanStraightV2CfgPPO,
)
from legged_gym.envs.fanfan.fanfan_straight_v2_env import FanfanStraightV2Robot
task_registry.register(
    "fanfan_straight_v2", FanfanStraightV2Robot,
    FanfanStraightV2Cfg(), FanfanStraightV2CfgPPO(),
)
