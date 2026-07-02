#!/usr/bin/env python3
from pathlib import Path

TASK_NAME = "fanfan_rouhe_straight"
INIT = Path(__file__).resolve().parent / "__init__.py"

REG_BLOCK = '\n\n# --- fanfan_rouhe_straight: straight-walk correction task ---\nfrom legged_gym.envs.fanfan_rouhe_straight.fanfan_config import (\n    FanfanRouheStraightCfg,\n    FanfanRouheStraightCfgPPO,\n)\nfrom legged_gym.envs.fanfan_rouhe_straight.fanfan_env import FanfanRouheStraightRobot\n\ntask_registry.register(\n    "fanfan_rouhe_straight",\n    FanfanRouheStraightRobot,\n    FanfanRouheStraightCfg(),\n    FanfanRouheStraightCfgPPO(),\n)\n'

if not INIT.exists():
    raise SystemExit(f"找不到 {INIT}，请把本脚本放在 legged_gym/envs 目录下运行")

text = INIT.read_text(encoding="utf-8")
if f'task_registry.register(\n    "{TASK_NAME}"' in text or f'task_registry.register("{TASK_NAME}"' in text:
    print(f"{TASK_NAME} 已经注册过，不重复修改 __init__.py")
else:
    INIT.write_text(text.rstrip() + REG_BLOCK + "\n", encoding="utf-8")
    print(f"已注册新任务：{TASK_NAME}")
    print(f"已修改：{INIT}")
