"""View a fixed vx/vy/yaw command in the Isaac Gym training environment."""
import sys
import time


def _pop_custom_args(argv):
    command = [0.0, 0.0, 0.0]
    duration = 120.0
    for flag, count in (("--command", 3), ("--duration", 1)):
        if flag in argv:
            index = argv.index(flag)
            values = argv[index + 1:index + 1 + count]
            if len(values) != count:
                raise SystemExit(f"{flag} requires {count} value(s)")
            del argv[index:index + 1 + count]
            if flag == "--command":
                command = [float(value) for value in values]
            else:
                duration = float(values[0])
    return command, duration


COMMAND, DURATION = _pop_custom_args(sys.argv)

# Isaac Gym must be imported before torch in this project.
import isaacgym
import torch

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry


def mirrored_negative_lateral_policy(policy, observations):
    """Match the symmetry wrapper embedded in the deployed omni ONNX."""
    observations = observations.clone()
    vx = observations[:, 9:10] / 2.0
    vy = observations[:, 10:11] / 2.0
    pure_lateral = torch.exp(-torch.square(vx / 0.03))
    lateral_activity = 1.0 - torch.exp(-torch.square(vy / 0.02))
    compensation = torch.clamp(0.07 - 0.45 * torch.abs(vy), 0.02, 0.06)
    observations[:, 9:10] += 2.0 * compensation * pure_lateral * lateral_activity
    if COMMAND[1] >= 0.0:
        return policy(observations)
    obs_index = list(range(52)); obs_sign = [1.0] * 52
    for index in (1, 3, 5, 7, 10, 11, 48, 49, 50):
        obs_sign[index] = -1.0
    leg_map = (1, 0, 3, 2)
    for start in (12, 24, 36):
        for dst_leg, src_leg in enumerate(leg_map):
            for joint in range(3):
                obs_index[start + dst_leg * 3 + joint] = start + src_leg * 3 + joint
            obs_sign[start + dst_leg * 3] = -1.0
    mirrored_obs = observations[:, obs_index] * torch.tensor(
        obs_sign, device=observations.device
    )
    mirrored_actions = policy(mirrored_obs)
    out_index = []
    out_sign = []
    for src_leg in leg_map:
        out_index.extend((src_leg * 3, src_leg * 3 + 1, src_leg * 3 + 2))
        out_sign.extend((-1.0, 1.0, 1.0))
    return mirrored_actions[:, out_index] * torch.tensor(
        out_sign, device=observations.device
    )


def main():
    args = get_args()
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = min(args.num_envs or 16, 16)
    env_cfg.terrain.num_rows = 2
    env_cfg.terrain.num_cols = 2
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_motor_strength = False

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    train_cfg.runner.resume = True
    runner, _ = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    policy = runner.get_inference_policy(device=env.device)
    command = torch.tensor(COMMAND, device=env.device)

    steps = int(DURATION / env.dt)
    for _ in range(steps):
        started = time.time()
        env.commands[:, :3] = command
        env.compute_observations()
        actions = mirrored_negative_lateral_policy(
            policy, env.get_observations().detach()
        )
        env.step(actions.detach())
        time.sleep(max(0.0, env.dt - (time.time() - started)))


if __name__ == "__main__":
    main()
