"""Forward-only Fanfan environment with world-frame path locking rewards."""
import torch

from legged_gym.envs.fanfan.fanfan_env import FanfanRobot


class FanfanStraightV2Robot(FanfanRobot):
    def _init_buffers(self):
        super()._init_buffers()
        self.straight_origin = self.root_states[:, :2].clone()
        self.straight_heading = self.rpy[:, 2].clone()

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if hasattr(self, "straight_origin") and len(env_ids):
            self.straight_origin[env_ids] = self.root_states[env_ids, :2]
            self.straight_heading[env_ids] = self.rpy[env_ids, 2]

    def _straight_heading_error(self):
        return torch.atan2(
            torch.sin(self.straight_heading - self.rpy[:, 2]),
            torch.cos(self.straight_heading - self.rpy[:, 2]),
        )

    def _reward_forward_progress(self):
        return torch.clamp(self.base_lin_vel[:, 0], min=0.0, max=0.30)

    def _reward_straight_cross_track(self):
        displacement = self.root_states[:, :2] - self.straight_origin
        normal = torch.stack(
            (-torch.sin(self.straight_heading), torch.cos(self.straight_heading)), dim=1
        )
        cross_track = torch.sum(displacement * normal, dim=1)
        return torch.square(cross_track)

    def _reward_straight_heading_error(self):
        return torch.square(self._straight_heading_error())
