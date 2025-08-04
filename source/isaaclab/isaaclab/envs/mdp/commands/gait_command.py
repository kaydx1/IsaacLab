

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg import GaitCommandCfg

class GaitCommand(CommandTerm):
    """Command generator that generates a velocity and gait frequency command 
    from a uniform distribution.
    """

    cfg: GaitCommandCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: GaitCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator."""
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]

        # Initialize commands and other buffers
        self.commands = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.gait_frequency = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.cmd_resample_time = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.is_standing_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.gait_process = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.dt = env.step_dt

        # Initialize metrics
        self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_gait_frequency"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "GaitCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tGait frequency dimension: 1\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tStanding probability: {self.cfg.still_proportion}"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame. Shape is (num_envs, 3)."""
        return self.commands

    def _update_metrics(self):
        """Update the metrics."""
        max_command_step = self.cfg.resampling_time_range[1] / self._env.step_dt
        # Velocity errors
        self.metrics["error_vel_xy"] += (
            torch.norm(self.commands[:, :2] - self.robot.data.root_lin_vel_b[:, :2], dim=-1) / max_command_step
        )
        self.metrics["error_vel_yaw"] += (
            torch.abs(self.commands[:, 2] - self.robot.data.root_ang_vel_b[:, 2]) / max_command_step
        )
        # Gait frequency error (assuming there's a way to measure current gait frequency)
        # If not, this can be adapted or removed. For now, let's assume a placeholder.
        # self.metrics["error_gait_frequency"] += torch.abs(self.gait_frequency - measured_gait_frequency) / max_command_step

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample commands for the given environment IDs."""
        if len(env_ids) == 0:
            return
        else:
            # Sample commands from uniform distributions
            self.commands[env_ids, 0] = torch.rand(len(env_ids), device=self.device) * (self.cfg.ranges.lin_vel_x[1] - self.cfg.ranges.lin_vel_x[0]) + self.cfg.ranges.lin_vel_x[0]
            self.commands[env_ids, 1] = torch.rand(len(env_ids), device=self.device) * (self.cfg.ranges.lin_vel_y[1] - self.cfg.ranges.lin_vel_y[0]) + self.cfg.ranges.lin_vel_y[0]
            self.commands[env_ids, 2] = torch.rand(len(env_ids), device=self.device) * (self.cfg.ranges.ang_vel_yaw[1] - self.cfg.ranges.ang_vel_yaw[0]) + self.cfg.ranges.ang_vel_yaw[0]

        self.gait_frequency[env_ids] = torch.rand(len(env_ids), device=self.device) * (self.cfg.ranges.gait_frequency[1] - self.cfg.ranges.gait_frequency[0]) + self.cfg.ranges.gait_frequency[0]
        self.gait_process[env_ids] = 0.0
        # Set a portion of environments to be still
        still_envs_count = int(self.cfg.still_proportion * len(env_ids))
        if still_envs_count > 0:
            still_indices = torch.randperm(len(env_ids))[:still_envs_count]
            still_envs = torch.tensor(env_ids, device=self.device)[still_indices]
            self.commands[still_envs, :] = 0.0
            self.gait_frequency[still_envs] = 0.0

        # Update the resample time for the environments
        resampling_time_min = int(self.cfg.resampling_time_range[0] / self._env.step_dt)
        resampling_time_max = int(self.cfg.resampling_time_range[1] / self._env.step_dt)
        self.cmd_resample_time[env_ids] += torch.randint(
            resampling_time_min,
            resampling_time_max,
            (len(env_ids),),
            device=self.device,
        )

    def _update_command(self):
        """Post-processes the commands."""
        self.gait_process[:] = torch.fmod(self.gait_process + self.dt * self.gait_frequency, 1.0)

        # Enforce standing for designated environments
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        if len(standing_env_ids) > 0:
            self.commands[standing_env_ids, :] = 0.0
            self.gait_frequency[standing_env_ids] = 0.0
    
    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set the visibility of debug visualization markers."""
        if debug_vis:
            if not hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        """Callback for debug visualization."""
        if not self.robot.is_initialized:
            return

        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5

        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self.command[:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])

        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts XY velocity to arrow visualization parameters."""
        default_scale = torch.tensor([1.0, 0.1, 0.1], device=self.device) # Example scale
        
        arrow_scale = default_scale.repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 2.0  # Scale with velocity magnitude

        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat_b = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)

        base_quat_w = self.robot.data.root_quat_w
        arrow_quat_w = math_utils.quat_mul(base_quat_w, arrow_quat_b)

        return arrow_scale, arrow_quat_w
