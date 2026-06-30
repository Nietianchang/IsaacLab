# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Navigation SE2 action term for hierarchical control."""
from __future__ import annotations

from typing import TYPE_CHECKING

import os
import torch
from collections.abc import Sequence

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers.action_manager import ActionTerm
import isaaclab.utils.string as string_utils
from isaaclab.utils.assets import check_file_path, read_file

if TYPE_CHECKING:
    from .navigation_se2_actions_cfg import PerceptiveNavigationSE2ActionCfg


class PerceptiveNavigationSE2Action(ActionTerm):
    """Actions to navigate a robot using hierarchical control with a pre-trained locomotion policy."""

    cfg: PerceptiveNavigationSE2ActionCfg
    _env: ManagerBasedRLEnv

    def __init__(self, cfg: PerceptiveNavigationSE2ActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        # Check if policy file exists
        if not check_file_path(cfg.low_level_policy_file):
            raise FileNotFoundError(f"Policy file '{cfg.low_level_policy_file}' does not exist.")
        # Load pre-trained locomotion policy
        file_bytes = read_file(self.cfg.low_level_policy_file)
        self.low_level_policy = torch.jit.load(file_bytes, map_location=self.device)
        self.low_level_policy.eval()
        self._is_go2w_policy = "go2w" in self.cfg.low_level_policy_file.lower()

        # prepare joint position actions
        self.low_level_position_action_term: ActionTerm = self.cfg.low_level_position_action.class_type(cfg.low_level_position_action, env)
        self.low_level_velocity_action_term: ActionTerm = self.cfg.low_level_velocity_action.class_type(cfg.low_level_velocity_action, env)

        self._use_explicit_pd_control = bool(self.cfg.explicit_pd_control)
        # For GO2W, drive the joints through the IMPLICIT actuator PD (position targets for the
        # legs, velocity targets for the wheels). The policy-rate target computation (scaling,
        # hip-scale reduction, default-pos offset) is kept EXACTLY the same as the explicit-PD
        # path; only the final application changes from `set_joint_effort_target` (we compute
        # torques ourselves) to `set_joint_position_target` / `set_joint_velocity_target`
        # (PhysX computes the PD torques internally using the stiffness/damping baked into the
        # ImplicitActuatorCfg in go2w.py). The PD gains therefore live in the actuator cfg and
        # must match self.p_gains/self.d_gains used by the explicit path.
        self._use_implicit_pd_control = bool(self._is_go2w_policy)
        if self._use_implicit_pd_control:
            print("[DEBUG] GO2W policy detected: using implicit actuator PD (position/velocity targets)")

        # Resolve controlled joint names/ids from low-level action terms.
        # Explicitly set position and velocity joint name lists as requested.
        self._position_joint_names = [
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
        ]
        self._velocity_joint_names = ["FL_foot_joint", "FR_foot_joint", "RL_foot_joint", "RR_foot_joint"]
        print(f"[DEBUG] overridden low_level_position_action_term joint names: {self._position_joint_names}")
        print(f"[DEBUG] overridden low_level_velocity_action_term joint names: {self._velocity_joint_names}")
        self._controlled_joint_names = self._position_joint_names + self._velocity_joint_names

        # Resolve joint ids from the joint NAMES against the actual articulation DOF order
        # (preserving the requested name order). Previously these ids were hard-coded as
        # [0,4,8,1,5,9,...] / [12,13,14,15], which silently assumes the USD's DOF layout is
        # exactly breadth-first [hip*4, thigh*4, calf*4, foot*4]. If the loaded go2w.usd uses
        # any other ordering, the policy's leg/wheel actions get applied to the WRONG joints
        # (while the name-resolved observations still look correct), which makes the robot
        # twitch, bounce and fall. Resolving by name guarantees the action->joint mapping
        # matches the policy's output ordering regardless of the USD layout.
        self._position_joint_ids = self._asset.find_joints(self._position_joint_names, preserve_order=True)[0]
        self._velocity_joint_ids = self._asset.find_joints(self._velocity_joint_names, preserve_order=True)[0]
        self._controlled_joint_ids = self._position_joint_ids + self._velocity_joint_ids
        print(f"[DEBUG] resolved low_level_position_action_term ids: {self._position_joint_ids}")
        print(f"[DEBUG] resolved low_level_velocity_action_term ids: {self._velocity_joint_ids}")
        self._num_position_dofs = len(self._position_joint_ids)
        self._wheel_dof_indices = torch.arange(
            self._num_position_dofs,
            self._num_position_dofs + len(self._velocity_joint_ids),
            dtype=torch.long,
            device=self.device,
        )
        self._leg_action_idx = (
            torch.tensor(self.cfg.leg_action_idx, dtype=torch.long, device=self.device)
            if self.cfg.leg_action_idx is not None
            else None
        )
        self._wheel_action_idx = (
            torch.tensor(self.cfg.wheel_action_idx, dtype=torch.long, device=self.device)
            if self.cfg.wheel_action_idx is not None
            else None
        )
        self._hip_leg_action_idx = (
            torch.tensor(self.cfg.hip_action_idx, dtype=torch.long, device=self.device)
            if self.cfg.hip_action_idx is not None
            else None
        )

        # PD scales used by explicit torque mode.
        self._position_action_scale = self._resolve_action_scale(self.cfg.low_level_position_action.scale, self._position_joint_names)
        self._velocity_action_scale = self._resolve_action_scale(self.cfg.low_level_velocity_action.scale, self._velocity_joint_names)
        print(f"[DEBUG] low_level_position_action scale: {self._position_action_scale.tolist()[0]}")
        print(f"[DEBUG] low_level_velocity_action scale: {self._velocity_action_scale.tolist()[0]}")
        # Requested explicit variable: default_dof_pos (legs only, no wheels).
        self.default_dof_pos = self._asset.data.default_joint_pos[:, self._position_joint_ids].clone()
        if self.cfg.default_dof_pos is not None:
            if isinstance(self.cfg.default_dof_pos, dict):
                index_list, _, value_list = string_utils.resolve_matching_names_values(
                    self.cfg.default_dof_pos, self._position_joint_names
                )
                self.default_dof_pos[:, index_list] = torch.tensor(value_list, device=self.device)
            elif isinstance(self.cfg.default_dof_pos, list):
                if len(self.cfg.default_dof_pos) != self._num_position_dofs:
                    raise ValueError(
                        f"Expected default_dof_pos list length {self._num_position_dofs}, got {len(self.cfg.default_dof_pos)}"
                    )
                self.default_dof_pos[:] = torch.tensor(self.cfg.default_dof_pos, device=self.device).unsqueeze(0)
            else:
                raise ValueError(f"Unsupported default_dof_pos type: {type(self.cfg.default_dof_pos)}")
        print(f"[DEBUG] self.default_dof_pos: {self.default_dof_pos.tolist()[0]}")

        # PD gains and limits for explicit torque mode.
        # By default these come from articulation defaults, but can be overridden in cfg.
        self.p_gains = self._asset.data.default_joint_stiffness[:, self._controlled_joint_ids].clone()
        # print(f"[DEBUG] self.p_gains: {self.p_gains.tolist()[0]}")
        self.d_gains = self._asset.data.default_joint_damping[:, self._controlled_joint_ids].clone()
        # print(f"[DEBUG] self.d_gains: {self.d_gains.tolist()[0]}")
        if self.cfg.explicit_p_gains is not None:
            self.p_gains = self._resolve_joint_gain_cfg(self.cfg.explicit_p_gains, self._controlled_joint_names)
        print(f"[DEBUG] self.p_gains: {self.p_gains.tolist()[0]}")
    
        if self.cfg.explicit_d_gains is not None:
            self.d_gains = self._resolve_joint_gain_cfg(self.cfg.explicit_d_gains, self._controlled_joint_names)
        print(f"[DEBUG] self.d_gains: {self.d_gains.tolist()[0]}")
        self.torque_limits = self._asset.data.joint_effort_limits[:, self._controlled_joint_ids].clone()

        # One-time mass/inertia dump to compare against the training URDF (go2w.urdf:
        # base mass=6.921 kg, iyy/pitch inertia=0.098). Enable with DEBUG_PD=1.
        # If the USD base mass/inertia is much smaller than the URDF, the same (correct)
        # leg/wheel torques produce an excessive body angular acceleration -> the robot
        # pitches/bounces and falls right from spawn even though the policy I/O is correct.
        if os.getenv("DEBUG_PD"):
            try:
                body_names = self._asset.body_names
                masses = self._asset.root_physx_view.get_masses()[0]  # (num_bodies,)
                inertias = self._asset.root_physx_view.get_inertias()[0]  # (num_bodies, 9)
                print(f"[DEBUG_MASS] total_mass={float(masses.sum()):.4f} kg, num_bodies={len(body_names)}")
                for i, name in enumerate(body_names):
                    inertia_diag = inertias[i].reshape(3, 3).diagonal().tolist() if inertias.shape[-1] == 9 else inertias[i].tolist()
                    print(f"[DEBUG_MASS]   {name}: mass={float(masses[i]):.4f}  inertia_diag={inertia_diag}")
            except Exception as e:  # pragma: no cover - diagnostics only
                print(f"[DEBUG_MASS] failed to read masses/inertias: {e}")

        # prepare buffers
        self._action_dim = 3  # [vx, vy, omega]

        # set up buffers
        self._init_buffers()

        # storage for the most recent raw actions_phase (used for GO2W policy obs)
        self._last_actions_phase: torch.Tensor | None = None
        # storage for the most recent estimated base linear velocity from the low-level policy
        self._last_estimated_vel: torch.Tensor | None = None

        # Low-pass filter state for velocity commands
        self._prev_filtered_velocity_commands = torch.zeros((self.num_envs, self._action_dim), device=self.device)
        self._low_pass_alpha = self.cfg.low_pass_filter_alpha if hasattr(self.cfg, 'low_pass_filter_alpha') else 0.8
        self._enable_low_pass_filter = self.cfg.enable_low_pass_filter if hasattr(self.cfg, 'enable_low_pass_filter') else True
        # Per-environment per-dimension alpha values (initialized to default, can be randomized per episode)
        # Shape: [num_envs, action_dim] where action_dim = 3 (vx, vy, omega)
        self._per_env_per_dim_low_pass_alpha = torch.full((self.num_envs, self._action_dim), self._low_pass_alpha, device=self.device)


    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_navigation_velocity_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_navigation_velocity_actions

    @property
    def filtered_velocity_commands(self) -> torch.Tensor:
        """Get the current filtered (smoothed) velocity commands."""
        return self._prev_filtered_velocity_commands

    @property
    def low_pass_alpha_values(self) -> torch.Tensor:
        """Get the current per-environment per-dimension low-pass filter alpha values.

        Returns:
            torch.Tensor: Alpha values with shape [num_envs, action_dim] where:
                - action_dim = 3 for [vx, vy, omega]
                - Each environment can have different alpha values for each command dimension
        """
        return self._per_env_per_dim_low_pass_alpha

    @property
    def low_level_actions(self) -> torch.Tensor:
        return torch.cat((self._low_level_position_actions, self._low_level_velocity_actions), dim=1)

    @property
    def low_level_position_actions(self) -> torch.Tensor:
        return self._low_level_position_actions

    @property
    def prev_low_level_position_actions(self) -> torch.Tensor:
        return self._prev_low_level_position_actions

    @property
    def low_level_velocity_actions(self) -> torch.Tensor:
        return self._low_level_velocity_actions

    @property
    def prev_low_level_velocity_actions(self) -> torch.Tensor:
        return self._prev_low_level_velocity_actions

    """
    Operations.
    """

    def apply_low_pass_filter(self, velocity_commands: torch.Tensor) -> torch.Tensor:
        """Apply low-pass filter to velocity commands for smoother locomotion.

        The low-pass filter implements exponential smoothing:
        filtered_cmd(t) = alpha * filtered_cmd(t-1) + (1 - alpha) * new_cmd(t)

        Where alpha is the smoothing factor:
        - alpha = 0.0: No smoothing (pass through)
        - alpha = 1.0: Maximum smoothing (no change)
        - alpha = 0.8: Good balance for locomotion (default)

        This implementation supports per-environment per-dimension alpha values, allowing:
        - Different smoothing for vx, vy, and omega in each environment
        - Independent control over linear and angular velocity response

        Args:
            velocity_commands (torch.Tensor): Raw velocity commands [num_envs, 3] (vx, vy, omega)

        Returns:
            torch.Tensor: Filtered velocity commands with same shape as input
        """
        if not self._enable_low_pass_filter:
            return velocity_commands

        # Use per-environment per-dimension alpha values for filtering
        # Shape: [num_envs, action_dim] - already matches velocity_commands shape
        alpha_values = self._per_env_per_dim_low_pass_alpha

        # Apply exponential smoothing (low-pass filter) with per-environment per-dimension alpha
        filtered_commands = (
            alpha_values * self._prev_filtered_velocity_commands
            + (1.0 - alpha_values) * velocity_commands
        )

        # Update previous filtered commands for next iteration
        self._prev_filtered_velocity_commands.copy_(filtered_commands)

        return filtered_commands

    def process_actions(self, actions):
        """Process low-level navigation actions. This function is called with a frequency of 10Hz.

        Args:
            actions (torch.Tensor): The low-level navigation actions.
        """
        # Store the raw low-level navigation actions
        self._raw_navigation_velocity_actions[:] = actions
        # Apply the affine transformations
        if not self.cfg.use_raw_actions:
            self._processed_navigation_velocity_actions = (
                self._raw_navigation_velocity_actions * self._scale + self._offset
            )
        else:
            self._processed_navigation_velocity_actions[:] = self._raw_navigation_velocity_actions

        if self.cfg.policy_distr_type == "gaussian":
            # scale the actions to the range [-1, 1] for gaussian distribution
            self._processed_navigation_velocity_actions = torch.tanh(self._processed_navigation_velocity_actions)
        elif self.cfg.policy_distr_type == "beta":
            # scale the actions to the range [-1, 1] for beta distribution
            self._processed_navigation_velocity_actions = (self._processed_navigation_velocity_actions - 0.5) * 2.0
        else:
            raise ValueError(f"Unknown policy distribution type: {self.cfg.policy_distr_type}")

        # Compute robot speed from simulator state to avoid relying on observation ordering.
        base_lin_vel = self._asset.data.root_lin_vel_b[:, :3]
        vel_xyz = base_lin_vel.norm(dim=1, keepdim=True)
        # print("self._policy_bias", self._policy_bias)
        # [vx, vy, omega]
        self._processed_navigation_velocity_actions = (self._processed_navigation_velocity_actions + vel_xyz * self._policy_bias) * self._policy_scaling

        # Apply low-pass filter to smooth velocity commands and add delay effect
        self._processed_navigation_velocity_actions = self.apply_low_pass_filter(self._processed_navigation_velocity_actions)

    @torch.inference_mode()
    def apply_actions(self):
        """Apply low-level actions for the simulator to the physics engine. This functions is called with the
        simulation frequency of 200Hz. Since low-level locomotion runs at 50Hz, we need to decimate the actions."""

        if self._counter % self.cfg.low_level_decimation == 0:
            self._counter = 0
            self._prev_low_level_position_actions[:] = self._low_level_position_actions.clone()
            self._prev_low_level_velocity_actions[:] = self._low_level_velocity_actions.clone()

            # Get low-level policy input (single-frame proprio observations).
            current_obs = self._env.observation_manager.compute_group(group_name=self.cfg.observation_group)

            # Update observation history to match the real deployment controller, which does:
            #     self.obs = concat(self.obs[1:], curr_obs)        # append current frame
            #     obs_hist_tensor = self.obs                        # history INCLUDES current
            #     action = policy(curr_obs, obs_hist_tensor)       # then run inference
            # i.e. the current frame is rolled into history[-1] BEFORE inference, and the
            # buffer starts zero-filled. Replicate that exactly here.
            if self.cfg.use_observation_history:
                if self._obs_history is None:
                    self._obs_history = torch.zeros(
                        (current_obs.shape[0], self.cfg.history_length, current_obs.shape[1]),
                        dtype=current_obs.dtype,
                        device=current_obs.device,
                    )
                self._obs_history = self._obs_history.roll(-1, dims=1)
                self._obs_history[:, -1, :] = current_obs

            # Run low-level policy.
            # GO2W uses dual-input (obs, obs_hist) in fp16 only. Other robots use single-input (obs) in fp32.
            policy_curr_obs = current_obs.half() if self._is_go2w_policy else current_obs
            policy_hist_obs = self._obs_history.half() if (self._is_go2w_policy and self._obs_history is not None) else self._obs_history

            if self._is_go2w_policy:
                if policy_hist_obs is None:
                    raise RuntimeError("GO2W policy requires obs_hist, but history buffer is None.")

                if policy_hist_obs.shape[0] != policy_curr_obs.shape[0]:
                    raise RuntimeError(
                        f"GO2W obs_hist batch mismatch: obs batch={policy_curr_obs.shape[0]}, obs_hist batch={policy_hist_obs.shape[0]}"
                    )
                if policy_hist_obs.shape[1] != self.cfg.history_length:
                    raise RuntimeError(
                        f"GO2W obs_hist length mismatch: expected {self.cfg.history_length}, got {policy_hist_obs.shape[1]}"
                    )
                if policy_hist_obs.shape[2] != policy_curr_obs.shape[1]:
                    raise RuntimeError(
                        f"GO2W obs_hist feature mismatch: obs dim={policy_curr_obs.shape[1]}, obs_hist dim={policy_hist_obs.shape[2]}"
                    )
                # print(f"[DEBUG] self.policy_curr_obs: {policy_curr_obs.tolist()[0]}")
                # print(f"[DEBUG] self.policy_hist_obs: {policy_hist_obs.tolist()[0]}")
                if hasattr(self.low_level_policy, "forward_with_vel"):
                    actions_phase, vel = self.low_level_policy.forward_with_vel(policy_curr_obs, policy_hist_obs)
                    # Store the intermediate estimated base linear velocity (fp32) so it can
                    # be consumed by high-level observation terms (e.g. base_ang_vel_delayed).
                    self._last_estimated_vel = vel.float()
                else:
                    out = self.low_level_policy(policy_curr_obs, policy_hist_obs)
                    if isinstance(out, (tuple, list)):
                        actions_phase, vel = out[0], out[1]
                        self._last_estimated_vel = vel.float()
                    else:
                        actions_phase = out
                # The exported GO2W jit model only accepts fp16 inputs, so inference itself
                # runs in fp16 (this is the precision floor and matches deployment). Cast the
                # fp16 model output back to fp32 IMMEDIATELY so that the recursive EMA filter
                # and all downstream target math run in full precision. This mirrors the
                # deploy controller, where `last_action` is a float32 numpy array and the
                # `last_action*0.2 + action*0.8` update is promoted to float32 -- i.e. only the
                # raw policy forward pass is fp16, never the accumulating filter state.
                actions_phase = actions_phase.float()
                # print(f"[DEBUG] self.actions_phase (raw): {actions_phase.tolist()[0]}")

                # Match the deployment controller (deploy_mujoco_go2w_np3o.py) EXACTLY:
                #     action = last_action * 0.2 + action * 0.8
                #     last_action = action.copy()
                # i.e. a TRUE recursive EMA where `last_action` holds the PREVIOUS step's
                # *filtered* action. Both the applied targets and the next step's `actions`
                # observation channel use this same filtered value, and the buffer starts
                # zero-filled at episode reset. Kept in fp32 to avoid fp16 accumulation error.
                if self._last_actions_phase is None:
                    self._last_actions_phase = torch.zeros_like(actions_phase)
                actions_phase = self._last_actions_phase * 0.2 + actions_phase * 0.8
                self._last_actions_phase = actions_phase.clone()
                # print(f"[DEBUG] self._last_actions_phase: {self._last_actions_phase.tolist()[0]}")

            else:
                actions_phase = self.low_level_policy(policy_curr_obs)
                # Non-GO2W policies do not use actions_phase storage
                self._last_actions_phase = None

            # Process actions and bring them in the right order.
            if self._leg_action_idx is not None and self._wheel_action_idx is not None:
                self._low_level_position_actions[:] = actions_phase[:, self._leg_action_idx]
                self._low_level_velocity_actions[:] = actions_phase[:, self._wheel_action_idx]
            else:
                self._low_level_position_actions[:] = actions_phase[:, :self.low_level_position_action_term.action_dim]
                self._low_level_velocity_actions[:] = actions_phase[:, self.low_level_position_action_term.action_dim:]

            # Debug prints: show tensor shapes for low-level actions

            # print(
            #     f"[DEBUG] low_level_position_actions shape: {self._low_level_position_actions.shape}, "
            #     f"low_level_velocity_actions shape: {self._low_level_velocity_actions.shape}"
            # )


            if self._use_explicit_pd_control or self._use_implicit_pd_control:
                # Only update the policy-rate targets here (50Hz). For explicit PD the actual
                # torque is recomputed every physics step (200Hz) in the apply block below; for
                # implicit PD these same targets are handed to PhysX which does the PD internally.
                # Target law (identical in both modes):
                #   joint_pos_target = pos_actions_scaled + default_dof_pos   (legs)
                #   wheel_vel_target = vel_actions_scaled                     (wheels)
                pos_actions_scaled = self._low_level_position_actions * self._position_action_scale
                vel_actions_scaled = self._low_level_velocity_actions * self._velocity_action_scale
                if self._hip_leg_action_idx is not None:
                    pos_actions_scaled[:, self._hip_leg_action_idx] *= self.cfg.hip_scale_reduction

                self._leg_pos_target[:] = pos_actions_scaled + self.default_dof_pos
                self._wheel_vel_target[:] = vel_actions_scaled
            else:
                # Process low level actions
                self.low_level_position_action_term.process_actions(self._low_level_position_actions)
                self.low_level_velocity_action_term.process_actions(self._low_level_velocity_actions)

        # Apply low level actions
        if self._use_implicit_pd_control:
            # Implicit actuator PD: hand the policy-rate targets to PhysX and let the implicit
            # actuator model compute the PD torques internally using the stiffness/damping baked
            # into the ImplicitActuatorCfg (legs: position control p/d; wheels: velocity control
            # with stiffness=0). Legs use position targets, wheels use velocity targets.
            self._asset.set_joint_position_target(self._leg_pos_target, joint_ids=self._position_joint_ids)
            self._asset.set_joint_velocity_target(self._wheel_vel_target, joint_ids=self._velocity_joint_ids)
        elif self._use_explicit_pd_control:
            # Recompute PD torques every physics step from the latest joint state, while the
            # policy targets (self._leg_pos_target / self._wheel_vel_target) are held constant
            # between policy updates. This restores per-sub-step damping and matches the
            # deployment-time control loop.
            #   torques        = p * (joint_pos_target - dof_pos) - d * dof_vel   (legs)
            #   torques_wheels = d * (wheel_vel_target - dof_vel)                 (wheels)
            dof_pos = self._asset.data.joint_pos[:, self._controlled_joint_ids]
            dof_vel = self._asset.data.joint_vel[:, self._controlled_joint_ids]

            full_joint_pos_target = dof_pos.clone()
            full_joint_pos_target[:, : self._num_position_dofs] = self._leg_pos_target
            torques = self.p_gains * (full_joint_pos_target - dof_pos) - self.d_gains * dof_vel

            full_vel_target = torch.zeros_like(dof_vel)
            full_vel_target[:, self._wheel_dof_indices] = self._wheel_vel_target
            torques_wheels = self.d_gains * (full_vel_target - dof_vel)
            torques[:, self._wheel_dof_indices] = torques_wheels[:, self._wheel_dof_indices]

            self._computed_torques.copy_(torch.clamp(torques, -self.torque_limits, self.torque_limits))
            # print(f"[DEBUG] computed_torques: {self._computed_torques.tolist()[0]}")
            # print(f"[DEBUG] self._controlled_joint_ids: {self._controlled_joint_ids}")
            self._asset.set_joint_effort_target(self._computed_torques, joint_ids=self._controlled_joint_ids)

            # Opt-in diagnostics (env 0, at policy rate). Enable with DEBUG_PD=1.
            # Reveals whether torques saturate, legs fail to track, or the base is tipping.
            # if os.getenv("DEBUG_PD") and (self._counter % self.cfg.low_level_decimation == 0):
            #     leg_err = (full_joint_pos_target[:, : self._num_position_dofs] - dof_pos[:, : self._num_position_dofs])
            #     tq = self._computed_torques
            #     sat = (tq.abs() >= (self.torque_limits - 1e-4)).float().mean().item()
            #     root_z = self._asset.data.root_pos_w[0, 2].item()
            #     grav_z = self._asset.data.projected_gravity_b[0, 2].item()
            #     lin_b = self._asset.data.root_lin_vel_b[0]
            #     grav_b = self._asset.data.projected_gravity_b[0]
            #     print(
            #         f"[DEBUG_PD] rootZ={root_z:.3f} gravZ={grav_z:.3f} "
            #         f"|legErr|={leg_err[0].abs().max().item():.3f} "
            #         f"|tq|max={tq[0].abs().max().item():.2f} satFrac={sat:.2f} "
            #         f"|dofVel|max={dof_vel[0].abs().max().item():.2f} "
            #         f"linVel_b=[{lin_b[0].item():.3f},{lin_b[1].item():.3f},{lin_b[2].item():.3f}] "
            #         f"grav_b=[{grav_b[0].item():.3f},{grav_b[1].item():.3f}]"
            #     )
        else:
            self.low_level_position_action_term.apply_actions()
            self.low_level_velocity_action_term.apply_actions()
        self._counter += 1

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset stateful buffers for specified environments."""
        if env_ids is None:
            env_ids = slice(None)
        self.reset_low_pass_filter(env_ids)
        self._prev_low_level_position_actions[env_ids] = 0.0
        self._prev_low_level_velocity_actions[env_ids] = 0.0
        self._low_level_position_actions[env_ids] = 0.0
        self._low_level_velocity_actions[env_ids] = 0.0
        self._computed_torques[env_ids] = 0.0
        # Reset PD targets: legs hold the default joint positions (zero torque at rest),
        # wheels target zero velocity.
        self._leg_pos_target[env_ids] = self.default_dof_pos[env_ids]
        self._wheel_vel_target[env_ids] = 0.0
        if self._obs_history is not None:
            # History can be created in apply_actions() under inference mode.
            # Clone once so indexed in-place reset works outside inference mode.
            if self._obs_history.is_inference():
                self._obs_history = self._obs_history.clone()
            self._obs_history[env_ids] = 0.0
        if self._prev_raw_actions_phase is not None:
            if self._prev_raw_actions_phase.is_inference():
                self._prev_raw_actions_phase = self._prev_raw_actions_phase.clone()
            self._prev_raw_actions_phase[env_ids] = 0.0
        # Match deploy: `last_action` (the recursive EMA state, also fed into the next
        # `actions` obs channel) resets to zeros at episode start.
        if self._last_actions_phase is not None:
            if self._last_actions_phase.is_inference():
                self._last_actions_phase = self._last_actions_phase.clone()
            self._last_actions_phase[env_ids] = 0.0
        if self._last_estimated_vel is not None:
            if self._last_estimated_vel.is_inference():
                self._last_estimated_vel = self._last_estimated_vel.clone()
            self._last_estimated_vel[env_ids] = 0.0

    def reset_low_pass_filter(self, env_ids: torch.Tensor):
        """Reset low-pass filter state for specified environments.

        Args:
            env_ids: Environment indices to reset.
        """
        self._prev_filtered_velocity_commands[env_ids] = 0.0

    """
    Helper functions
    """

    def _init_buffers(self):
        # Prepare buffers
        self._raw_navigation_velocity_actions = torch.zeros(self.num_envs, self._action_dim, device=self.device)
        self._processed_navigation_velocity_actions = torch.zeros((self.num_envs, self._action_dim), device=self.device)
        self._low_level_position_actions = torch.zeros(self.num_envs, self.low_level_position_action_term.action_dim, device=self.device)
        self._low_level_velocity_actions = torch.zeros(self.num_envs, self.low_level_velocity_action_term.action_dim, device=self.device)
        self._prev_low_level_position_actions = torch.zeros_like(self._low_level_position_actions)
        self._prev_low_level_velocity_actions = torch.zeros_like(self._low_level_velocity_actions)
        self._computed_torques = torch.zeros(
            self.num_envs,
            self._num_position_dofs + len(self._velocity_joint_ids),
            device=self.device,
        )
        # Policy-rate targets held constant between low-level policy updates and used by the
        # per-sub-step explicit PD torque computation.
        self._leg_pos_target = torch.zeros(self.num_envs, self._num_position_dofs, device=self.device)
        self._wheel_vel_target = torch.zeros(self.num_envs, len(self._velocity_joint_ids), device=self.device)
        self._low_level_step_dt = self.cfg.low_level_decimation * self._env.physics_dt
        self._counter = 0
        self._scale = torch.tensor(self.cfg.scale, device=self.device)
        self._offset = torch.tensor(self.cfg.offset, device=self.device)
        self._policy_scaling = torch.tensor(self.cfg.policy_scaling, device=self.device).repeat(self.num_envs, 1)
        self._policy_bias = torch.zeros(self.num_envs, self._action_dim, device=self.device)
        self._obs_history: torch.Tensor | None = None
        # [EXPERIMENT] previous RAW policy action, used by the training-faithful FIR action
        # filter when EMA_FILTER is set (filtered = 0.2*prev_raw + 0.8*raw).
        self._prev_raw_actions_phase: torch.Tensor | None = None

    def _normalize_joint_ids(self, joint_ids: slice | list[int]) -> list[int]:
        if isinstance(joint_ids, slice):
            return list(range(self._asset.num_joints))[joint_ids]
        return list(joint_ids)

    def _resolve_action_scale(self, scale_cfg, joint_names: list[str]) -> torch.Tensor:
        if isinstance(scale_cfg, (float, int)):
            return torch.full((1, len(joint_names)), float(scale_cfg), device=self.device)
        if isinstance(scale_cfg, dict):
            scale = torch.ones((1, len(joint_names)), device=self.device)
            index_list, _, value_list = string_utils.resolve_matching_names_values(scale_cfg, joint_names)
            scale[:, index_list] = torch.tensor(value_list, device=self.device)
            return scale
        raise ValueError(f"Unsupported action scale type: {type(scale_cfg)}")

    def _resolve_joint_gain_cfg(self, gains_cfg, joint_names: list[str]) -> torch.Tensor:
        if isinstance(gains_cfg, list):
            if len(gains_cfg) != len(joint_names):
                raise ValueError(
                    f"Expected gain list length {len(joint_names)}, got {len(gains_cfg)}"
                )
            return torch.tensor(gains_cfg, dtype=torch.float, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        if isinstance(gains_cfg, dict):
            gains = torch.zeros((1, len(joint_names)), dtype=torch.float, device=self.device)
            index_list, _, value_list = string_utils.resolve_matching_names_values(gains_cfg, joint_names)
            gains[:, index_list] = torch.tensor(value_list, dtype=torch.float, device=self.device)
            return gains.repeat(self.num_envs, 1)
        raise ValueError(f"Unsupported gains type: {type(gains_cfg)}")
