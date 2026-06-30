# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import MISSING
from typing import Optional

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from .navigation_se2_actions import PerceptiveNavigationSE2Action


@configclass
class PerceptiveNavigationSE2ActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = PerceptiveNavigationSE2Action
    """ Class of the action term."""
    low_level_decimation: int = 4
    """Decimation factor for the low level action term."""
    use_raw_actions: bool = False
    """Whether to use raw actions or not."""
    scale: list[float] = [1.0, 1.0, 1.0]
    """Scale for the actions [vx, vy, w]."""
    offset: list[float] = [0.0, 0.0, 0.0]
    """Offset for the actions [vx, vy, w]."""
    low_level_velocity_action: ActionTermCfg = MISSING
    """Configuration of the low level velocity action term."""
    low_level_position_action: ActionTermCfg = MISSING
    """Configuration of the low level position action term."""
    low_level_policy_file: str = MISSING
    """Path to the low level policy file."""
    observation_group: str = "policy"
    """Observation group to use for the low level policy."""
    policy_scaling: list[float] = [1.0, 1.0, 1.0]
    """Policy dependent scaling for the actions [vx, vy, w]."""
    reorder_joint_list: list[str] | None = None
    """Reorder the joint actions given from the low-level policy to match the Isaac Sim order if policy has been
    trained with a different order. Set to None to disable reordering."""
    policy_distr_type: str = "gaussian"
    """Policy distribution type: 'gaussian', 'beta'."""
    # Low-pass filter parameters
    enable_low_pass_filter: bool = True
    """Whether to enable low-pass filtering for velocity commands."""
    low_pass_filter_alpha: float = 0.5
    """Low-pass filter smoothing factor (0.0 = no smoothing, 1.0 = maximum smoothing).
    Formula: filtered_cmd = alpha * prev_filtered_cmd + (1 - alpha) * new_cmd"""
    use_observation_history: bool = False
    """Whether to call low-level policy with two inputs: (curr_obs, obs_history)."""
    history_length: int = 10
    """History length for low-level policy observation stack when use_observation_history is enabled."""
    use_half_precision_inference: bool = True
    """Whether to cast policy inputs to fp16 before JIT inference (matches deployment usage)."""
    hip_scale_reduction: float = 1.0
    """Additional multiplicative scale for hip position actions from the low-level policy."""
    hip_joint_names_expr: Optional[list[str]] = None
    """Regex expressions to identify hip joints within low-level position action joints.

    If None, no additional hip-specific scaling is applied.
    """
    explicit_pd_control: bool = False
    """Whether to compute and apply explicit PD torques from low-level actions."""
    explicit_p_gains: dict[str, float] | list[float] | None = None
    """Optional proportional gains for explicit PD over controlled joints (legs + wheels).

    Supported formats:
    - dict[str, float]: matched against controlled joint names
    - list[float]: ordered list with length equal to number of controlled joints

    If None, defaults are taken from articulation stiffness.
    """
    explicit_d_gains: dict[str, float] | list[float] | None = None
    """Optional derivative gains for explicit PD over controlled joints (legs + wheels).

    Supported formats:
    - dict[str, float]: matched against controlled joint names
    - list[float]: ordered list with length equal to number of controlled joints

    If None, defaults are taken from articulation damping.
    """
    default_dof_pos: dict[str, float] | list[float] | None = None
    """Optional default leg-joint positions used as `default_dof_pos` for explicit PD control.

    Supported formats:
    - dict[str, float]: matched against low-level position (leg) joint names
    - list[float]: ordered list with length equal to number of leg joints

    If None, the articulation default leg-joint positions are used.
    """
    leg_action_idx: list[int] | None = None
    """Indices of leg actions inside the low-level policy output action vector."""
    wheel_action_idx: list[int] | None = None
    """Indices of wheel actions inside the low-level policy output action vector."""
    hip_action_idx: list[int] | None = None
    """Indices of hip joints inside leg-action sub-vector (used for hip scale reduction)."""
