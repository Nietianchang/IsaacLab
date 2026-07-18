# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT


from __future__ import annotations

import math
from dataclasses import MISSING
from typing import TYPE_CHECKING, Literal

from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg
from .goal_commands import RobotNavigationGoalCommand


"""
Base command generator.
"""

@configclass
class RobotNavigationGoalCommandCfg(CommandTermCfg):
    """Configuration for the robot goal command generator."""

    class_type: type = RobotNavigationGoalCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    robot_to_goal_line_vis: bool = True
    """If true, visualize the line from the robot to the goal."""

    # ---------------------------------------------------------------------
    # Stairs goal curriculum
    # ---------------------------------------------------------------------
    stair_curriculum_enabled: bool = True
    """If true, enable the stairs-specific goal curriculum.

    When a goal is sampled on a stairs tile, the goal is snapped onto the step
    whose height is closest to the current curriculum height. The curriculum
    height slowly increases as the success rate on stair goals improves.
    """

    stair_curriculum_start_height: float = 0.4
    """Initial target step height (meters) for stair goals."""

    stair_curriculum_increment: float = 0.3
    """Amount (meters) to raise the target step height on each curriculum step."""

    stair_curriculum_max_height: float = 3.0
    """Maximum target step height (meters). Per-tile the target is additionally
    capped by the tallest available step on that staircase."""

    stair_curriculum_success_threshold: float = 0.45
    """Success rate on stair goals required before raising the target height."""

    stair_curriculum_buffer_size: int = 100
    """Size of the rolling buffer tracking stair-goal successes/failures."""

    stair_curriculum_min_samples: int = 30
    """Minimum number of recorded stair-goal outcomes before the success rate is
    trusted enough to advance the curriculum."""

    stair_curriculum_min_goals: int = 600
    """Ensure at least this many stair goals exist globally. If fewer goals land on
    staircases, nearby flat-walkway goals inside stairs tiles are converted onto the
    nearest step at the corresponding curriculum height until this count is reached."""

