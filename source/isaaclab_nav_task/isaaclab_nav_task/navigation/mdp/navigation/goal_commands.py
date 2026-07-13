# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Goal command generator for robot navigation tasks.

Simplified architecture:
1. Terrain generation creates `valid_mask` (boolean mask of valid positions with safety padding)
2. This module samples goal/spawn positions uniformly from valid positions
3. Z-height is looked up from the visual height field

The terrain module handles:
- Height field generation
- Obstacle detection
- Safety padding (dilation)
- Border exclusion
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Tuple, Optional

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import (
    CUBOID_MARKER_CFG,
    GREEN_ARROW_X_MARKER_CFG,
    RED_ARROW_X_MARKER_CFG,
)
from isaaclab.utils.math import subtract_frame_transforms, transform_points, yaw_quat

from isaaclab_nav_task.navigation.mdp.math_utils import vec_to_quat
from isaaclab_nav_task.terrains.terrain_constants import VERTICAL_SCALE

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from .goal_commands_cfg import RobotNavigationGoalCommandCfg


# =============================================================================
# Position Sampler (Simplified)
# =============================================================================

class PositionSampler:
    """Samples positions uniformly from valid terrain cells.

    Uses pre-computed valid_mask from terrain generation (already has safety padding).
    For spawn positions, uses spawn_mask with larger padding to account for robot body.

    Coordinate System:
    - The height field has shape (num_cells_x, num_cells_y), e.g., (299, 299)
    - Each cell is horizontal_scale meters (e.g., 0.1m)
    - The mesh is generated with a border: border_pixels = int(border_width / horizontal_scale) + 1
    - Our valid_mask corresponds to the inner terrain (no border)
    - Local coordinates are centered: (-terrain_size/2, -terrain_size/2) to (+terrain_size/2, +terrain_size/2)
    - Border pixel offset is computed dynamically from terrain config (not hardcoded)
    """

    def __init__(
        self,
        heights: torch.Tensor,
        valid_mask: torch.Tensor,
        platform_mask: torch.Tensor,
        terrain_size: float,
        horizontal_scale: float,
        device: torch.device,
        platform_repeat_count: int = 10,
        spawn_mask: torch.Tensor = None,
        border_width: float = 0.0,
    ):
        """Initialize the sampler.

        Args:
            heights: Height field (num_terrains, width, height) for Z-lookup.
            valid_mask: Boolean mask of valid goal positions (num_terrains, width, height).
            platform_mask: Boolean mask of platform positions for curriculum.
            terrain_size: Size of each terrain in meters (full mesh size).
            horizontal_scale: Resolution of height field in meters per cell.
            device: Torch device.
            platform_repeat_count: Repetition count for platform positions.
            spawn_mask: Boolean mask of valid spawn positions with larger padding.
                        If None, defaults to valid_mask.
            border_width: Border width around terrain in meters (from terrain config).
        """
        self.device = device
        self.terrain_size = terrain_size
        self.horizontal_scale = horizontal_scale
        self.heights = heights
        self.valid_mask = valid_mask
        self.platform_mask = platform_mask
        # Use spawn_mask if provided, otherwise fall back to valid_mask
        self.spawn_mask = spawn_mask if spawn_mask is not None else valid_mask

        # Use horizontal_scale as cell size (correct resolution)
        self.cell_size = horizontal_scale

        # Compute border pixel offset dynamically based on terrain configuration
        # Formula matches patches.py: border_pixels = int(border_width / horizontal_scale) + 1
        # This ensures valid_mask indices map correctly to mesh coordinates
        self.border_pixels = int(border_width / horizontal_scale) + 1

        # Mesh center offset (mesh is centered at origin after transform)
        self.mesh_center = terrain_size / 2  # e.g., 30 / 2 = 15m

        # Build position tables for both goal and spawn sampling
        self._build_position_tables(platform_repeat_count)

    def _build_position_tables(self, platform_repeat_count: int):
        """Build pre-computed position tensors for efficient sampling.

        Creates two sets of position tables:
        - Goal positions: from valid_mask with platform repetition for curriculum
        - Spawn positions: from spawn_mask (larger padding for robot body clearance)
        """
        num_terrains = self.valid_mask.shape[0]

        # =========================
        # Build GOAL position table (from valid_mask with platform repetition)
        # =========================
        valid_indices = self.valid_mask.nonzero(as_tuple=False)

        # Build enhanced indices with platform repetition
        enhanced_indices = []
        for terrain_idx in range(num_terrains):
            terrain_valid = valid_indices[valid_indices[:, 0] == terrain_idx]

            if len(terrain_valid) == 0:
                enhanced_indices.append(terrain_valid)
                continue

            # Find platform positions
            terrain_platform = self.platform_mask[terrain_idx]
            platform_positions = terrain_platform.nonzero(as_tuple=False)

            if len(platform_positions) > 0:
                # Check which valid positions are platforms (vectorized)
                valid_xy = terrain_valid[:, 1:]  # (num_valid, 2)
                plat_xy = platform_positions  # (num_platforms, 2)

                # Broadcast compare: (num_valid, 1, 2) vs (1, num_platforms, 2)
                matches = (valid_xy.unsqueeze(1) == plat_xy.unsqueeze(0)).all(dim=2)
                is_platform = matches.any(dim=1)

                # Repeat platform positions
                platform_valid = terrain_valid[is_platform]
                if len(platform_valid) > 0:
                    repeated = platform_valid.repeat(platform_repeat_count, 1)
                    terrain_valid = torch.cat([terrain_valid, repeated], dim=0)

            enhanced_indices.append(terrain_valid)

        # Count positions per terrain for goals
        self.count_per_terrain = torch.zeros(num_terrains, dtype=torch.long, device=self.device)
        for terrain_idx in range(num_terrains):
            self.count_per_terrain[terrain_idx] = len(enhanced_indices[terrain_idx])

        # Create padded tensor for goal positions
        max_count = max(1, self.count_per_terrain.max().item())
        self.positions = torch.full(
            (num_terrains, max_count, 3), -1, dtype=torch.long, device=self.device
        )

        # Fill goal position tables
        for terrain_idx in range(num_terrains):
            terrain_positions = enhanced_indices[terrain_idx]
            num_pos = terrain_positions.shape[0]
            if num_pos > 0:
                self.positions[terrain_idx, :num_pos] = terrain_positions

        # =========================
        # Build SPAWN position table (from spawn_mask, no platform repetition)
        # =========================
        spawn_indices = self.spawn_mask.nonzero(as_tuple=False)

        # Count spawn positions per terrain
        self.spawn_count_per_terrain = torch.zeros(num_terrains, dtype=torch.long, device=self.device)
        spawn_positions_list = []
        for terrain_idx in range(num_terrains):
            terrain_spawn = spawn_indices[spawn_indices[:, 0] == terrain_idx]
            self.spawn_count_per_terrain[terrain_idx] = len(terrain_spawn)
            spawn_positions_list.append(terrain_spawn)

        # Create padded tensor for spawn positions
        max_spawn_count = max(1, self.spawn_count_per_terrain.max().item())
        self.spawn_positions = torch.full(
            (num_terrains, max_spawn_count, 3), -1, dtype=torch.long, device=self.device
        )

        # Fill spawn position tables
        for terrain_idx in range(num_terrains):
            terrain_positions = spawn_positions_list[terrain_idx]
            num_pos = terrain_positions.shape[0]
            if num_pos > 0:
                self.spawn_positions[terrain_idx, :num_pos] = terrain_positions

    def sample(self, terrain_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample GOAL positions for given terrain indices.

        Uses valid_mask with platform repetition for curriculum learning.

        Args:
            terrain_indices: Tensor of terrain indices to sample from.

        Returns:
            Tuple of (x, y, z) local coordinates in meters.
        """
        return self._sample_from_table(
            terrain_indices,
            self.positions,
            self.count_per_terrain
        )

    def sample_spawn(self, terrain_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample SPAWN positions for given terrain indices.

        Uses spawn_mask with larger padding to account for:
        - Robot body dimensions with random orientation
        - Platform edge safety margins
        - Controller startup behavior

        Args:
            terrain_indices: Tensor of terrain indices to sample from.

        Returns:
            Tuple of (x, y, z) local coordinates in meters.
        """
        return self._sample_from_table(
            terrain_indices,
            self.spawn_positions,
            self.spawn_count_per_terrain
        )

    def _sample_from_table(
        self,
        terrain_indices: torch.Tensor,
        positions_table: torch.Tensor,
        count_per_terrain: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Internal method to sample from a position table.

        Args:
            terrain_indices: Tensor of terrain indices to sample from.
            positions_table: Pre-computed position table (num_terrains, max_count, 3).
            count_per_terrain: Number of valid positions per terrain.

        Returns:
            Tuple of (x, y, z) local coordinates in meters.
        """
        num_samples = terrain_indices.shape[0]

        # Random indices within valid range
        valid_counts = count_per_terrain[terrain_indices].float().clamp(min=1)
        random_indices = (torch.rand(num_samples, device=self.device) * valid_counts).long()

        # Lookup positions
        selected = positions_table[terrain_indices, random_indices]  # (n, 3)
        is_valid = selected[:, 0] >= 0

        local_x = torch.zeros(num_samples, device=self.device)
        local_y = torch.zeros(num_samples, device=self.device)
        local_z = torch.zeros(num_samples, device=self.device)

        if is_valid.any():
            valid_selected = selected[is_valid]
            x_idx = valid_selected[:, 1]
            y_idx = valid_selected[:, 2]

            # Convert to meters (accounting for border pixel offset)
            # The mesh is generated with @height_field_to_mesh which adds a border
            # border_pixels = int(border_width / horizontal_scale) + 1 (computed dynamically)
            # Mesh vertex at (i, j) has position: (i * h_scale - terrain_size/2, j * h_scale - terrain_size/2)
            # Our valid_mask[i, j] corresponds to mesh heights[i + border_pixels, j + border_pixels]
            # So the world position for valid_mask[i, j] is:
            #   x = (i + border_pixels) * h_scale - terrain_size/2
            #   y = (j + border_pixels) * h_scale - terrain_size/2
            local_x[is_valid] = (x_idx.float() + self.border_pixels) * self.cell_size - self.mesh_center
            local_y[is_valid] = (y_idx.float() + self.border_pixels) * self.cell_size - self.mesh_center

            # Lookup Z from heights (our heights tensor matches valid_mask dimensions, no border offset needed)
            height_values = self.heights[valid_selected[:, 0], x_idx, y_idx]
            local_z[is_valid] = height_values.float() * VERTICAL_SCALE

        return local_x, local_y, local_z


# =============================================================================
# Success Rate Tracker
# =============================================================================

class SuccessRateTracker:
    """Tracks navigation success rates using a rolling buffer."""

    def __init__(self, num_envs: int, device: torch.device, buffer_size: int = 10):
        self.device = device
        self.buffer_size = buffer_size
        self.buffer = torch.full((num_envs, buffer_size), -1.0, device=device)
        self.write_index = torch.zeros(num_envs, dtype=torch.long, device=device)
        # Per-env verdict of the most recent episode termination (1=success,
        # 0=failure, -1=no result yet). Updated by every termination through
        # record_result, so other consumers (e.g. the stairs curriculum) can reuse
        # the exact same success/failure decision instead of recomputing it.
        self.last_result = torch.full((num_envs,), -1.0, device=device)

    def record_result(self, success: torch.Tensor, env_ids: torch.Tensor):
        indices = self.write_index[env_ids] % self.buffer_size
        self.buffer[env_ids, indices] = success[env_ids].float()
        self.write_index[env_ids] += 1
        self.last_result[env_ids] = success[env_ids].float()

    def add(self, results: torch.Tensor, env_ids: torch.Tensor):
        """Legacy alias."""
        self.record_result(results, env_ids)

    def clear(self, env_ids: torch.Tensor):
        self.buffer[env_ids] = -1.0
        self.write_index[env_ids] = 0

    def get_success_rate(self) -> torch.Tensor:
        filled_count = (self.buffer >= 0).sum(dim=1).clamp(min=1)
        success_count = (self.buffer > 0).sum(dim=1)
        return success_count.float() / filled_count.float()


# =============================================================================
# Main Navigation Goal Command Generator
# =============================================================================

class RobotNavigationGoalCommand(CommandTerm):
    """Command generator for robot navigation goal positions.

    Samples goal and spawn positions from terrain-provided valid_mask.
    """

    cfg: RobotNavigationGoalCommandCfg

    # Termination reason codes written by mdp/terminations.py.
    TERM_REASON_TIMEOUT = 0
    TERM_REASON_AT_GOAL = 1
    TERM_REASON_CONTACT = 2
    TERM_REASON_LARGE_ANGLE = 3
    TERM_REASON_TERRAIN_FALL = 4

    def __init__(self, cfg: RobotNavigationGoalCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.env = env
        self.robot: Articulation = env.scene[cfg.asset_name]

        # Terrain configuration
        terrain_cfg = env.scene.terrain.cfg.terrain_generator
        self.num_terrain_rows = terrain_cfg.num_rows
        self.num_terrain_cols = terrain_cfg.num_cols
        self.terrain_size = terrain_cfg.size[0]

        # Initialize buffers
        self._init_command_buffers()
        self._init_tracking_buffers()
        self._init_metrics()
        self._init_stair_curriculum()

        # Position sampling (lazy initialization)
        self._sampling_initialized = False
        self._position_sampler: Optional[PositionSampler] = None

    def _init_command_buffers(self):
        """Initialize command state buffers."""
        # Goal in body frame: [direction_x, direction_y, direction_z, log_distance]
        self.goal_command_body = torch.zeros(self.num_envs, 4, device=self.device)
        self.goal_command_body_unscaled = torch.ones(self.num_envs, 3, device=self.device)

        # World frame positions
        self.goal_position_world = torch.zeros(self.num_envs, 3, device=self.device)
        self.goal_position_world[:, 2] = 0.5

        self.spawn_position_world = torch.zeros(self.num_envs, 3, device=self.device)
        self.spawn_position_world[:, 2] = 0.5
        self.spawn_position_world[:, :2] = self.env.scene.env_origins[:, :2]

        self.spawn_heading_world = torch.zeros(self.num_envs, device=self.device)

    def _init_tracking_buffers(self):
        """Initialize goal tracking buffers."""
        self.steps_at_goal = torch.zeros(self.num_envs, device=self.device)
        self.time_at_goal = torch.zeros(self.num_envs, device=self.device)
        self.required_steps_at_goal = 4.0 / self.env.step_dt

        self.initial_distance_to_goal = torch.zeros(self.num_envs, device=self.device)
        self.distance_to_goal = torch.zeros(self.num_envs, device=self.device)
        self.closest_distance_to_goal = torch.zeros(self.num_envs, device=self.device)

        self.total_distance_traveled = torch.zeros(self.num_envs, device=self.device)
        self.previous_position = torch.zeros(self.num_envs, 3, device=self.device)

        self.goal_reach_count = torch.zeros(self.num_envs, device=self.device)
        self.success_tracker = SuccessRateTracker(self.num_envs, self.device, buffer_size=10)
        self.success_rate_buffer = torch.full((self.num_envs, 10), -1.0, device=self.device)
        # Last termination reason per env (-1 = unknown/not yet set).
        self.last_termination_reason = torch.full((self.num_envs,), -1, dtype=torch.int8, device=self.device)

    def mark_termination_reason(self, env_ids: torch.Tensor, reason: int):
        """Record the latest episode termination reason for specified envs."""
        if env_ids.numel() == 0:
            return
        self.last_termination_reason[env_ids] = int(reason)

    def _init_metrics(self):
        """Initialize performance metrics."""
        self.metrics["velocity_toward_goal"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["velocity_magnitude"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["success_rate"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["stair_curriculum_level"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["stair_curriculum_height"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["stair_success_rate"] = torch.zeros(self.num_envs, device=self.device)

    def _init_stair_curriculum(self):
        """Initialize state for the stairs-specific goal curriculum."""
        # Discrete curriculum level (0 = easiest). The unlocked height for the
        # current level is start + level * increment (capped by max_height).
        self.stair_curriculum_level = 0
        self.stair_curriculum_max_level = max(
            0,
            int(round(
                (float(self.cfg.stair_curriculum_max_height) - float(self.cfg.stair_curriculum_start_height))
                / max(float(self.cfg.stair_curriculum_increment), 1e-6)
            )),
        )
        self.stair_curriculum_height = self._stair_height_for_level(self.stair_curriculum_level)

        # Per-env flag: True if the env's current goal was placed on a staircase.
        self.stair_goal_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Global rolling buffer of stair-goal outcomes (-1 = empty, 0 = fail, 1 = success).
        buf_size = int(self.cfg.stair_curriculum_buffer_size)
        self.stair_success_buffer = torch.full((buf_size,), -1.0, device=self.device)
        self.stair_success_write = 0

        # Minimum sampled height (m) for a goal to count as "on the staircase". Goals
        # sampled on the flat walkway of a stairs tile (z ~ 0) are NOT snapped, so we
        # do not convert good flat-ground goals into stair goals.
        self._stair_min_step_height = 0.05

        # Throttle counter so the periodic status print does not spam the console.
        self._stair_eval_count = 0

        # Populated once terrain data is available (see _precompute_stair_data).
        self._is_stairs_terrain: Optional[torch.Tensor] = None
        self._stair_platform_positions: dict[int, torch.Tensor] = {}
        self._stair_platform_heights: dict[int, torch.Tensor] = {}

    # =========================================================================
    # Command Interface
    # =========================================================================

    def __str__(self) -> str:
        return f"NavigationGoalCommand:\n\tCommand dimension: {tuple(self.command.shape[1:])}\n"

    @property
    def command(self) -> torch.Tensor:
        return self.goal_command_body

    def _get_unscaled_command(self) -> torch.Tensor:
        return self.goal_command_body_unscaled

    # =========================================================================
    # Position Sampling
    # =========================================================================

    def _initialize_position_sampling(self):
        """Initialize position sampling from terrain-provided masks."""
        if self._sampling_initialized:
            return

        # Get terrain data from scene.terrain (stored by patches system)
        terrain = self.env.scene.terrain

        # Check for height field data
        heights_raw = getattr(terrain, '_height_field_visual', None)
        valid_mask_raw = getattr(terrain, '_height_field_valid_mask', None)
        platform_mask_raw = getattr(terrain, '_height_field_platform_mask', None)
        spawn_mask_raw = getattr(terrain, '_height_field_spawn_mask', None)

        if heights_raw is None or valid_mask_raw is None:
            raise ValueError(
                "No height field data found on terrain. "
                "Ensure add_goal=True is set in terrain configuration and patches are applied."
            )

        # Move to device
        heights = heights_raw.to(self.device)
        valid_mask = valid_mask_raw.to(self.device)

        # Platform mask defaults to empty if not provided
        if platform_mask_raw is not None:
            platform_mask = platform_mask_raw.to(self.device)
        else:
            platform_mask = torch.zeros_like(valid_mask)

        # Spawn mask defaults to valid_mask if not provided
        if spawn_mask_raw is not None:
            spawn_mask = spawn_mask_raw.to(self.device)
        else:
            spawn_mask = valid_mask  # Fall back to goal mask

        # Get terrain configuration parameters
        terrain_cfg = self.env.scene.terrain.cfg.terrain_generator
        horizontal_scale = terrain_cfg.horizontal_scale
        # Note: border_width for height_field_to_mesh comes from sub-terrain config (HfTerrainBaseCfg),
        # NOT from TerrainGeneratorCfg. Sub-terrain configs default to border_width=0.0.
        # TerrainGeneratorCfg.border_width (e.g., 30.0) is for the outer grid border, not per-tile.
        sub_terrain_border_width = 0.0  # Default from HfTerrainBaseCfg

        # Create sampler with both goal (valid_mask) and spawn (spawn_mask) masks
        self._position_sampler = PositionSampler(
            heights=heights,
            valid_mask=valid_mask,
            platform_mask=platform_mask,
            terrain_size=self.terrain_size,
            horizontal_scale=horizontal_scale,
            device=self.device,
            spawn_mask=spawn_mask,
            border_width=sub_terrain_border_width,
        )

        # Pre-compute stairs-tile metadata for the goal curriculum.
        self._precompute_stair_data()

        self._sampling_initialized = True

    def _precompute_stair_data(self):
        """Pre-compute which terrain tiles are staircases and their step data.

        A tile is considered a staircase iff it has any platform_mask cells
        (only ``StairGenerator`` produces platform_mask). For each such tile we
        cache the valid staircase-step (x, y) indices and their heights in meters
        so goals that land on the staircase can be snapped to a specific step
        height by the curriculum. Candidate cells are the *raised* step cells
        (height > 0), not just the top platform, so all step heights are available.
        """
        sampler = self._position_sampler
        platform = sampler.platform_mask  # (num_terrains, W, H) bool
        heights = sampler.heights          # (num_terrains, W, H) raw height field
        valid = sampler.valid_mask         # (num_terrains, W, H) bool
        num_terrains = platform.shape[0]

        # Cache sampler geometry for local-coordinate conversion.
        self._sampler_cell = sampler.cell_size
        self._sampler_border = sampler.border_pixels
        self._sampler_center = sampler.mesh_center

        self._is_stairs_terrain = platform.view(num_terrains, -1).any(dim=1)
        self._stair_platform_positions = {}
        self._stair_platform_heights = {}

        for t in torch.nonzero(self._is_stairs_terrain, as_tuple=False).flatten().tolist():
            t = int(t)
            # Staircase-step cells = raised, valid cells. In a stairs tile only the
            # staircase is raised; the walkway stays at height 0. Using every raised
            # step (not just the top platform) gives the curriculum a full range of
            # heights to snap to.
            mask_t = valid[t] & (heights[t] > 0)
            pos = mask_t.nonzero(as_tuple=False)  # (N, 2)
            if pos.shape[0] == 0:
                # Fallback to the top platform if step cells were dilated away.
                pos = (platform[t] & valid[t]).nonzero(as_tuple=False)
            if pos.shape[0] == 0:
                pos = platform[t].nonzero(as_tuple=False)
            if pos.shape[0] == 0:
                self._is_stairs_terrain[t] = False
                continue
            h = heights[t, pos[:, 0], pos[:, 1]].float() * VERTICAL_SCALE
            self._stair_platform_positions[t] = pos
            self._stair_platform_heights[t] = h

    def _compute_stair_goals(
        self, terrain_indices: torch.Tensor, orig_goal_z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute snapped goal positions for goals that landed on a staircase.

        Only goals whose originally-sampled height is above the walkway (i.e. that
        already landed on a raised staircase step) are snapped; goals sampled on
        the flat walkway of a stairs tile are left untouched so flat-ground goals
        are not converted into stair goals. For the snapped goals, the target step
        is the one whose height is closest to the current curriculum height (capped
        by the tallest step of that particular staircase).

        Args:
            terrain_indices: Terrain index per env being resampled, shape (n,).
            orig_goal_z: Originally-sampled goal height (meters) per env, shape (n,).

        Returns:
            Tuple ``(stair_mask, x, y, z)`` each shape (n,). ``stair_mask`` marks
            which entries were snapped; x/y/z are local coordinates (meters) that
            are only meaningful where ``stair_mask`` is True.
        """
        n = terrain_indices.shape[0]
        stair_mask = self._is_stairs_terrain[terrain_indices].clone()
        # Only snap goals that already landed on a raised staircase step; leave
        # flat-walkway goals (z ~ 0) as-is so we don't create extra stair goals.
        stair_mask &= orig_goal_z > self._stair_min_step_height
        sx = torch.zeros(n, device=self.device)
        sy = torch.zeros(n, device=self.device)
        sz = torch.zeros(n, device=self.device)

        for i in torch.nonzero(stair_mask, as_tuple=False).flatten().tolist():
            tidx = int(terrain_indices[i].item())
            pos = self._stair_platform_positions.get(tidx)
            if pos is None or pos.shape[0] == 0:
                stair_mask[i] = False
                continue

            target = self._stair_height_for_level(self._sample_stair_level())
            lx, ly, lz = self._pick_stair_cell(tidx, target)
            sx[i] = lx
            sy[i] = ly
            sz[i] = lz

        return stair_mask, sx, sy, sz

    def _sample_stair_level(self) -> int:
        """Sample a curriculum level: 50% the current top level, 50% a review level.

        The review level is drawn uniformly from [0, current_level] so lower steps
        keep appearing after the curriculum advances.
        """
        if float(torch.rand(1, device=self.device).item()) < 0.5:
            return self.stair_curriculum_level
        return int(torch.randint(0, self.stair_curriculum_level + 1, (1,), device=self.device).item())

    def _pick_stair_cell(self, tidx: int, target: float, ref_xy=None):
        """Pick a staircase cell whose height is closest to ``target``.

        Among the steps nearest to the target curriculum height, either a random
        one is chosen (``ref_xy`` is None) or the one closest to ``ref_xy`` (used
        when converting a flat goal onto the nearest step).

        Returns:
            Local ``(x, y, z)`` coordinates in meters.
        """
        pos = self._stair_platform_positions[tidx]
        h = self._stair_platform_heights[tidx]

        # Cap the target by the tallest step available on this staircase, then keep
        # the steps whose height is closest to it.
        t = min(target, float(h.max().item()))
        diff = (h - t).abs()
        min_diff = diff.min()
        candidates = torch.nonzero(diff <= min_diff + 1e-3, as_tuple=False).flatten()

        cand_x = (pos[candidates, 0].float() + self._sampler_border) * self._sampler_cell - self._sampler_center
        cand_y = (pos[candidates, 1].float() + self._sampler_border) * self._sampler_cell - self._sampler_center
        if ref_xy is None:
            j = int(torch.randint(len(candidates), (1,), device=self.device).item())
        else:
            d2 = (cand_x - ref_xy[0]) ** 2 + (cand_y - ref_xy[1]) ** 2
            j = int(d2.argmin().item())
        return float(cand_x[j].item()), float(cand_y[j].item()), float(h[candidates[j]].item())

    def _augment_stair_goals(self, env_ids_tensor, terrain_indices, stair_mask, goal_x, goal_y, goal_z):
        """Top up the number of stair goals to ``stair_curriculum_min_goals``.

        Flat-walkway goals inside stairs tiles are converted onto the nearest step
        at the corresponding curriculum height until the global stair-goal count
        reaches the configured minimum (or no more flat candidates remain).

        Returns the (possibly modified) ``goal_x, goal_y, goal_z, stair_mask``.
        """
        min_goals = int(self.cfg.stair_curriculum_min_goals)
        if min_goals <= 0:
            return goal_x, goal_y, goal_z, stair_mask

        # Global stair-goal count once this batch's current mask is applied.
        tentative = self.stair_goal_mask.clone()
        tentative[env_ids_tensor] = stair_mask
        deficit = min_goals - int(tentative.sum().item())
        if deficit <= 0:
            return goal_x, goal_y, goal_z, stair_mask

        # Convertible goals: inside a stairs tile but currently on the flat walkway.
        is_stairs = self._is_stairs_terrain[terrain_indices]
        flat_candidates = torch.nonzero(is_stairs & ~stair_mask, as_tuple=False).flatten()
        if flat_candidates.numel() == 0:
            return goal_x, goal_y, goal_z, stair_mask

        # Randomly pick up to ``deficit`` of them and snap each onto the nearest step.
        perm = flat_candidates[torch.randperm(flat_candidates.numel(), device=self.device)]
        for i in perm[:deficit].tolist():
            tidx = int(terrain_indices[i].item())
            pos = self._stair_platform_positions.get(tidx)
            if pos is None or pos.shape[0] == 0:
                continue
            target = self._stair_height_for_level(self._sample_stair_level())
            lx, ly, lz = self._pick_stair_cell(
                tidx, target, ref_xy=(float(goal_x[i].item()), float(goal_y[i].item()))
            )
            goal_x[i] = lx
            goal_y[i] = ly
            goal_z[i] = lz
            stair_mask[i] = True

        return goal_x, goal_y, goal_z, stair_mask

    def _record_stair_results(self, results: torch.Tensor):
        """Append stair-goal outcomes (0/1 float tensor) to the rolling buffer."""
        n = results.shape[0]
        if n == 0:
            return
        buf_size = self.stair_success_buffer.shape[0]
        idx = (torch.arange(n, device=self.device) + self.stair_success_write) % buf_size
        self.stair_success_buffer[idx] = results.float()
        self.stair_success_write = int((self.stair_success_write + n) % buf_size)

    def _stair_success_rate(self) -> float:
        """Return the current success rate over recorded stair goals."""
        filled = self.stair_success_buffer >= 0
        if not bool(filled.any()):
            return 0.0
        return float((self.stair_success_buffer[filled] > 0).float().mean().item())

    def _stair_height_for_level(self, level: int) -> float:
        """Convert a curriculum level to a target step height (meters)."""
        height = float(self.cfg.stair_curriculum_start_height) + level * float(self.cfg.stair_curriculum_increment)
        return min(height, float(self.cfg.stair_curriculum_max_height))

    def _maybe_advance_stair_curriculum(self):
        """Raise the target step level once stair goals are solved reliably.

        Also periodically prints the current curriculum level and success rate.
        """
        filled = self.stair_success_buffer >= 0
        num_filled = int(filled.sum().item())
        if num_filled < int(self.cfg.stair_curriculum_min_samples):
            return

        rate = float((self.stair_success_buffer[filled] > 0).float().mean().item())
        threshold = float(self.cfg.stair_curriculum_success_threshold)
        at_max = self.stair_curriculum_level >= self.stair_curriculum_max_level

        if rate > threshold and not at_max:
            old_level = self.stair_curriculum_level
            old_height = self.stair_curriculum_height
            self.stair_curriculum_level += 1
            self.stair_curriculum_height = self._stair_height_for_level(self.stair_curriculum_level)
            # Reset the buffer so the next level is evaluated from scratch.
            self.stair_success_buffer[:] = -1.0
            self.stair_success_write = 0
            self._stair_eval_count = 0
            print(
                f"[楼梯课程] 升级: 等级 {old_level} -> {self.stair_curriculum_level} "
                f"(上限 {self.stair_curriculum_max_level}), "
                f"目标最高台阶高度 {old_height:.2f}m -> {self.stair_curriculum_height:.2f}m, "
                f"成功率 {rate * 100:.1f}% ({num_filled} 样本)",
                flush=True,
            )
            return

        # Periodic status print (throttled) when not advancing.
        self._stair_eval_count += 1
        if self._stair_eval_count % 50 == 0:
            status = "已达最高等级" if at_max else "未达升级阈值"
            print(
                f"[楼梯课程] 当前等级 {self.stair_curriculum_level}/{self.stair_curriculum_max_level}, "
                f"目标最高台阶高度 {self.stair_curriculum_height:.2f}m, "
                f"成功率 {rate * 100:.1f}% ({num_filled} 样本, 阈值 {threshold * 100:.0f}%, {status})",
                flush=True,
            )

    def _get_terrain_indices(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Get terrain indices for given environment IDs.

        The terrain index formula depends on the generation order:
        - curriculum=True:  column-major (row + col * num_rows)
        - curriculum=False: row-major (row * num_cols + col)

        Note: terrain_levels corresponds to row, terrain_types to column.
        """
        terrain = self.env.scene.terrain
        levels = terrain.terrain_levels[env_ids]  # row
        types = terrain.terrain_types[env_ids]    # col

        # Check if curriculum mode
        terrain_cfg = self.env.scene.terrain.cfg.terrain_generator
        if terrain_cfg.curriculum:
            # Column-major order (curriculum mode iterates: for col: for row:)
            return levels + types * self.num_terrain_rows
        else:
            # Row-major order (random mode uses np.unravel_index with (num_rows, num_cols))
            return levels * self.num_terrain_cols + types

    # =========================================================================
    # Command Sampling and Update
    # =========================================================================

    def _resample_command(self, env_ids: Sequence[int]):
        """Sample new goal and spawn positions.

        Goal positions use valid_mask (smaller padding, robot just needs to reach).
        Spawn positions use spawn_mask (larger padding for robot body clearance).
        """
        self._initialize_position_sampling()

        if isinstance(env_ids, torch.Tensor):
            env_ids_tensor = env_ids.clone().to(device=self.device, dtype=torch.long)
        else:
            env_ids_tensor = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        # Reset tracking
        self._reset_tracking_state(env_ids)

        # Get terrain indices
        terrain_indices = self._get_terrain_indices(env_ids_tensor)

        # Sample goal positions (from valid_mask with smaller padding)
        goal_x, goal_y, goal_z = self._position_sampler.sample(terrain_indices)
        # Sample spawn positions (from spawn_mask with larger padding for robot body)
        spawn_x, spawn_y, spawn_z = self._position_sampler.sample_spawn(terrain_indices)

        # Stairs curriculum: snap goals that land on a staircase to a specific
        # step height that grows with the curriculum.
        if self.cfg.stair_curriculum_enabled:
            stair_mask, stair_x, stair_y, stair_z = self._compute_stair_goals(terrain_indices, goal_z)
            goal_x = torch.where(stair_mask, stair_x, goal_x)
            goal_y = torch.where(stair_mask, stair_y, goal_y)
            goal_z = torch.where(stair_mask, stair_z, goal_z)
            # Ensure at least ``stair_curriculum_min_goals`` stair goals globally by
            # converting nearby flat-walkway goals in stairs tiles into stair goals.
            goal_x, goal_y, goal_z, stair_mask = self._augment_stair_goals(
                env_ids_tensor, terrain_indices, stair_mask, goal_x, goal_y, goal_z
            )
            self.stair_goal_mask[env_ids_tensor] = stair_mask
        else:
            self.stair_goal_mask[env_ids_tensor] = False

        # Convert to world coordinates
        terrain = self.env.scene.terrain
        levels = terrain.terrain_levels[env_ids]
        types = terrain.terrain_types[env_ids]
        terrain_origins = terrain.terrain_origins[levels, types]

        # Update goal position
        self.goal_position_world[env_ids, 0] = terrain_origins[:, 0] + goal_x
        self.goal_position_world[env_ids, 1] = terrain_origins[:, 1] + goal_y
        height_offset = torch.rand(len(env_ids), device=self.device) * 0.6 + 0.2
        self.goal_position_world[env_ids, 2] = goal_z + height_offset

        # Small spawn height offset to prevent clipping into terrain
        # Note: The robot's default_root_state already includes standing height (~0.5m)
        spawn_offset = 0.05

        # Update spawn/env origin
        terrain.env_origins[env_ids, 0] = terrain_origins[:, 0] + spawn_x
        terrain.env_origins[env_ids, 1] = terrain_origins[:, 1] + spawn_y
        terrain.env_origins[env_ids, 2] = spawn_z + spawn_offset

        # Track spawn position
        self.spawn_position_world[env_ids, 0] = terrain_origins[:, 0] + spawn_x
        self.spawn_position_world[env_ids, 1] = terrain_origins[:, 1] + spawn_y
        self.spawn_position_world[env_ids, 2] = spawn_z + spawn_offset

        # Initialize distance metrics
        self.initial_distance_to_goal[env_ids] = torch.norm(
            self.robot.data.root_pos_w[env_ids] - self.goal_position_world[env_ids], dim=1
        )
        self.closest_distance_to_goal[env_ids] = self.initial_distance_to_goal[env_ids]

    def _reset_tracking_state(self, env_ids: Sequence[int]):
        """Reset tracking state for specified environments."""
        self.steps_at_goal[env_ids] = 0
        self.time_at_goal[env_ids] = 0
        self.total_distance_traveled[env_ids] = 0.0
        self.previous_position[env_ids] = self.robot.data.root_pos_w[env_ids].clone()

    def _update_command(self):
        """Update command in body frame."""
        # Transform goal to body frame
        inverse_pos, inverse_rot = subtract_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w
        )
        goal_in_body = transform_points(
            self.goal_position_world.unsqueeze(1),
            inverse_pos,
            inverse_rot
        ).squeeze(1)

        self.goal_command_body_unscaled = goal_in_body.clone()

        # Normalized direction and log distance
        distance = torch.norm(goal_in_body, dim=-1, keepdim=True)
        direction = goal_in_body / torch.clamp(distance, min=1e-6)
        log_distance = torch.log(distance + 1.0)

        self.goal_command_body[:, :3] = direction
        self.goal_command_body[:, 3:] = log_distance

        self._update_distance_tracking()

    def _update_distance_tracking(self):
        """Update distance metrics."""
        self.distance_to_goal = torch.norm(
            self.robot.data.root_pos_w - self.goal_position_world, dim=1
        )
        self.closest_distance_to_goal = torch.min(
            self.closest_distance_to_goal, self.distance_to_goal
        )

        step_distance = torch.norm(
            self.robot.data.root_pos_w - self.previous_position, dim=1
        )
        self.total_distance_traveled += step_distance
        self.previous_position = self.robot.data.root_pos_w.clone()

    def _resample_spawn_positions(self, env_ids: Sequence[int]):
        """Update spawn position tracking."""
        self.spawn_position_world[env_ids, :2] = self.env.scene.env_origins[env_ids, :2]

    # =========================================================================
    # Metrics and Reset
    # =========================================================================

    def _update_metrics(self):
        """Update performance metrics."""
        position_error = self.goal_position_world - self.robot.data.root_pos_w[:, :3]
        position_error_2d = position_error[:, :2]
        velocity_2d = self.robot.data.root_state_w[:, 7:9]

        self.metrics["velocity_magnitude"] = torch.norm(velocity_2d, dim=1)

        direction_to_goal = position_error_2d / torch.clamp(torch.norm(position_error_2d, dim=1, keepdim=True), min=1e-6)
        self.metrics["velocity_toward_goal"] = (velocity_2d * direction_to_goal).sum(dim=1)
        self.metrics["success_rate"] = self.success_tracker.get_success_rate()
        self.metrics["stair_curriculum_level"][:] = float(self.stair_curriculum_level)
        self.metrics["stair_curriculum_height"][:] = self.stair_curriculum_height
        self.metrics["stair_success_rate"][:] = self._stair_success_rate()

        # 实时跟踪一个楼梯目标环境：若当前观测 env 不再是楼梯目标，
        # 立即切换到新的楼梯目标并继续每步打印。
        if self.cfg.stair_curriculum_enabled:
            if not hasattr(self, "_watch_stair_env"):
                self._watch_stair_env = -1

            def _is_valid_stair_watch(eid: int) -> bool:
                if eid < 0:
                    return False
                if not bool(self.stair_goal_mask[eid].item()):
                    return False
                t_idx = int(self._get_terrain_indices(torch.tensor([eid], device=self.device, dtype=torch.long))[0].item())
                return bool(self._is_stairs_terrain[t_idx].item())

            # 若未锁定，或当前观测 env 已不在楼梯目标上，则重新选择。
            need_switch = not _is_valid_stair_watch(self._watch_stair_env)
            if need_switch:
                stair_ids = self.stair_goal_mask.nonzero(as_tuple=False).flatten()
                self._watch_stair_env = -1
                for sid in stair_ids.tolist():
                    sid = int(sid)
                    if _is_valid_stair_watch(sid):
                        self._watch_stair_env = sid
                        break

            if _is_valid_stair_watch(self._watch_stair_env):
                eid = self._watch_stair_env
                t_idx = int(self._get_terrain_indices(torch.tensor([eid], device=self.device, dtype=torch.long))[0].item())
                dist = float(self.distance_to_goal[eid].item())
                reached = bool((self.time_at_goal[eid] > 0.0).item())
                verdict = float(self.success_tracker.last_result[eid].item())
                verdict_str = "成功" if verdict > 0.0 else ("失败" if verdict == 0.0 else "未结算")
                print(
                    f"[楼梯实时] env{eid}(楼梯,tile={t_idx}) 距目标{dist:.2f}m "
                    f"{'已到达' if reached else '未到达'} 判定:{verdict_str}",
                    flush=True,
                )

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        """Reset command generator and compute episode metrics."""
        if env_ids is None:
            env_ids_tensor = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        elif isinstance(env_ids, torch.Tensor):
            env_ids_tensor = env_ids.to(device=self.device, dtype=torch.long)
        else:
            env_ids_tensor = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        metrics_obs = self.env.observation_manager.compute_group(group_name="metrics")
        success = metrics_obs["in_goal"][env_ids_tensor].squeeze(-1)
        failed = ~success

        # Update legacy success rate buffer
        self.success_rate_buffer[env_ids_tensor] = torch.roll(
            self.success_rate_buffer[env_ids_tensor], 1, dims=1
        )
        rate = success.float() - failed.float()
        rate[rate == 0] = -1
        self.success_rate_buffer[env_ids_tensor, 0] = rate

        # Diagnostic stats requested by user:
        # flat goal (non-stair) with final distance in (0.5m, 1.5m).
        stair_sel_all = self.stair_goal_mask[env_ids_tensor]
        flat_sel = ~stair_sel_all
        dist_end_all = self.distance_to_goal[env_ids_tensor]
        near_band_sel = (dist_end_all > 0.5) & (dist_end_all < 1.5)
        flat_near_band_sel = flat_sel & near_band_sel

        timeout_sel = self.last_termination_reason[env_ids_tensor] == self.TERM_REASON_TIMEOUT
        flat_near_band_timeout_sel = flat_near_band_sel & timeout_sel

        batch_size = int(env_ids_tensor.numel())
        flat_near_band_count = int(flat_near_band_sel.sum().item())
        flat_near_band_timeout_count = int(flat_near_band_timeout_sel.sum().item())
        timeout_count = int(timeout_sel.sum().item())

        flat_near_band_ratio = (flat_near_band_count / batch_size) if batch_size > 0 else 0.0
        timeout_flat_near_band_ratio = (
            flat_near_band_timeout_count / timeout_count if timeout_count > 0 else 0.0
        )

        print(
            f"[平地近终点统计] batch={batch_size} 平地且0.5<dist<1.5:{flat_near_band_count}"
            f" ({flat_near_band_ratio * 100:.1f}%) timeout内:{flat_near_band_timeout_count}"
            f"/{timeout_count} ({timeout_flat_near_band_ratio * 100:.1f}%)",
            flush=True,
        )

        # Stairs curriculum: record outcomes for envs whose goal was on a
        # staircase, then advance the target height if solved reliably. Must run
        # before _resample overwrites stair_goal_mask / clears time_at_goal.
        if self.cfg.stair_curriculum_enabled:
            stair_sel = self.stair_goal_mask[env_ids_tensor]
            if bool(stair_sel.any()):
                # Reuse the EXACT verdict the termination functions already computed
                # for this episode (stored per-env in success_tracker.last_result).
                # That verdict counts reaching the goal as success but a fall /
                # illegal contact / large tilt as failure, matching the official
                # navigation success. It is stricter than "time_at_goal > 0", which
                # would still count a robot that reached the step and then crashed
                # as a success. -1 means no termination result yet (skip it).
                outcome = self.success_tracker.last_result[env_ids_tensor]
                valid = stair_sel & (outcome >= 0.0)
                if bool(valid.any()):
                    self._record_stair_results(outcome[valid])

                # 调试打印：目标点刷在楼梯上的环境，打印其 env id、距离终点状态
                # (整回合最近距离 + 回合结束时距离) 以及是否成功 (是否到达过 + 最终判定)。
                eids = env_ids_tensor
                stair_eids = eids[stair_sel]
                dist_min = self.closest_distance_to_goal[stair_eids]
                dist_end = self.distance_to_goal[stair_eids]
                reached = self.time_at_goal[stair_eids] > 0.0
                verdict = self.success_tracker.last_result[stair_eids]
                max_print = min(stair_eids.numel(), 10)
                parts = []
                for k in range(max_print):
                    v = float(verdict[k].item())
                    v_str = "成功" if v > 0 else ("失败" if v == 0 else "无结果")
                    parts.append(
                        f"env{int(stair_eids[k].item())}(最近{float(dist_min[k].item()):.2f}m,"
                        f"结束{float(dist_end[k].item()):.2f}m,"
                        f"到达{'是' if bool(reached[k].item()) else '否'},{v_str})"
                    )
                more = "" if stair_eids.numel() <= max_print else f" ...共{stair_eids.numel()}个"
                print("[楼梯调试] " + " ".join(parts) + more, flush=True)
            self._maybe_advance_stair_curriculum()
            # Clear the consumed verdicts so a later reset without a fresh
            # termination cannot double-count a stale result.
            self.success_tracker.last_result[env_ids_tensor] = -1.0

        # Reset command state
        self.command_counter[env_ids_tensor] = 0
        self._resample(env_ids_tensor)

        # Return mean metrics
        extras = {}
        for name, value in self.metrics.items():
            extras[name] = torch.mean(value[env_ids_tensor]).item()
            value[env_ids_tensor] = 0.0

        # Send flat near-goal diagnostics to TensorBoard.
        extras["flat_goal_0p5_1p5_count"] = float(flat_near_band_count)
        extras["flat_goal_0p5_1p5_ratio"] = float(flat_near_band_ratio)
        extras["flat_goal_0p5_1p5_timeout_count"] = float(flat_near_band_timeout_count)
        extras["flat_goal_0p5_1p5_timeout_ratio_in_timeouts"] = float(timeout_flat_near_band_ratio)
        extras["timeout_count"] = float(timeout_count)

        # Clear consumed reason codes for reset envs.
        self.last_termination_reason[env_ids_tensor] = -1

        return extras

    # =========================================================================
    # Success/Failure Tracking
    # =========================================================================

    def update_success(self, at_goal: torch.Tensor):
        self.goal_reach_count += at_goal.int()

    def update_failures(self, failed: torch.Tensor):
        self.goal_reach_count -= failed.int()

    # =========================================================================
    # Visualization
    # =========================================================================

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            self._setup_visualizers()
        else:
            self._hide_visualizers()

    def _setup_visualizers(self):
        """Create visualization markers."""
        if not hasattr(self, "goal_marker"):
            cfg = CUBOID_MARKER_CFG.copy()
            cfg.prim_path = "/Visuals/Command/goal_position"
            cfg.markers["cuboid"].size = (0.2, 0.2, 0.2)
            cfg.markers["cuboid"].visual_material.diffuse_color = (0.0, 0.0, 1.0)
            self.goal_marker = VisualizationMarkers(cfg)

        if not hasattr(self, "spawn_marker"):
            cfg = CUBOID_MARKER_CFG.copy()
            cfg.prim_path = "/Visuals/Command/spawn_position"
            cfg.markers["cuboid"].size = (0.2, 0.2, 0.2)
            cfg.markers["cuboid"].visual_material.diffuse_color = (1.0, 0.5, 0.0)
            self.spawn_marker = VisualizationMarkers(cfg)

        if not hasattr(self, "desired_velocity_marker"):
            cfg = GREEN_ARROW_X_MARKER_CFG.copy()
            cfg.prim_path = "/Visuals/Command/desired_velocity"
            cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
            self.desired_velocity_marker = VisualizationMarkers(cfg)

        if not hasattr(self, "current_velocity_marker"):
            cfg = RED_ARROW_X_MARKER_CFG.copy()
            cfg.prim_path = "/Visuals/Command/current_velocity"
            cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
            self.current_velocity_marker = VisualizationMarkers(cfg)

        self.goal_marker.set_visibility(True)
        self.spawn_marker.set_visibility(True)
        self.desired_velocity_marker.set_visibility(True)
        self.current_velocity_marker.set_visibility(True)

    def _hide_visualizers(self):
        for name in ["goal_marker", "spawn_marker", "desired_velocity_marker", "current_velocity_marker"]:
            if hasattr(self, name):
                getattr(self, name).set_visibility(False)

    def _debug_vis_callback(self, event):
        """Update visualization markers."""
        self.goal_marker.visualize(self.goal_position_world)
        self.spawn_marker.visualize(self.spawn_position_world)

        arrow_position = self.robot.data.root_pos_w.clone()
        arrow_position[:, 2] += 0.5

        desired_scale, desired_quat = self._compute_velocity_arrow(
            self.command[:, :3], is_goal_direction=True
        )
        self.desired_velocity_marker.visualize(arrow_position, desired_quat, desired_scale)

        current_scale, current_quat = self._compute_velocity_arrow(
            self.robot.data.root_lin_vel_b, is_goal_direction=False
        )
        self.current_velocity_marker.visualize(arrow_position, current_quat, current_scale)

    def _compute_velocity_arrow(
        self,
        velocity: torch.Tensor,
        is_goal_direction: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute arrow visualization parameters."""
        base_scale = torch.tensor(
            self.desired_velocity_marker.cfg.markers["arrow"].scale,
            device=self.device
        ).repeat(velocity.shape[0], 1)

        if not is_goal_direction:
            velocity = velocity.clone()
            velocity[:, 2] = 0.0

        base_scale[:, 0] *= torch.norm(velocity, dim=1) * 3.0
        quat = vec_to_quat(velocity)

        if is_goal_direction:
            quat = math_utils.quat_mul(self.robot.data.root_quat_w, quat)
        else:
            quat = math_utils.quat_mul(yaw_quat(self.robot.data.root_quat_w), quat)

        return base_scale, quat

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def _is_position_in_bounds(self, position: torch.Tensor) -> torch.Tensor:
        half_side = math.ceil(self.num_envs ** 0.5) * self.env.cfg.scene.env_spacing / 2
        return (position[:, :2].abs() < half_side).all(dim=1)

    def _clamp_to_bounds(self, position: torch.Tensor) -> torch.Tensor:
        origins = self.env.scene.terrain.terrain_origins.view(-1, 3)
        bounds_max = origins.max(dim=0)[0][:2]
        bounds_min = origins.min(dim=0)[0][:2]
        position[:, 0] = position[:, 0].clamp(bounds_min[0], bounds_max[0])
        position[:, 1] = position[:, 1].clamp(bounds_min[1], bounds_max[1])
        return position


# =============================================================================
# Legacy Aliases
# =============================================================================

RobotNavigationGoalCommand.pos_command_b = property(lambda self: self.goal_command_body)
RobotNavigationGoalCommand.pos_command_w = property(lambda self: self.goal_position_world)
RobotNavigationGoalCommand.pos_spawn_w = property(lambda self: self.spawn_position_world)
RobotNavigationGoalCommand.closes_distance_to_goal = property(
    lambda self: self.closest_distance_to_goal
)
RobotNavigationGoalCommand.time_at_goal_in_steps = property(lambda self: self.steps_at_goal)
RobotNavigationGoalCommand.required_time_at_goal_in_steps = property(
    lambda self: self.required_steps_at_goal
)
RobotNavigationGoalCommand.goal_reached_buffer = property(lambda self: self.success_tracker)
RobotNavigationGoalCommand.goal_reached_counter = property(lambda self: self.goal_reach_count)
RobotNavigationGoalCommand.distance_traveled = property(lambda self: self.total_distance_traveled)
RobotNavigationGoalCommand.previous_pos_3d = property(lambda self: self.previous_position)
