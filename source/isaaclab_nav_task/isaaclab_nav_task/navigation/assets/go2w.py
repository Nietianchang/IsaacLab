# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Configuration for the GO2W robot."""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# Local assets directory for this extension
_ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

__all__ = ["GO2W_CFG"]


GO2W_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # go2w_cyl3/ = foot collisions are analytic cylinders (radius 0.086, length 0.0518,
        # axis Y) with the CORRECT lateral offset (origin y=±0.0481) matching the wheel mesh
        # geometric center. go2w_cyl2 had the cylinders centered at y=0, which shifted all four
        # wheel contacts ~5cm inward and narrowed the lateral support base (caused vy tipping).
        # Converted with --merge-joints so config.yaml matches the original go2w/.
        usd_path=f"{_ASSETS_DIR}/Robots/go2w_cyl3/go2w.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=None,
            max_angular_velocity=None,
            max_depenetration_velocity=1.0,
            enable_gyroscopic_forces=True,
        ),
        # [PHYSX ALIGN] IsaacGym/NP3O training uses a GLOBAL physx contact_offset=0.01,
        # rest_offset=0.0 (legged_robot_config.py). IsaacLab leaves these at the per-collider
        # USD/PhysX default (contact_offset~0.02), which makes the wheel-ground contact engage
        # ~1cm earlier/softer than training -- a candidate for the lateral crab-walk sim2sim
        # gap (robot tips slowly instead of stepping). Override the robot colliders to match.
        # Overridable for sweeping via env var CONTACT_OFFSET.
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=float(os.getenv("CONTACT_OFFSET", "0.01")),
            rest_offset=0.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    # spawn=sim_utils.UrdfFileCfg(
    #     fix_base=False,
    #     merge_fixed_joints=True,
    #     replace_cylinders_with_capsules=False,
    #     asset_path=f"/home/a/daohang3102/IsaacLab/robot_lab/source/robot_lab/data/Robots/unitree/go2w_description/urdf/go2w_description.urdf",
    #     activate_contact_sensors=True,
    #     rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #         disable_gravity=False,
    #         retain_accelerations=False,
    #         linear_damping=0.0,
    #         angular_damping=0.0,
    #         max_linear_velocity=1000.0,
    #         max_angular_velocity=1000.0,
    #         max_depenetration_velocity=1.0,
    #     ),
    #     articulation_props=sim_utils.ArticulationRootPropertiesCfg(
    #         enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
    #     ),
    #     joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
    #         gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
    #     ),
    # ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.39),
        joint_pos={
            "FL_hip_joint": 0.1,
            "RL_hip_joint": 0.1,
            "FR_hip_joint": -0.1,
            "RR_hip_joint": -0.1,
            "FL_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "FR_thigh_joint": 0.8,
            "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
            "FL_foot_joint": 0.0,
            "RL_foot_joint": 0.0,
            "FR_foot_joint": 0.0,
            "RR_foot_joint": 0.0,
        },
    ),
    # Implicit actuator PD: the PhysX articulation drive computes the joint PD torques
    # internally from the position/velocity targets we send each step. The gains below MUST
    # match the PD gains used by the low-level policy controller
    # (navigation_env_cfg.py: explicit_p_gains / explicit_d_gains):
    #   legs (hip/thigh/calf): stiffness(p)=40.0, damping(d)=1.0   -> position control
    #   wheels (foot):         stiffness(p)=0.0,  damping(d)=0.5    -> velocity control
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*hip_joint", ".*thigh_joint"],
            effort_limit_sim=23.7,
            velocity_limit_sim=30.1,
            stiffness={".*": 40.0},
            damping={".*": 1.0},
        ),
        "legs_calf": ImplicitActuatorCfg(
            joint_names_expr=[".*calf_joint"],
            effort_limit_sim=35.55,
            velocity_limit_sim=20.07,
            stiffness={".*": 40.0},
            damping={".*": 1.0},
        ),
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=[".*foot_joint"],
            effort_limit_sim=23.7,
            velocity_limit_sim=30.1,
            stiffness={".*": 0.0},
            damping={".*": 0.5},
        ),
    },
    soft_joint_pos_limit_factor=0.9,
)
"""Configuration of GO2W robot using implicit actuator configs (position/velocity control)."""
