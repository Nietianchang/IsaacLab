import glob
import os

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_nav_task.navigation.navigation_env_cfg import NavigationEnvCfg
import isaaclab_nav_task.navigation.mdp as mdp

from isaaclab_nav_task.navigation.assets import GO2W_CFG, ISAACLAB_NAV_TASKS_ASSETS_DIR  # isort: skip


LEG_JOINT_NAMES = [".*hip_joint", ".*thigh_joint", ".*calf_joint"]
WHEEL_JOINT_NAMES = [".*foot_joint"]

# Use deterministic joint ordering to match NP3O export tensor layout.
LEG_JOINTS_ORDERED = [
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
LEG_JOINTS_OBS_ORDERED = [
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FL_foot_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FR_foot_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RL_foot_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RR_foot_joint",
]
WHEEL_JOINTS_ORDERED = ["FL_foot_joint", "FR_foot_joint", "RL_foot_joint", "RR_foot_joint"]

GO2W_PROPRIO_OBS_ORDER = [
    "base_ang_vel",
    "projected_gravity",
    "velocity_commands",
    "joint_pos",
    "joint_vel",
    "actions",
]


def _resolve_go2w_low_level_policy() -> str:
    """Resolve go2w low-level blind-walking policy file with graceful fallback."""
    go2w_policy_dir = os.path.join(ISAACLAB_NAV_TASKS_ASSETS_DIR, "Policies", "locomotion", "go2w")

    # Prefer NP3O go2w_constraint_him_b exports first.
    candidates = sorted(glob.glob(os.path.join(go2w_policy_dir, "*go2w*constraint*him_b*.pt")))
    if not candidates:
        candidates = sorted(glob.glob(os.path.join(go2w_policy_dir, "*constraint*him_b*.pt")))
    if not candidates:
        candidates = sorted(glob.glob(os.path.join(go2w_policy_dir, "*him_b*.pt")))
    if not candidates:
        candidates = sorted(glob.glob(os.path.join(go2w_policy_dir, "*.pt")))
    if candidates:
        return candidates[0]

    return os.path.join(
        ISAACLAB_NAV_TASKS_ASSETS_DIR,
        "Policies",
        "locomotion",
        "b2w",
        "policy_b2w_new_2.pt",
    )


@configclass
class GO2WNavigationEnvCfg(NavigationEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        from isaaclab_nav_task.navigation.mdp.depth_utils.camera_config import get_camera_config
        from isaaclab_nav_task.navigation.mdp.observations import initialize_depth_noise_generator

        initialize_depth_noise_generator(robot_name="go2w", use_jit_precompiled=False)

        camera_config = get_camera_config("go2w")
        CAMERA_RESOLUTION = camera_config.resolution

        self.scene.robot = GO2W_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.scene.raycast_camera.prim_path = "{ENV_REGEX_NS}/Robot/base"
        self.scene.raycast_camera.offset.pos = (0.27, 0.0, 0.03)
        self.scene.raycast_camera.offset.rot = (0.9990482215818578, 0.0, -0.043619387365336, 0.0)
        self.scene.height_scanner_critic.prim_path = "{ENV_REGEX_NS}/Robot/base"

        self.terminations.base_contact.params = {
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base", ".*hip", ".*thigh"]),
            "threshold": 1.0,
        }

        self.actions.velocity_command.low_level_position_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=LEG_JOINTS_ORDERED,
            scale=0.25,
            use_default_offset=True,
        )
        self.actions.velocity_command.low_level_velocity_action = mdp.JointVelocityActionCfg(
            asset_name="robot",
            joint_names=WHEEL_JOINTS_ORDERED,
            scale=5.0,
            use_default_offset=True,
        )
        self.actions.velocity_command.low_level_policy_file = _resolve_go2w_low_level_policy()
        self.actions.velocity_command.hip_scale_reduction = 0.5
        self.actions.velocity_command.explicit_pd_control = True
        self.actions.velocity_command.explicit_p_gains = [
            40.0,
            40.0,
            40.0,
            40.0,
            40.0,
            40.0,
            40.0,
            40.0,
            40.0,
            40.0,
            40.0,
            40.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
        self.actions.velocity_command.explicit_d_gains = [
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            0.5,
            0.5,
            0.5,
            0.5,
        ]
        self.actions.velocity_command.default_dof_pos = [
            0.1,
            0.8,
            -1.5,
            -0.1,
            0.8,
            -1.5,
            0.1,
            1.0,
            -1.5,
            -0.1,
            1.0,
            -1.5,
        ]
        self.actions.velocity_command.leg_action_idx = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]
        self.actions.velocity_command.wheel_action_idx = [3, 7, 11, 15]
        self.actions.velocity_command.hip_action_idx = [0, 3, 6, 9]
        self.actions.velocity_command.use_observation_history = True
        self.actions.velocity_command.history_length = 10
        self.actions.velocity_command.use_half_precision_inference = True
        self.rewards.joint_acc_l2_joint.params = {"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES+WHEEL_JOINT_NAMES)}

        # Match deploy_real_go2w_him_edit_beifen.py one-step input:
        # [ang_vel(3), gravity(3), cmd(3), qj(16), dqj(16), last_action(16)] => 57 dims.
        self.observations.low_level_policy.base_lin_vel = None
        self.observations.low_level_policy.enable_corruption = False
        self.events.randomize_low_pass_filter_alpha.params = {
            "alpha_range": (0.2, 0.2),
            "action_term": "velocity_command",
            "per_dimension": True,
            "alpha_range_vx": (0.2, 0.2),
            "alpha_range_vy": (0.2, 0.2),
            "alpha_range_omega": (0.2, 0.2),
        }
        # Align low-level policy input preprocessing with deployment pipeline.
        self.observations.low_level_policy.base_ang_vel.func = mdp.base_ang_vel_scaled
        self.observations.low_level_policy.base_ang_vel.params = {"scale": 0.25}
        self.observations.low_level_policy.joint_pos.func = mdp.joint_pos_rel_scaled_zero_wheels
        self.observations.low_level_policy.joint_pos.params = {
            "scale": 1.0,
            "wheel_joint_ids": [3, 7, 11, 15],
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS_OBS_ORDERED, preserve_order=True),
        }
        self.observations.low_level_policy.joint_vel.func = mdp.joint_vel_rel_scaled
        self.observations.low_level_policy.joint_vel.params = {
            "scale": 0.05,
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS_OBS_ORDERED, preserve_order=True),
        }
        # Align the command channel with deployment: cmd * [2.0, 2.0, 0.25] then yaw * 1.5
        # (net per-axis scaling [2.0, 2.0, 0.375]).
        self.observations.low_level_policy.velocity_commands.func = mdp.velocity_commands_scaled
        self.observations.low_level_policy.velocity_commands.params = {
            "action_name": "velocity_command",
            "scale": (2.0, 2.0, 0.25),
            "yaw_extra_scale": 1.0,
            # b2w-style command noise added to the ORIGINAL (unscaled) command before scaling,
            # slightly larger than b2w. Handled inside the func, so disable the ObsTerm-level noise.
            "noise_min": (-0.15, -0.15, -0.25),
            "noise_max": (0.15, 0.15, 0.25),
        }
        self.observations.low_level_policy.velocity_commands.noise = None
        # self.observations.low_level_policy.actions.params = {"action_term": "velocity_command"}
        self.scene.terrain.max_init_terrain_level = 10
        self.scene.terrain.terrain_generator.difficulty_range = [0.5, 1.0]
        self.scene.terrain.terrain_generator.curriculum = False

@configclass
class GO2WNavigationEnvCfg_PLAY(GO2WNavigationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 20
        self.scene.env_spacing = 2.5

        # [TEMP DEBUG] Flatten the terrain to isolate the blind go2w low-level policy from
        # terrain effects. We must NOT null out terrain_generator: the navigation goal command
        # generator reads num_rows/num_cols/size and the generated valid_mask from it (setting
        # it to None -> 'NoneType has no attribute num_rows'). Instead we keep the maze grid
        # (so all metadata + valid_mask are still produced) but set every sub-terrain's
        # wall_height to 0, which makes the geometry perfectly flat: no walls / steps / pits
        # for the robot to bump into. Remove this block to restore the real maze terrain.
        import copy

        gen = copy.deepcopy(self.scene.terrain.terrain_generator)
        gen.num_rows = 2
        gen.num_cols = 2
        # for sub in gen.sub_terrains.values():
        #     sub.wall_height = 0.0
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = gen
        self.scene.terrain.max_init_terrain_level = None

        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        # [LATERAL FIX] Match the IsaacGym/NP3O training contact physics exactly.
        # Lateral (vy) motion is SLIDING-dominated: the support wheels must grip sideways while
        # the legs crab-step. The IsaacLab DEFAULT terrain material is COMPLIANT (soft) contact
        # (compliant_contact_stiffness=5e5) with dynamic_friction=0.8 and restitution=0.1, and
        # the robot material is randomized to dynamic 0.7-1.0 with friction_combine_mode=
        # "multiply" -> effective lateral dynamic friction ~0.8*0.85=0.68 on a springy surface.
        # That lets the wheels skid sideways and the body slowly rolls over.
        # Training (legged_robot_config.py) uses RIGID contact, static=dynamic=1.0, restitution=0.
        # Replicate that here: rigid ground, friction 1.0/1.0, restitution 0, and lock the robot
        # material randomization to 1.0/1.0/0.0 so the multiplied effective friction is 1.0.
        # [DIAGNOSTIC] friction overridable via env var FRIC (default 1.0) to test whether the
        # lateral crab-walk stalls because the analytic-cylinder wheels grip sideways too hard
        # (isotropic mu) vs the real/IsaacGym wheels. Sweep FRIC=0.5/0.7 to see if sustained
        # lateral motion returns.
        _fric = float(os.getenv("FRIC", "1.0"))
        self.scene.terrain.physics_material.static_friction = _fric
        self.scene.terrain.physics_material.dynamic_friction = _fric
        self.scene.terrain.physics_material.restitution = 0.0
        self.scene.terrain.physics_material.compliant_contact_stiffness = 0.0
        self.scene.terrain.physics_material.compliant_contact_damping = 0.0
        self.events.physics_material.params["static_friction_range"] = (_fric, _fric)
        self.events.physics_material.params["dynamic_friction_range"] = (_fric, _fric)
        self.events.physics_material.params["restitution_range"] = (0.0, 0.0)
        print(
            "[LATERAL FIX] PLAY terrain material set -> "
            f"static={self.scene.terrain.physics_material.static_friction} "
            f"dynamic={self.scene.terrain.physics_material.dynamic_friction} "
            f"restitution={self.scene.terrain.physics_material.restitution} "
            f"compliant_stiffness={self.scene.terrain.physics_material.compliant_contact_stiffness} "
            f"compliant_damping={self.scene.terrain.physics_material.compliant_contact_damping}"
        )


@configclass
class GO2WNavigationEnvCfg_DEV(GO2WNavigationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 30
        self.scene.terrain.max_init_terrain_level = 10
        self.scene.terrain.terrain_generator.difficulty_range = [0.5, 1.0]
        self.scene.terrain.terrain_generator.curriculum = False