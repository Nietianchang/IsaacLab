# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""RSL-RL agent configurations for GO2W navigation tasks."""

from isaaclab.utils import configclass

from isaaclab_nav_task.navigation.config.rl_cfg import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class GO2WNavMDPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """MDPO runner configuration for GO2W navigation."""

    num_steps_per_env = 16
    max_iterations = 15000
    save_interval = 100
    logger = "wandb"
    seed = 60
    wandb_project = "isaaclab_nav_go2w"
    experiment_name = "go2w_navigation_mdpo"
    empirical_normalization = False
    reward_shifting_value = 0.05
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCriticSRU",
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        rnn_hidden_size=512,
        rnn_type="lstm_sru",
        rnn_num_layers=1,
        dropout=0.2,
        num_cameras=1,
        image_input_dims=(64, 5, 8),
        height_input_dims=(64, 7, 7),
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="MDPO",
        value_loss_coef=0.02,
        use_clipped_value_loss=True,
        clip_param=0.2,
        value_clip_param=0.2,
        entropy_coef=0.00375,
        num_learning_epochs=5,
        num_mini_batches=5,
        learning_rate=1.0e-3,
        schedule="exponential",
        gamma=0.999,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class GO2WNavMDPORunnerDevCfg(GO2WNavMDPORunnerCfg):
    """Development configuration for MDPO with reduced iterations."""

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 300
        self.experiment_name = "go2w_navigation_mdpo_dev"
        self.logger = "tensorboard"


@configclass
class GO2WNavPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner configuration for GO2W navigation."""

    num_steps_per_env = 16
    max_iterations = 15000
    save_interval = 200
    logger = "wandb"
    seed = 60
    wandb_project = "isaaclab_nav_go2w"
    experiment_name = "go2w_navigation_ppo"
    empirical_normalization = False
    reward_shifting_value = 0.05
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCriticSRU",
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        rnn_hidden_size=512,
        rnn_type="lstm_sru",
        rnn_num_layers=1,
        dropout=0.2,
        num_cameras=1,
        image_input_dims=(64, 5, 8),
        height_input_dims=(64, 7, 7),
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=0.02,
        use_clipped_value_loss=True,
        clip_param=0.2,
        value_clip_param=0.2,
        entropy_coef=0.00375,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.995,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class GO2WNavPPORunnerDevCfg(GO2WNavPPORunnerCfg):
    """Development configuration for PPO with reduced iterations."""

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 300
        self.experiment_name = "go2w_navigation_ppo_dev"
        self.logger = "tensorboard"
