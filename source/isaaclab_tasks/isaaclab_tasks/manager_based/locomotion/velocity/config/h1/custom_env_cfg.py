# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

##
# Pre-defined configs
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG, ROUGH_TERRAINS_CFG_CUSTOM  # isort: skip


##
# Scene definition
##


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG_CUSTOM,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = MISSING

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        # attach_yaw_only=True,
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/pelvis/.*", history_length=3, track_air_time=True)
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.GaitCommandCfg(
        asset_name="robot",
        resampling_time_range=(4.0, 6.0),
        still_proportion=0.1,
        debug_vis=True,
        ranges=mdp.GaitCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), 
            lin_vel_y=(-1.0, 1.0), 
            ang_vel_yaw=(-1.0, 1.0),
            gait_frequency=(0.5, 1.5) # Example range for gait frequency in Hz
        ),
    )

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group, mirroring the original structure."""

        # 1. Projected Gravity: (num_envs, 3)
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            scale=1.0,  # Corresponds to cfg["normalization"]["gravity"]
            noise=Unoise(n_min=-0.05, n_max=0.05), # A reasonable default noise
        )
        
        # 2. Base Angular Velocity: (num_envs, 3)
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            scale=1.0, # Corresponds to cfg["normalization"]["ang_vel"]
            noise=Unoise(n_min=-0.2, n_max=0.2),
        )
        
        # 3. Velocity Commands: (num_envs, 3)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"}, 
        )
        
        # 4. & 5. Gait Phase (Sine/Cosine): (num_envs, 2)
        gait_phase = ObsTerm(
            func=mdp.gait_phase_sin_cos, # Our new function
            params={"command_name": "base_velocity"},
        )
        
        # 6. Relative Joint Positions: (num_envs, num_dof)
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, # (dof_pos - default_dof_pos)
            scale=1.0, # Corresponds to cfg["normalization"]["dof_pos"]
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        
        # 7. Relative Joint Velocities: (num_envs, num_dof)
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel, # equivalent to dof_vel
            scale=0.1, # Corresponds to cfg["normalization"]["dof_vel"]
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        
        # 8. Last Actions: (num_envs, num_actions)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            """Post-initialization checks."""
            # Enable noise and scaling
            self.enable_corruption = True
            # Concatenate all terms into a single flat observation tensor
            self.concatenate_terms = True

    @configclass
    class CriticCfg(PolicyCfg):
         # 1. Randomized Base Mass
        # base_mass = ObsTerm(
        #     func=mdp.body_mass,
        #     params={"asset_cfg": SceneEntityCfg("robot", body_names=".*torso_link")},
        #     scale=1.0,  
        # )
        
        # 2. True Base Linear Velocity (without noise)
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            scale=1.0, 
        )

        def __post_init__(self):
            self.enable_corruption = False
            # Concatenate all terms into a single flat observation tensor
            self.concatenate_terms = True

    # Define the observation group
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.4, 1.0),
            "dynamic_friction_range": (0.4, 1.0),
            "restitution_range": (0.0, 1.0),
            "num_buckets": 64,
        },
    )

    add_body_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.02, 0.02)},
        },
    )


    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    # reset_robot_joints = EventTerm(
    #     func=mdp.reset_joints_by_scale,
    #     mode="reset",
    #     params={
    #         "position_range": (0.8, 1.2),
    #         "velocity_range": (0.0, 0.0),
    #     },
    # )

    randomize_reset_joints = EventTerm(
        # func=mdp.reset_joints_by_scale,
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.2, 0.2),
            "velocity_range": (-0.5, 0.5),
        },
    )


    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )

    robot_joint_friction_and_armature = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": (0.9, 1.1),
            "armature_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 7.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )



# @configclass
# class RewardsCfg:
#     """Reward terms for the MDP."""

#     # -- task
#     track_lin_vel_xy_exp = RewTerm(
#         func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
#     )
#     track_ang_vel_z_exp = RewTerm(
#         func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
#     )
#     # -- penalties
#     lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
#     ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
#     dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
#     dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
#     action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
#     feet_air_time = RewTerm(
#         func=mdp.feet_air_time,
#         weight=0.125,
#         params={
#             "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_link"),
#             "command_name": "base_velocity",
#             "threshold": 0.5,
#         },
#     )
#     undesired_contacts = RewTerm(
#         func=mdp.undesired_contacts,
#         weight=-1.0,
#         params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_link"), "threshold": 1.0},
#     )
#     # -- optional penalties
#     flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0)
#     dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0)

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- Task and Survival Rewards --
    survival = RewTerm(func=mdp.is_alive, weight=0.5)
    
    # Combined tracking for linear velocity (X and Y)
    # Note: original had separate x and y terms. This combines them, which is standard practice.
    tracking_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp, 
        weight=2.0,  # Corresponds to tracking_lin_vel_x and tracking_lin_vel_y
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)} # std = sqrt(tracking_sigma)
    )
    tracking_ang_vel = RewTerm(
        func=mdp.track_ang_vel_z_exp, 
        weight=0.5, 
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    
    # -- Stability and Safety Penalties --
    base_height = RewTerm(
        func=mdp.base_height_l2, 
        weight=-5.0, 
        params={
            "target_height": 0.9,
            # Assumes a downward-facing ray-caster sensor is configured on the robot for terrain height
            "sensor_cfg": SceneEntityCfg("height_scanner") 
        }
    )
    orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.2)
    # Use body_lin_acc_l2 on the root body as the best replacement for root_acc
    root_acc = RewTerm(
        func=mdp.body_lin_acc_l2, 
        weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="pelvis")} # Use the name of your root body
    )

    # -- Effort and Joint Penalties --
    torques = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-4)
    dof_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1.0e-4)
    dof_acc = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-7)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1.0)
    power = RewTerm(func=mdp.positive_power_l1, weight=-2.0e-3) # Using new function
    torque_tiredness = RewTerm(func=mdp.torque_tiredness_l2, weight=-1.0e-2) # Using new function

    # -- Limit Penalties --
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
    # The following are disabled (weight 0) as per the original config scales
    dof_vel_limits = RewTerm(func=mdp.joint_vel_limits, weight=0.0, params={"soft_ratio": 1.0})
    torque_limits = RewTerm(func=mdp.joint_torque_limits, weight=0.0, params={"soft_ratio": 1.0}) # Using new function

    # -- Gait and Foot Penalties --
    feet_slip = RewTerm(
        func=mdp.feet_slip_l2, # Using new function
        weight=-0.1,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_link")} # Modify regex for your foot names
    )
    # Using feet_air_time as a substitute for the complex feet_swing reward
    feet_swing = RewTerm(
        func=mdp.feet_air_time,
        weight=3.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_link"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.3})
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    "torso_link",
                ],
            ),
            "threshold": 1.0,
        },
    )

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


##
# Environment configuration
##


@configclass
class LocomotionVelocityRoughEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False

from isaaclab_assets import H1_CFG_CUSTOM,H1_MINIMAL_CFG, H1_MINIMAL_CFG_CUSTOM, H1_CFG_CUSTOM_DFKI  # isort: skip


@configclass
class H1Rewards:
    """Reward terms for the MDP."""

    # -- Task and Survival Rewards --
    survival = RewTerm(func=mdp.is_alive, weight=0.5)
    
    # Combined tracking for linear velocity (X and Y)
    # Note: original had separate x and y terms. This combines them, which is standard practice.
    tracking_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp, 
        weight=2.0,  # Corresponds to tracking_lin_vel_x and tracking_lin_vel_y
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)} # std = sqrt(tracking_sigma)
    )
    tracking_ang_vel = RewTerm(
        func=mdp.track_ang_vel_z_exp, 
        weight=0.5, 
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    
    # -- Stability and Safety Penalties --
    base_height = RewTerm(
        func=mdp.base_height_l2, 
        weight=-5.0, 
        params={
            "target_height": 0.9,
            # Assumes a downward-facing ray-caster sensor is configured on the robot for terrain height
            "sensor_cfg": SceneEntityCfg("height_scanner") 
        }
    )
    orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.2)
    # Use body_lin_acc_l2 on the root body as the best replacement for root_acc
    root_acc = RewTerm(
        func=mdp.body_lin_acc_l2, 
        weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="pelvis")} # Use the name of your root body
    )


    # -- Effort and Joint Penalties --
    torques = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-4)
    dof_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1.0e-4)
    dof_acc = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-7)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1.0)
    power = RewTerm(func=mdp.positive_power_l1, weight=-2.0e-3) # Using new function
    torque_tiredness = RewTerm(func=mdp.torque_tiredness_l2, weight=-1.0e-2) # Using new function

    # -- Limit Penalties --
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
    # The following are disabled (weight 0) as per the original config scales
    dof_vel_limits = RewTerm(func=mdp.joint_vel_limits, weight=0.0, params={"soft_ratio": 1.0})
    torque_limits = RewTerm(func=mdp.joint_torque_limits, weight=0.0, params={"soft_ratio": 1.0}) # Using new function

    # -- Gait and Foot Penalties --
    feet_slip = RewTerm(
        func=mdp.feet_slip_l2, # Using new function
        weight=-0.1,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_link")} # Modify regex for your foot names
    )
    # Using feet_air_time as a substitute for the complex feet_swing reward
    feet_swing = RewTerm(
        func=mdp.feet_swing_phase_reward,  # Use the new function here
        weight=3.0,
        params={
            "swing_period": 0.2,
            "command_name": "base_velocity",  # Must match the name of your GaitCommand instance
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",   # Name of the ContactSensor in the scene
                # IMPORTANT: Must be in [left, right] order
                body_names=["left_ankle_link", "right_ankle_link"] 
            ),
            "contact_threshold": 1.0,
        },
    )

    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_.*", ".*_elbow_.*"])},
    )
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1, weight=-0.2, params={"asset_cfg": SceneEntityCfg("robot", joint_names="torso_joint")}
    )

@configclass
class H1RoughEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    rewards: H1Rewards = H1Rewards()

    def __post_init__(self):
        # post init of parent
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False


        # Scene
        # self.scene.robot = H1_CFG_CUSTOM_DFKI.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot = H1_CFG_CUSTOM.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Events
        self.events.base_external_force_torque.params["asset_cfg"].body_names = [".*torso_link"]
        self.events.reset_base.params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.2, 0.2),
                "y": (-0.1, 0.1),
                "z": (-0.0, 0.0),
                "roll": (-0.2, 0.2),
                "pitch": (-0.2, 0.2),
                "yaw": (-0.2, 0.2),
            },
        }

        # Terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = ".*torso_link"

@configclass
class H1RoughEnvCfg_PLAY(H1RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 32
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_yaw = (0.0, 1.0)
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.base_external_force_torque = None
        self.events.randomize_reset_joints.params = {
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        }
        self.events.reset_base.params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        }
