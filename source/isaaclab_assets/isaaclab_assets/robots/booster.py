import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg, IdealPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg



"""
Configuration for the Booster T1 Humanoid robot
action_dims = 23
['AAHead_yaw', 'Left_Shoulder_Pitch', 'Right_Shoulder_Pitch', 'Waist', 'Head_pitch', 
'Left_Shoulder_Roll', 'Right_Shoulder_Roll', 'Left_Hip_Pitch', 'Right_Hip_Pitch', 
'Left_Elbow_Pitch', 'Right_Elbow_Pitch', 'Left_Hip_Roll', 'Right_Hip_Roll', 'Left_Elbow_Yaw', 
'Right_Elbow_Yaw', 'Left_Hip_Yaw', 'Right_Hip_Yaw', 'Left_Knee_Pitch', 'Right_Knee_Pitch', 
'Left_Ankle_Pitch', 'Right_Ankle_Pitch', 'Left_Ankle_Roll', 'Right_Ankle_Roll']
"""


BOOSTER_MODEL_DIR = "/home/user/IsaacLab/assets"

BOOSTER_T1_CFG = ArticulationCfg(
    actuator_value_resolution_debug_print=True,
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{BOOSTER_MODEL_DIR}/t1/t1.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.75),
        joint_pos={
            # Head
            "AAHead_yaw": 0.0,
            "Head_pitch": 0.0,
            # Arm
            ".*_Shoulder_Pitch": 0.2,
            "Left_Shoulder_Roll": -1.35,
            "Right_Shoulder_Roll": 1.35,
            ".*_Elbow_Pitch": 0.0,
            "Left_Elbow_Yaw": -0.5,
            "Right_Elbow_Yaw": 0.5,
            # Waist
            "Waist": 0.0,
            # Leg
            ".*_Hip_Pitch": -0.20,
            ".*_Hip_Roll": 0.0,
            ".*_Hip_Yaw": 0.0,
            ".*_Knee_Pitch": 0.4,
            ".*_Ankle_Pitch": -0.25,
            ".*_Ankle_Roll": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    # init_state=ArticulationCfg.InitialStateCfg(
    #     pos=(0.0, 0.0, 0.72),
    #     joint_pos={
    #         # Head
    #         "AAHead_yaw": 0.0,
    #         "Head_pitch": 0.0,
    #         # Arm
    #         ".*_Shoulder_Pitch": 0.0,
    #         "Left_Shoulder_Roll": 0.0,
    #         "Right_Shoulder_Roll": 0.0,
    #         ".*_Elbow_Pitch": 0.0,
    #         "Left_Elbow_Yaw": 0.0,
    #         "Right_Elbow_Yaw": 0.0,
    #         # Waist
    #         "Waist": 0.0,
    #         # Leg
    #         ".*_Hip_Pitch": -0.20,
    #         ".*_Hip_Roll": 0.0,
    #         ".*_Hip_Yaw": 0.0,
    #         ".*_Knee_Pitch": 0.4,
    #         ".*_Ankle_Pitch": -0.25,
    #         ".*_Ankle_Roll": 0.0,
    #     },
    #     joint_vel={".*": 0.0},
    # ),
    soft_joint_pos_limit_factor=0.95,
    actuators={
        "legs": IdealPDActuatorCfg(
            joint_names_expr=[
                ".*_Hip_Pitch",
                ".*_Hip_Roll",
                ".*_Hip_Yaw",
                ".*_Knee_Pitch",
                "Waist",
            ],
            effort_limit_sim={
                ".*_Hip_Pitch": 45.0,
                ".*_Hip_Roll": 30.0,
                ".*_Hip_Yaw": 30.0,
                ".*_Knee_Pitch": 60.0,
                "Waist": 30.0,
            },
            velocity_limit_sim={
                ".*_Hip_Pitch": 12.5,
                ".*_Hip_Roll": 10.9,
                ".*_Hip_Yaw": 10.9,
                ".*_Knee_Pitch": 11.7,
                "Waist": 10.88,
            },
            stiffness=200.0,
            damping=5.0,
            armature=0.01,
        ),
        "feet": IdealPDActuatorCfg(
            joint_names_expr=[".*_Ankle_Pitch", ".*_Ankle_Roll"],
            effort_limit_sim={".*_Ankle_Pitch": 24, ".*_Ankle_Roll": 15},
            velocity_limit_sim={".*_Ankle_Pitch": 18.8, ".*_Ankle_Roll": 12.4},
            stiffness=50.0,
            damping=1.0,
            armature=0.01,
        ),
        "arms": IdealPDActuatorCfg(
            joint_names_expr=[
                ".*_Shoulder_Pitch",
                ".*_Shoulder_Roll",
                ".*_Elbow_Pitch",
                ".*_Elbow_Yaw",
            ],
            effort_limit_sim=18.0,
            velocity_limit_sim=18.8,
            stiffness=40.0,
            damping=2.0, # i changed this from 10 to 2
            armature=0.01,
        ),
        "head": IdealPDActuatorCfg(
            joint_names_expr=["AAHead_yaw", "Head_pitch"],
            effort_limit_sim=7.0,
            velocity_limit_sim=7.0,
            stiffness=40.0,
            damping=2.0,
            armature=0.01,
        ),
    },
)

"""Configuration for the Booster T1 Humanoid robot."""