# Inspired by https://github.com/droid-dataset/droid
import abc
import numpy as np

from utils.transformations import add_poses, pose_diff
from env.robot.inverse_kinematics.robot_ik_solver import RobotIKSolver


class FrankaBase(abc.ABC):
    """
    Base class for Franka robot - holds the inverse kinematics solver
    Args:
        robot_type (str): type of robot
        control_hz (int): control frequency
        gripper (bool): whether the robot has a gripper
    """

    def __init__(
        self,
        robot_type="panda",
        control_hz=15,
        gripper=True,
    ):
        self._gripper = gripper

        self.robot_type = robot_type
        self.control_hz = control_hz

        self.launch_ik()

    def launch_ik(self):
        self._ik_solver = RobotIKSolver(
            robot_type=self.robot_type, control_hz=self.control_hz
        )

    def update_command(
        self, command, action_space="cartesian_velocity", blocking=False
    ):
        action_dict = self.create_action_dict(command, action_space=action_space)

        if self._gripper:
            self.update_gripper(
                action_dict["gripper_position"], velocity=False, blocking=blocking
            )

        self.update_joints(
            action_dict["joint_position"], velocity=False, blocking=blocking
        )

        return action_dict

    def create_action_dict(self, action, action_space, robot_state=None):
        assert action_space in [
            "cartesian_position",
            "joint_position",
            "joint_position_slow",
            "cartesian_velocity",
            "joint_velocity",
        ]
        if robot_state is None:
            robot_state = self.get_robot_state()[0]
        action_dict = {"robot_state": robot_state}
        velocity = "velocity" in action_space

        if velocity:
            action_dict["gripper_velocity"] = action[-1]
            gripper_delta = self._ik_solver.gripper_velocity_to_delta(action[-1])
            gripper_position = robot_state["gripper_position"] + gripper_delta
            action_dict["gripper_position"] = float(np.clip(gripper_position, 0, 1))
        else:
            action_dict["gripper_position"] = float(np.clip(action[-1], 0, 1))
            gripper_delta = (
                action_dict["gripper_position"] - robot_state["gripper_position"]
            )
            gripper_velocity = self._ik_solver.gripper_delta_to_velocity(gripper_delta)
            action_dict["gripper_delta"] = gripper_velocity

        if "cartesian" in action_space:
            if velocity:
                action_dict["cartesian_velocity"] = action[:-1]
                cartesian_delta = self._ik_solver.cartesian_velocity_to_delta(
                    action[:-1]
                )
                action_dict["cartesian_position"] = add_poses(
                    cartesian_delta, robot_state["cartesian_position"]
                ).tolist()
            else:
                action_dict["cartesian_position"] = action[:-1]
                cartesian_delta = pose_diff(
                    action[:-1], robot_state["cartesian_position"]
                )
                cartesian_velocity = self._ik_solver.cartesian_delta_to_velocity(
                    cartesian_delta
                )
                action_dict["cartesian_velocity"] = cartesian_velocity.tolist()

            action_dict["joint_velocity"] = (
                self._ik_solver.cartesian_velocity_to_joint_velocity(
                    action_dict["cartesian_velocity"], robot_state=robot_state
                ).tolist()
            )
            joint_delta = self._ik_solver.joint_velocity_to_delta(
                action_dict["joint_velocity"]
            )
            action_dict["joint_position"] = (
                joint_delta + np.array(robot_state["joint_positions"])
            ).tolist()

        if "joint" in action_space:
            # NOTE: Joint to Cartesian has undefined dynamics due to IK
            if velocity:
                action_dict["joint_velocity"] = action[:-1]
                joint_delta = self._ik_solver.joint_velocity_to_delta(action[:-1])
                action_dict["joint_position"] = (
                    joint_delta + np.array(robot_state["joint_positions"])
                ).tolist()
            else:
                action_dict["joint_position"] = action[:-1]
                joint_delta = np.array(action[:-1]) - np.array(
                    robot_state["joint_positions"]
                )
                joint_velocity = self._ik_solver.joint_delta_to_velocity(joint_delta)
                action_dict["joint_velocity"] = joint_velocity.tolist()

        return action_dict

    @abc.abstractmethod
    def get_ee_pose(self):
        """Get endeffector pose [pos (xyz), angle (euler)]"""

    @abc.abstractmethod
    def get_ee_pos(self):
        """Get endeffector position (xyz)"""

    @abc.abstractmethod
    def get_ee_angle(self):
        """Get endeffector angle (euler)"""

    @abc.abstractmethod
    def get_joint_positions(self):
        """Get robot joint positions"""

    @abc.abstractmethod
    def get_joint_velocities(self):
        """Get robot joint velocities"""

    @abc.abstractmethod
    def get_robot_state(self):
        """Get robot state"""

    @abc.abstractmethod
    def get_gripper_state(self):
        """Get gripper state"""

    @abc.abstractmethod
    def update_joints(self, qpos, velocity=False, blocking=False):
        """Update robot joint positions"""

    @abc.abstractmethod
    def update_gripper(self, gripper, velocity=False, blocking=False):
        """Update griper"""
