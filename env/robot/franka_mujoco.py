# Inspired by https://github.com/facebookresearch/fairo/blob/main/polymetis/polymetis/python/polysim/envs/mujoco_manipulator.py

import os

import mujoco
import mujoco_viewer
import numpy as np
import torch

from utils.transformations import euler_to_quat, rmat_to_euler
from env.robot.franka_base import FrankaBase


class MujocoManipulatorEnv(FrankaBase):
    """
    Mujoco Manipulator Environment
    Args:
        control_hz (int): control frequency
        gripper (bool): whether the robot has a gripper
        model_name (str): name of the mujoco model file
        has_renderer (bool): whether to render the environment onscreen
        has_offscreen_renderer (bool): whether to render the environment offscreen
        camera_names (list): list of camera names (cameras should be defined in the mujoco model)
        use_rgb (bool): whether to render rgb images
        use_depth (bool): whether to render depth images
        img_height (int): height of the rendered images
        img_width (int): width of the rendered images
    """

    def __init__(
        self,
        # robot
        control_hz=15,
        gripper=True,
        model_name="base_franka",
        # rendering
        has_renderer=False,
        has_offscreen_renderer=True,
        camera_names=["front", "left"],
        use_rgb=True,
        use_depth=False,
        img_height=480,
        img_width=640,
    ):
        super().__init__(
            control_hz=control_hz,
            gripper=gripper,
        )

        self.robot_desc_mjcf_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), f"assets/{model_name}.xml"
        )

        # mujoco setup
        self.model = mujoco.MjModel.from_xml_path(self.robot_desc_mjcf_path)
        self.data = mujoco.MjData(self.model)

        self.model.opt.gravity = np.array([0, 0, -9.81])
        self._max_gripper_width = 0.08

        self.frame_skip = int((1 / self.control_hz) / self.model.opt.timestep)

        # get mujoco ids
        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "EEF")
        self.franka_joint_ids = []
        for i in range(7):
            self.franka_joint_ids += [
                mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, f"panda_joint{i+1}"
                )
            ]
        self.franka_finger_joint_ids = [
            mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, "robot:finger_joint1"
            ),
            mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, "robot:finger_joint2"
            ),
        ]

        # rendering
        assert (
            has_renderer and has_offscreen_renderer
        ) is False, "both has_renderer and has_offscreen_renderer not supported"
        self.has_renderer = has_renderer
        self.has_offscreen_renderer = has_offscreen_renderer
        self.use_rgb = use_rgb
        self.use_depth = use_depth
        self.img_width = img_width
        self.img_height = img_height
        self.camera_names = camera_names
        self.viewer = None

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def update_joints(self, command, velocity=False, blocking=False):
        """
        General update joints function
        Args:
            command (np.array): joint positions or joint velocities
            velocity (bool): whether the command is joint velocities
            blocking (bool): blocking control - wait for the robot to reach command
        """

        if velocity:
            joint_delta = self._ik_solver.joint_velocity_to_delta(command)
            command = joint_delta + self.get_joint_positions()

        if blocking:
            time_to_go = self.adaptive_time_to_go(command)
            self.move_to_joint_positions(command, time_to_go=time_to_go)
        else:
            self.update_desired_joint_positions(command)

    def update_desired_joint_positions(self, joint_pos_desired=None):
        """
        Update joint positions
        Args:
            joint_pos_desired (np.array): desired joint positions
        """

        self.data.ctrl[: len(self.franka_joint_ids)] = joint_pos_desired
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)

    def move_to_joint_positions(self, joint_pos_desired=None, time_to_go=3):
        """
        Update joint positions with time_to_go
        Args:
            joint_pos_desired (np.array): desired joint positions
            time_to_go (float): time to reach the desired joint positions
        """
        self.data.ctrl[: len(self.franka_joint_ids)] = joint_pos_desired
        # use position control -> skip sim for time_to_go
        mujoco.mj_step(
            self.model, self.data, nstep=int(time_to_go // self.model.opt.timestep)
        )

    def update_gripper(self, command, velocity=False, blocking=False):
        """
        Update gripper position
        Args:
            command (float): gripper position
            velocity (bool): whether the command is gripper velocity
            blocking (bool): blocking control - wait for the robot to reach command
        """
        # 1. -> close, 0. -> open
        if velocity:
            gripper_delta = self._ik_solver.gripper_velocity_to_delta(command)
            command = gripper_delta + self.get_gripper_position()

        command = float(np.clip(command, 0, 1))
        if command > 0.0:
            self.data.ctrl[len(self.franka_joint_ids) :] = 0.0
        else:
            self.data.ctrl[len(self.franka_joint_ids) :] = 255.0

        if blocking:
            mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)

    def set_robot_state(self, robot_state):
        """
        Set robot state
        Args:
            robot_state (dict): robot state
        """
        self.data.qpos = robot_state.joint_positions
        self.data.qvel = robot_state.joint_velocities
        self.data.ctrl = self.data.qfrc_bias
        mujoco.mj_step(self.model, self.data)
        if self.gui:
            self.render()

    def get_ee_pose(self):
        """
        Get end-effector pose
        Returns:
            np.array: end-effector pose
        """
        return np.concatenate((self.get_ee_pos(), self.get_ee_angle()))

    def get_ee_pos(self):
        """
        Get end-effector position
        Returns:
            np.array: end-effector position
        """
        ee_pos = self.data.site_xpos[self.ee_site_id].copy()
        return ee_pos

    def get_ee_angle(self, quat=False):
        """
        Get end-effector angle
        Args:
            quat (bool): whether to return quaternion
        Returns:
            np.array: end-effector angle euler or quaternion
        """
        ee_mat = self.data.site_xmat[self.ee_site_id].copy().reshape(3, 3)
        ee_angle = rmat_to_euler(ee_mat)
        if quat:
            return euler_to_quat(ee_angle)
        else:
            return ee_angle

    def get_joint_positions(self):
        """
        Get joint positions
        Returns:
            np.array: joint positions
        """
        qpos = self.data.qpos[self.franka_joint_ids].copy()
        return qpos

    def get_joint_velocities(self):
        qvel = self.data.qvel[self.franka_joint_ids].copy()
        return qvel

    def get_robot_state(self):
        """
        Get robot state
        Returns:
            dict: robot state
            dict: additional info
        """

        joint_positions = self.get_joint_positions()
        joint_velocities = self.get_joint_velocities()

        gripper_position = self.get_gripper_state()
        pos, quat = self.get_ee_pos(), self.get_ee_angle(quat=False)

        state_dict = {
            "cartesian_position": np.concatenate([pos, quat]),
            "gripper_position": gripper_position,
            "joint_positions": joint_positions,
            "joint_velocities": joint_velocities,
        }

        return state_dict, {}

    def get_gripper_state(self):
        """
        Get gripper state
        Returns:
            float: gripper state
        """
        if self._gripper:
            return (
                self.data.qpos[self.franka_finger_joint_ids[0]]
                + self.data.qpos[self.franka_finger_joint_ids[1]]
            )
        else:
            return 0.0

    def get_gripper_position(self):
        """
        Get gripper position
        Returns:
            float: gripper position
        """
        if self._gripper:
            return 1 - (self.get_gripper_state() / self._max_gripper_width)
        else:
            return 0.0

    def _adaptive_time_to_go_polymetis(
        self, joint_displacement: torch.Tensor, time_to_go_default=1.0
    ):
        """
        Compute adaptive time_to_go (polymetis)

        Computes the corresponding time_to_go such that the mean velocity is equal to one-eighth
        of the joint velocity limit:
        time_to_go = max_i(joint_displacement[i] / (joint_velocity_limit[i] / 8))
        (Note 1: The magic number 8 is deemed reasonable from hardware tests on a Franka Emika.)
        (Note 2: In a min-jerk trajectory, maximum velocity is equal to 1.875 * mean velocity.)

        The resulting time_to_go is also clipped to a minimum value of the default time_to_go.

        Args:
            joint_displacement (torch.Tensor): joint displacement
            time_to_go_default (float): default time to go
        Returns:
            float: time to go
        """

        # TODO verify those limits
        # https://frankaemika.github.io/docs/control_parameters.html
        joint_vel_limits = torch.tensor(
            [2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26]
        ).float()  # robot_model.get_joint_velocity_limits()
        joint_pos_diff = torch.abs(torch.tensor(joint_displacement)).float()
        time_to_go = torch.max(joint_pos_diff / joint_vel_limits * 8.0)
        return max(time_to_go, time_to_go_default)

    def adaptive_time_to_go(self, desired_joint_position, t_min=0, t_max=4):
        """
        Compute adaptive time_to_go
        Args:
            desired_joint_position (np.array): desired joint positions
            t_min (float): minimum time to go
            t_max (float): maximum time to go
        Returns:
            float: time to go
        """
        curr_joint_position = self.get_joint_positions()
        displacement = desired_joint_position - curr_joint_position
        time_to_go = self._adaptive_time_to_go_polymetis(displacement)
        clamped_time_to_go = min(t_max, max(time_to_go, t_min))
        return clamped_time_to_go

    def viewer_setup(self):
        """
        Setup viewer
        Returns:
            mujoco_viewer.MujocoViewer or mujoco.Renderer: viewer
        """
        if self.has_renderer:
            return mujoco_viewer.MujocoViewer(
                self.model,
                self.data,
                height=self.img_height,
                width=self.img_width,
                hide_menus=True,
            )
        if self.has_offscreen_renderer:
            return mujoco.Renderer(
                self.model, height=self.img_height, width=self.img_width
            )

    def render(self):
        """
        Rendering.
        If has_renderer, render the environment onscreen. If has_offscreen_renderer, render the environment offscreen.
        Returns:
            list: list of images if has_offscreen_renderer or empty list if has_renderer
        """
        imgs = []

        if not self.viewer:
            self.viewer = self.viewer_setup()

        if self.has_renderer:
            self.viewer.render()
        elif self.has_offscreen_renderer:
            for camera in self.camera_names:
                self.viewer.update_scene(self.data, camera=camera)

                color_image = None
                if self.use_rgb:
                    color_image = self.viewer.render().copy()
                    dict_1 = {
                        "serial_number": camera,
                        "array": color_image,
                        "shape": color_image.shape if color_image is not None else None,
                        "type": "rgb",
                    }
                    imgs.append(dict_1)

                depth = None
                if self.use_depth:
                    self.viewer.enable_depth_rendering()
                    depth = self.viewer.render().copy()
                    self.viewer.disable_depth_rendering()
                    dict_2 = {
                        "serial_number": camera,
                        "array": depth,
                        "shape": depth.shape if depth is not None else None,
                        "type": "depth",
                    }
                    imgs.append(dict_2)

        return imgs
