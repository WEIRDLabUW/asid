# Inspired by https://github.com/droid-dataset/droid

import numpy as np

import gym
from gym.spaces import Box, Dict

from utils.transformations import add_angles, angle_diff
from env.robot.franka_mujoco import MujocoManipulatorEnv


class RobotEnv(gym.Env):
    """
    Robot environment for controlling a Franka robot in Mujoco
    Args:
        control_hz (int): control frequency
        DoF (int): degrees of freedom of the end effector
        gripper (bool): whether the robot has a gripper
        qpos (bool): whether to include joint positions in observation space
        ee_pos (bool): whether to include end effector position in observation space
        imgs (bool): whether to include images in observation space
        max_path_length (int): maximum path length before resetting
        camera_names (list): list of camera names to use in sim
        camera_rgb (bool): whether to render rgb cameras
        camera_depth (bool): whether to render depth cameras
        model_name (str): name of mujoco model
        on_screen_rendering (bool): whether to render on screen
        device_id (int): gpu device id
    """

    def __init__(
        self,
        # control frequency
        control_hz=10,
        DoF=3,
        gripper=True,
        # observation space configuration
        qpos=True,
        ee_pos=True,
        imgs=True,
        # specify path length if resetting after a fixed length
        max_path_length=None,
        # cameras to use in sim
        camera_names=["front"],
        camera_rgb=True,
        camera_depth=False,
        # mujoco
        model_name="base_franka",
        on_screen_rendering=False,
        # gpu
        device_id=None,
    ):
        # initialize gym environment
        super().__init__()

        # physics
        self.DoF = DoF
        self.gripper = gripper
        self.control_hz = control_hz

        self._episode_count = 0
        self._max_path_length = max_path_length
        self.curr_path_length = 0

        self._reset_joint_qpos = np.array(
            [
                -5.65335140e-05,
                -1.47445112e-01,
                5.44415554e-03,
                -2.57991934e00,
                2.13176832e-02,
                2.43316126e00,
                7.82760382e-01,
            ]
        )

        if self.DoF == 2:
            self._reset_joint_qpos = np.array(
                [
                    0.85290707,
                    0.29776727,
                    0.0438237,
                    -2.70994978,
                    -0.00481878,
                    2.89241547,
                    1.67766532,
                ]
            )

        # observation space config
        self._qpos = qpos
        self._ee_pos = ee_pos
        self._imgs = imgs

        # action space
        action_low, action_high = -0.1, 0.1
        self.action_space = Box(
            np.array(
                [action_low] * (self.DoF + 1 if self.gripper else self.DoF),
                dtype=np.float32,
            ),  # dx_low, dy_low, dz_low, dgripper_low
            np.array(
                [action_high] * (self.DoF + 1 if self.gripper else self.DoF),
                dtype=np.float32,
            ),  # dx_high, dy_high, dz_high, dgripper_high
        )
        self.action_shape = self.action_space.shape

        # EE position (x, y, z) + EE rot (roll, pitch, yaw) + gripper width
        ee_space_low = np.array([0.12, -1.0, 0.11, -np.pi, -np.pi, -np.pi, 0.00])
        ee_space_high = np.array([1.0, 1.0, 0.7, np.pi, np.pi, np.pi, 0.085])

        # EE position (x, y, fixed z)
        if self.DoF == 2:
            ee_space_low = ee_space_low[:3]
            ee_space_high = ee_space_high[:3]
        # EE position (x, y, z)
        if self.DoF == 3:
            ee_space_low = ee_space_low[:3]
            ee_space_high = ee_space_high[:3]
        # EE position (x, y, z) + EE rot (single axis)
        elif self.DoF == 4:
            ee_space_low = np.concatenate((ee_space_low[:3], ee_space_low[5:6]))
            ee_space_high = np.concatenate((ee_space_high[:3], ee_space_high[5:6]))
        # EE position (x, y, z) + EE rot
        elif self.DoF == 6:
            ee_space_low = ee_space_low[:6]
            ee_space_high = ee_space_high[:6]

        # gripper width
        if self.gripper:
            ee_space_low = np.concatenate((ee_space_low, ee_space_low[-1:]))
            ee_space_high = np.concatenate((ee_space_high, ee_space_high[-1:]))

        self.ee_space = Box(
            low=np.float32(ee_space_low), high=np.float32(ee_space_high)
        )

        # joint limits + gripper
        # https://frankaemika.github.io/docs/control_parameters.html
        self._jointmin = np.array(
            [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, 0.0045],
            dtype=np.float32,
        )
        self._jointmax = np.array(
            [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 0.085],
            dtype=np.float32,
        )

        self._robot = MujocoManipulatorEnv(
            model_name=model_name,
            control_hz=self.control_hz,
            has_renderer=on_screen_rendering,
            has_offscreen_renderer=not on_screen_rendering,
            use_rgb=camera_rgb,
            use_depth=camera_depth,
            camera_names=camera_names,
        )

        # joint space + gripper
        self.qpos_space = Box(self._jointmin, self._jointmax)

        # final observation space configuration
        env_obs_spaces = {}

        if self._qpos:
            env_obs_spaces["lowdim_ee"] = self.ee_space
        if self._ee_pos:
            env_obs_spaces["lowdim_qpos"] = self.qpos_space

        if self._imgs:
            imgs = self.get_images()
            if len(imgs) > 0:
                for sn, img in imgs.items():
                    for m, modality in img.items():
                        if m == "rgb":
                            env_obs_spaces[f"{sn}_{m}"] = Box(
                                0, 255, modality.shape, np.uint8
                            )
                        elif m == "depth":
                            env_obs_spaces[f"{sn}_{m}"] = Box(
                                0, 65535, modality.shape, np.uint16
                            )
                        elif m == "points":
                            pass

        self.observation_space = Dict(env_obs_spaces)

        self.observation_shape = {}
        self.observation_type = {}
        for k in env_obs_spaces.keys():
            self.observation_shape[k] = env_obs_spaces[k].shape
            self.observation_type[k] = env_obs_spaces[k].dtype

        self._seed = 0

    def get_spaces(self):
        return self.observation_space, self.action_space

    def step(self, action):
        if not self.gripper:
            assert len(action) == (
                self.DoF
            ), f"Expected action shape: ({self.DoF},) got {action.shape}"
        else:
            assert len(action) == (
                self.DoF + 1
            ), f"Expected action shape: ({self.DoF+1},) got {action.shape}"

        # clip action to action space
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # formate action to DoF
        pos_action, angle_action, gripper = self._format_action(action)

        # clipping + any safety corrections for position
        desired_pos = self._get_valid_pos(self._curr_pos + pos_action)
        desired_angle = add_angles(angle_action, self._curr_angle)

        # cartesian position control
        self._update_robot(
            np.concatenate((desired_pos, desired_angle, [gripper])),
            action_space="cartesian_position",
            blocking=False,
        )

        # get observations
        obs = self.get_observation()

        self.curr_path_length += 1
        done = False
        if (
            self._max_path_length is not None
            and self.curr_path_length >= self._max_path_length
        ):
            done = True

        return obs, 0.0, done, {}

    def reset_gripper(self):
        self._robot.update_gripper(0.0, velocity=False, blocking=True)

    def reset(self):

        # ensure robot releases grasp before reset
        if self.gripper:
            self.reset_gripper()
        else:
            # default is closed gripper if not self.gripper
            self._robot.update_gripper(1.0, velocity=False, blocking=True)
        # reset to home pose
        for _ in range(3):
            self._robot.update_joints(
                self._reset_joint_qpos.tolist(), velocity=False, blocking=True
            )

            epsilon = 0.1
            is_reset, joint_dist = self.is_robot_reset(epsilon=epsilon)

            if is_reset:
                break

        # fix default pos and angle at first joint reset
        if self._episode_count == 0:
            self._default_pos = self._robot.get_ee_pos()
            self._default_angle = self._robot.get_ee_angle()

            # overwrite fixed z for 2DoF EE control with reset z
            if self.DoF == 2:
                self.ee_space.low[2] = self._default_pos[2]
                self.ee_space.high[2] = self._default_pos[2]

                # overwrite fixed z for 2DoF EE control with 0.14
                self.ee_space.low[2] = 0.14
                self.ee_space.high[2] = 0.14

        self.curr_path_length = 0
        self._episode_count += 1

        return self.get_observation()

    def _format_action(self, action):
        """Returns [x,y,z], [yaw, pitch, roll], close_gripper"""
        default_delta_angle = angle_diff(self._default_angle, self._curr_angle)

        if self.DoF == 2:
            delta_pos, delta_angle = (
                np.concatenate(
                    (action[:2], self._default_pos[2:]),
                ),
                default_delta_angle,
            )
        if self.DoF == 3:
            delta_pos, delta_angle = (
                action[:3],
                default_delta_angle,
            )
        elif self.DoF == 4:
            delta_pos, delta_angle = (
                action[:3],
                [default_delta_angle[0], default_delta_angle[1], action[3]],
            )
        elif self.DoF == 6:
            delta_pos, delta_angle = action[:3], action[3:6]

        if self.gripper:
            gripper = action[-1]
        else:
            # default is closed gripper if not self.gripper
            gripper = 1.0
        return np.array(delta_pos), np.array(delta_angle), gripper

    def _get_valid_pos(self, pos):

        # clip commanded position to satisfy box constraints
        x_low, y_low, z_low = self.ee_space.low[:3]
        x_high, y_high, z_high = self.ee_space.high[:3]
        pos[0] = pos[0].clip(x_low, x_high)  # new x
        pos[1] = pos[1].clip(y_low, y_high)  # new y
        pos[2] = pos[2].clip(z_low, z_high)  # new z

        return pos

    def _update_robot(self, action, action_space, blocking=False):
        assert action_space in [
            "cartesian_position",
            "joint_position",
            "cartesian_velocity",
            "joint_velocity",
        ]
        action_info = self._robot.update_command(
            action, action_space=action_space, blocking=blocking
        )
        return action_info

    @property
    def _curr_pos(self):
        return self._robot.get_ee_pos()

    @property
    def _curr_angle(self):
        return self._robot.get_ee_angle()

    @property
    def _num_cameras(self):
        return len(self._robot.camera_names)

    def render(self, mode=None, sn=None):
        if self._robot.has_renderer:
            self._robot.render()
        else:
            imgs = self.get_images()
            if sn is None:
                sn = next(iter(imgs))
            return imgs[sn]["rgb"]

    def get_images(self):
        imgs = []
        if not self._robot.has_renderer:
            imgs = self._robot.render()
        else:
            imgs = self._camera_reader.read_cameras()

        img_dict = {}
        for img in imgs:
            sn = img["serial_number"].split("/")[0]

            if img_dict.get(sn) is None:
                img_dict[sn] = {}

            if img["type"] == "depth":
                img_dict[sn]["depth"] = img["array"]
            elif img["type"] == "rgb":
                img_dict[sn]["rgb"] = img["array"]

        return img_dict

    def get_state(self):
        state_dict = {}
        if self.gripper:
            gripper_state = self._robot.get_gripper_state()

        state_dict["control_key"] = "current_pose"

        state_dict["current_pose"] = (
            np.concatenate(
                [self._robot.get_ee_pos(), self._robot.get_ee_angle(), [gripper_state]]
            )
            if self.gripper
            else np.concatenate([self._robot.get_ee_pos(), self._robot.get_ee_angle()])
        )

        state_dict["joint_positions"] = self._robot.get_joint_positions()
        state_dict["joint_velocities"] = self._robot.get_joint_velocities()
        # don't track gripper velocity
        state_dict["gripper_velocity"] = 0

        return state_dict

    def get_observation(self):
        # get state and images
        current_state = self.get_state()

        # set gripper width
        gripper_width = current_state["current_pose"][-1:]

        if self.DoF == 3 or self.DoF == 2:
            ee_pos = (
                np.concatenate([current_state["current_pose"][:3], gripper_width])
                if self.gripper
                else current_state["current_pose"][:3]
            )
        elif self.DoF == 4:
            ee_pos = (
                np.concatenate(
                    [
                        current_state["current_pose"][:3],
                        current_state["current_pose"][5:6],
                        gripper_width,
                    ]
                )
                if self.gripper
                else np.concatenate(
                    [
                        current_state["current_pose"][:3],
                        current_state["current_pose"][5:6],
                    ]
                )
            )
        elif self.DoF == 6:
            ee_pos = (
                np.concatenate(
                    [
                        current_state["current_pose"][:6],
                        gripper_width,
                    ]
                )
                if self.gripper
                else current_state["current_pose"][:6]
            )

        qpos = np.concatenate([current_state["joint_positions"], gripper_width])

        obs_dict = {
            "lowdim_qpos": qpos,
            "lowdim_ee": ee_pos,
        }

        if self._imgs:
            current_images = self.get_images()
            if len(current_images) > 0:
                for sn, img in current_images.items():
                    for m, modality in img.items():
                        obs_dict[f"{sn}_{m}"] = modality

        return obs_dict

    def is_robot_reset(self, epsilon=0.1):
        curr_joints = self._robot.get_joint_positions()
        joint_dist = np.linalg.norm(curr_joints - self._reset_joint_qpos)
        return joint_dist < epsilon, joint_dist

    def seed(self, seed):
        self._seed = seed
        np.random.seed(seed)
