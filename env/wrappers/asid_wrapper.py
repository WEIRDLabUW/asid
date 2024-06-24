import copy

import gym
import mujoco
import numpy as np

from env.wrappers.obj_wrapper import ObjWrapper
from env.wrappers.asid_reward import ASIDRewardWrapper


class ASIDWrapper(gym.Wrapper):
    """
    ASID Wrapper
    Args:
        env (object): environment
        parameter_dict (dict): physics parameter dictionary
        obs_noise (float): observation noise std
        verbose (bool): verbose
    """

    def __init__(self, env, parameter_dict={}, obs_noise=0.0, verbose=False, **kwargs):
        super(ASIDWrapper, self).__init__(env)

        assert type(env) is ObjWrapper, "Environment must be wrapped in ObjWrapper!"
        self.verbose = verbose

        self.obs_noise = obs_noise

        # Mujoco object ids
        self.obj_geom_ids = mujoco.mj_name2id(
            self.env.unwrapped._robot.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            f"{self.obj_id}_geom",
        )
        # if single geom not found, check for multiple geoms
        if self.obj_geom_ids == -1:
            self.obj_geom_ids = []
            for i in range(5):
                obj_geom_id = mujoco.mj_name2id(
                    self.env.unwrapped._robot.model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    f"{self.obj_id}_geom_{i}",
                )
                if obj_geom_id == -1:
                    break
                self.obj_geom_ids.append(obj_geom_id)

        if -1 in self.obj_geom_ids:
            self.obj_geom_ids = mujoco.mj_name2id(
                self.env.unwrapped._robot.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                f"{self.obj_id}_geom",
            )

        self.obj_body_id = mujoco.mj_name2id(
            self.env.unwrapped._robot.model,
            mujoco.mjtObj.mjOBJ_BODY,
            f"{self.obj_id}_body",
        )

        self.surface_geom_id = mujoco.mj_name2id(
            self.env.unwrapped._robot.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "surface_geom",
        )

        # Physics parameters
        self.parameter_dict = parameter_dict

        self.reset_parameters()
        self.resample_parameters()

        # Exploration reward
        self.last_action = np.zeros(
            self.env.DoF if not self.env.gripper else self.env.DoF + 1
        )
        self.reward_first = True
        self.exp_reward = None

    def add_noise(self, obs):
        """
        Add observation noise
        Args:
            obs (np.ndarray): observation
        Returns:
            obs (np.ndarray): observation with noise
        """
        if self.obs_noise == 0.0:
            return obs

        if type(obs) is dict:
            for k in obs.keys():
                if "rgb" not in k and "depth" not in k:
                    obs[k] += np.random.normal(0, self.obs_noise, obs[k].shape)
        else:
            obs += np.random.normal(0, self.obs_noise, obs.shape)

        return obs

    def step(self, action):

        self.last_action = action

        if self.reward_first:
            asid_reward = self.compute_reward(action)

        obs, reward, done, info = self.env.step(action)

        if not self.reward_first:
            asid_reward = self.compute_reward(action)

        return self.add_noise(obs), reward + asid_reward, done, info

    def reset(self, *args, **kwargs):

        # randomize obj parameters | mujoco reset data
        self.resample_parameters()

        obs = self.env.reset()

        return self.add_noise(obs)

    def set_parameters(self, parameters):
        assert parameters.shape == self.get_parameters().shape
        if type(parameters) is dict:
            for k in parameters.key():
                self.parameter_dict[k]["value"] = parameters[k]
        else:
            for k, v in zip(self.parameter_dict.keys(), parameters):
                self.parameter_dict[k]["value"] = v
        self.params_set = True
        self.reset()

    def get_parameters(self):
        parameters = []
        for k in self.parameter_dict.keys():
            parameters.append(self.parameter_dict[k]["value"])
        return np.array(parameters)

    def get_parameters_distribution(self):
        return self.parameter_dict

    def set_parameters_distribution(self, parameter_dict):
        self.parameter_dict = parameter_dict

    def reset_parameters(self):
        self.params_set = False

    def resample_parameters(self):
        """
        Resample physics parameters
        Get physics parameters from parameter_dict and sample new values.
        Set new values in mujoco model.
        """
        for key in self.parameter_dict:
            # sample new parameter value

            # use set parameter + clip
            if self.params_set:
                value = self.parameter_dict[key]["value"]
                value = np.clip(
                    value,
                    self.parameter_dict[key]["min"],
                    self.parameter_dict[key]["max"],
                )
            # sample parameter from uniform
            elif self.parameter_dict[key]["type"] == "uniform":
                value = np.random.uniform(
                    low=self.parameter_dict[key]["min"],
                    high=self.parameter_dict[key]["max"],
                )
            # sample parameter from gaussian
            elif self.parameter_dict[key]["type"] == "gaussian":
                value = np.random.normal(
                    loc=self.parameter_dict[key]["mean"],
                    scale=self.parameter_dict[key]["std"],
                )
            self.parameter_dict[key]["value"] = value

            # set new parameter value
            if key == "inertia":
                com_body_id = mujoco.mj_name2id(
                    self.env.unwrapped._robot.model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    f"{self.obj_id}_com",
                )
                self.env.unwrapped._robot.model.body_pos[com_body_id][1] = value

            elif key == "friction":
                for geom_id in self.obj_geom_ids:
                    self.env.unwrapped._robot.model.geom_friction[geom_id] = value
                    # self.env.unwrapped._robot.model.geom_friction[self.surface_geom_id][:2] = value
                    # self.env.unwrapped._robot.model.geom_friction[self.surface_geom_id][2] = 0.0
            else:
                print(
                    f"WARNING: {key} not supported! Choose one of [inertia, friction]."
                )

            # elif key == "surface_friction":
            #     self.env.unwrapped._robot.model.geom_friction[self.surface_geom_id][
            #         :2
            #     ] = value
            #     self.env.unwrapped._robot.model.geom_friction[self.surface_geom_id][
            #         2
            #     ] = 0.0
            # elif key == "mass":
            #     self.env.unwrapped._robot.model.body_mass[self.obj_body_id] = value
            # elif key == "damping":
            #     joint_id = mujoco.mj_name2id(
            #         self.env.unwrapped._robot.model,
            #         mujoco.mjtObj.mjOBJ_JOINT,
            #         f"{self.obj_id}_freejoint",
            #     )
            #     self.env.unwrapped._robot.model.dof_damping[joint_id] = value
            # elif key == "frictionloss":
            #     joint_id = mujoco.mj_name2id(
            #         self.env.unwrapped._robot.model,
            #         mujoco.mjtObj.mjOBJ_JOINT,
            #         f"{self.obj_id}_freejoint",
            #     )
            #     self.env.unwrapped._robot.model.dof_frictionloss[joint_id] = value

        if self.verbose:
            print(
                f"Parameters: {self.get_parameters()} - seed {self.env.unwrapped._seed}"
            )

        # update sim
        mujoco.mj_resetData(
            self.env.unwrapped._robot.model, self.env.unwrapped._robot.data
        )

    def set_data(self, new_data):
        self.env.unwrapped._robot.data = new_data
        mujoco.mj_forward(
            self.env.unwrapped._robot.model, self.env.unwrapped._robot.data
        )

    def get_data(self):
        return copy.deepcopy(self.env.unwrapped._robot.data)

    def get_full_state(self):
        full_state = {}
        full_state["last_action"] = self.last_action.copy()
        full_state["curr_path_length"] = copy.copy(self.env.curr_path_length)
        full_state["robot_data"] = self.get_data()
        return full_state

    def set_full_state(self, full_state):
        self.last_action = full_state["last_action"]
        self.env.curr_path_length = full_state["curr_path_length"]
        self.set_data(full_state["robot_data"])

    def create_exp_reward(
        self,
        env_func,
        robot_cfg_dict,
        env_cfg_dict,
        asid_cfg_dict,
        seed=0,
        device_id=0,
    ):
        """
        Create exploration reward env.
        Initializes a copy of the env used to compute gradients with finite differences. Turn off rendering, randomization.
        Args:
            env_func (object): environment function
            robot_cfg_dict (dict): robot configuration dictionary
            env_cfg_dict (dict): environment configuration dictionary
            asid_cfg_dict (dict): ASID configuration dictionary
            seed (int): seed
            device_id (int): device id
        """

        # no rendering
        robot_cfg_dict["on_screen_rendering"] = False
        # don't randomize obj pose
        env_cfg_dict["obj_pose_noise_dict"] = None
        # don't compute ASID reward -> make sure it doesn't run into infinite loop
        asid_cfg_dict["reward"] = False
        # no obs noise
        asid_cfg_dict["obs_noise"] = 0.0

        exp_env = env_func(
            robot_cfg_dict,
            env_cfg_dict,
            asid_cfg_dict=asid_cfg_dict,
            seed=seed,
            device_id=device_id,
            verbose=False,
        )

        # create wrapper that holds envs for gradient computation
        self.exp_reward = ASIDRewardWrapper(
            exp_env,
            delta=asid_cfg_dict["delta"],
            normalization=asid_cfg_dict["normalization"],
        )

    def compute_reward(self, action):
        """
        Compute exploration reward for current simulator state, parameters, and action taken.
        Args:
            action (np.ndarray): action
        Returns:
            reward (float): reward
        """

        if self.exp_reward:
            full_state = self.get_full_state()
            current_param = self.get_parameters()
            return self.exp_reward.get_reward(
                full_state,
                action,
                params=current_param,
                verbose=False,
            )
        else:
            return 0.0
