import copy

import gym
import mujoco
import numpy as np

from utils.transformations_mujoco import euler_to_quat_mujoco


class ObjWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        obj_id="rod",
        obj_pose_init=None,
        obj_pose_noise_dict=None,
        obs_keys=None,
        safety_penalty=0.,
        flatten=True,
        verbose=False,
    ):
        super(ObjWrapper, self).__init__(env)
        
        self.verbose = verbose
        self.obj_id = obj_id

        # intialize robot to the bottom left
        if self.obj_id == "rod" and self.env.DoF == 2:
            self.env._reset_joint_qpos = np.array(
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

        # Mujoco object ids
        self.obj_body_id = mujoco.mj_name2id(
            self.env._robot.model, mujoco.mjtObj.mjOBJ_BODY, f"{self.obj_id}_body"
        )

        self.obj_joint_id = mujoco.mj_name2id(
            self.env._robot.model, mujoco.mjtObj.mjOBJ_JOINT, f"{self.obj_id}_freejoint"
        )
        self.obj_joint_id = self.env._robot.model.jnt_qposadr[self.obj_joint_id]


        self.obj_geom_id = mujoco.mj_name2id(
            self.env._robot.model, mujoco.mjtObj.mjOBJ_GEOM, f"{self.obj_id}_geom"
        )

        # Object position
        self.obj_pose_noise_dict = obj_pose_noise_dict
        self.obj_pos_noise = obj_pose_noise_dict is not None
        self.init_obj_pose = self.get_obj_pose() if obj_pose_init is None else obj_pose_init
        self.curr_obj_pose = None

        # Reward
        self.safety_penalty = safety_penalty

        # Observations
        self.obs_keys = (
            obs_keys if obs_keys is not None else self.env.observation_space.keys()
        )
        # obs space dict to array
        for k in copy.deepcopy(self.env.observation_space.keys()):
            if k not in self.obs_keys:
                del self.env.observation_space.spaces[k]

        self.flatten = flatten
        obj_pose_low = -np.inf * np.ones(7)
        obj_pose_high = np.inf * np.ones(7)
        if self.flatten:
            low = np.concatenate([v.low for v in self.env.observation_space.values()])
            high = np.concatenate([v.high for v in self.env.observation_space.values()])
            # add obj pose
            low = np.concatenate([low, obj_pose_low])
            high = np.concatenate([high, obj_pose_high])
            # overwrite observation_space
            self.observation_space = gym.spaces.Box(low=low, high=high, shape=low.shape)
        else:
            self.observation_space["obj_pose"] = gym.spaces.Box(
                low=obj_pose_low, high=obj_pose_high
            )
        
        mujoco.mj_resetData(
            self.env._robot.model, self.env._robot.data
        )

    def augment_observations(self, obs, flatten=True):
        obs["obj_pose"] = self.get_obj_pose()

        if flatten:
            tmp = []
            for k in self.obs_keys:
                tmp.append(obs[k])
            obs = np.concatenate(tmp)

        return obs

    def obj_on_table(self):
        return self.get_obj_pose()[2] > 0.

    def step(self, action):

        obs, reward, done, info = self.env.step(action)

        reward -= self.safety_penalty * int(not self.obj_on_table())

        return self.augment_observations(obs, flatten=self.flatten), reward, done, info

    def reset(self, *args, **kwargs):
        
        # randomize obj position |
        self.resample_obj_pose()

        if self.curr_obj_pose is None:
            obj_pose = self.init_obj_pose.copy()
        else:
            obj_pose = self.curr_obj_pose.copy()
        # set obj qpos | mujoco forward
        self.update_obj(obj_pose)

        # reset robot |
        obs = self.env.reset()
        
        return self.augment_observations(obs, flatten=self.flatten)

    def get_obj_pose(self):
        return self.env._robot.data.qpos[self.obj_joint_id:self.obj_joint_id+7]
        
    def set_obj_pose(self, obj_pose):
        self.obj_pos_noise = True
        self.init_obj_pose = obj_pose.copy()
        self.update_obj(obj_pose)

    def resample_obj_pose(self):
        pose = self.init_obj_pose.copy()
       
        if self.obj_pos_noise:
            pose[0] += np.random.uniform(
                self.obj_pose_noise_dict["x"]["min"],
                self.obj_pose_noise_dict["x"]["max"],
            )
            pose[1] += np.random.uniform(
                self.obj_pose_noise_dict["y"]["min"],
                self.obj_pose_noise_dict["y"]["max"],
            )
            pose[3:7] = euler_to_quat_mujoco(
                [
                    0.0,
                    0.0,
                    np.random.uniform(
                        self.obj_pose_noise_dict["yaw"]["min"],
                        self.obj_pose_noise_dict["yaw"]["max"],
                        size=1,
                    ).item(),
                ]
            )
            

        if self.verbose:
            print(f"Object pose: {pose} - seed {self.env._seed}")
        self.curr_obj_pose = pose.copy()

    def update_obj(self, qpos):
        self.env._robot.data.qpos[self.obj_joint_id:self.obj_joint_id+7] = qpos
        mujoco.mj_step(self.env._robot.model,self.env._robot.data,nstep=self.env.unwrapped._robot.frame_skip)
        