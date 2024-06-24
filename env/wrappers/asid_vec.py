from functools import partial

from env.robot.robot_env import RobotEnv
from env.wrappers.obj_wrapper import ObjWrapper
from env.wrappers.vec_wrapper import SubVecEnv
from env.wrappers.asid_wrapper import ASIDWrapper

def make_env(
    robot_cfg_dict,
    env_cfg_dict=None,
    asid_cfg_dict=None,
    seed=0,
    device_id=0,
    verbose=False,
):

    if verbose:
        print(robot_cfg_dict)
        print(env_cfg_dict)
        print(asid_cfg_dict)

    env = RobotEnv(**robot_cfg_dict, device_id=device_id)

    env = ObjWrapper(env, **env_cfg_dict, verbose=verbose)
    
    env = ASIDWrapper(env, **asid_cfg_dict, verbose=verbose)

    if asid_cfg_dict["reward"]:
        env.create_exp_reward(
            make_env,
            robot_cfg_dict,
            env_cfg_dict,
            asid_cfg_dict,
            seed=seed,
            device_id=device_id,
        )

    env.seed(seed)

    return env


def make_vec_env(
    robot_cfg_dict,
    env_cfg_dict=None,
    asid_cfg_dict=None,
    num_workers=1,
    seed=0,
    device_id=0,
    verbose=False,
):

    env_fns = [
        partial(
            make_env,
            robot_cfg_dict,
            env_cfg_dict,
            asid_cfg_dict,
            seed=seed + i,
            device_id=device_id,
            verbose=bool(i == 0) and verbose,
        )
        for i in range(num_workers)
    ]
    return SubVecEnv(env_fns)
