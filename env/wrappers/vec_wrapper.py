# https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/vec_env/subproc_vec_env.py
import multiprocessing as mp

import cloudpickle
import numpy as np

from env.wrappers.base_vec_env import CloudpickleWrapper
from env.wrappers.patch_gym import _patch_env


def _worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = _patch_env(env_fn_wrapper.var())
    # env = env_fn_wrapper.var()
    rewards, successes = [], []
    while True:
        try:
            cmd, data = remote.recv()
            # gym interface
            if cmd == "step":
                obs, reward, done, _, info = env.step(data)
                rewards.append(reward)
                successes.append(info.get("success", 0))
                if done:
                    info["terminal_obs"] = obs
                    info["episode_return"] = sum(rewards)
                    info["episode_success"] = float(sum(successes) > 0)
                    rewards, successes = [], []
                    obs, _ = env.reset()
                remote.send((obs, reward, done, info))
            elif cmd == "seed":
                remote.send(env.seed(data))
            elif cmd == "reset":
                obs, _ = env.reset()
                remote.send(obs)
                rewards, successes = [], []
            elif cmd == "render":
                remote.send(env.render())
            elif cmd == "close":
                env.close()
                remote.close()
                break

            # misc
            elif cmd == "get_spaces":
                remote.send((env.observation_space, env.action_space))
            elif cmd == "env_method":
                method = getattr(env, data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == "get_attr":
                remote.send(getattr(env, data))
            elif cmd == "set_attr":
                remote.send(setattr(env, data[0], data[1]))

            # physics parameters
            elif cmd == "get_parameters":
                remote.send(env.get_parameters())
            elif cmd == "get_parameters_distribution":
                remote.send(env.get_parameters_distribution())
            elif cmd == "set_parameters":
                remote.send(env.set_parameters(data))
            elif cmd == "set_parameters_distribution":
                remote.send(env.set_parameters_distribution(data))

            # pose
            elif cmd == "get_obj_pose":
                remote.send(env.get_obj_pose())
            elif cmd == "set_obj_pose":
                remote.send(env.set_obj_pose(data))

            # asid reward
            elif cmd == "get_full_state":
                remote.send(env.get_full_state())
            elif cmd == "set_full_state":
                remote.send(env.set_full_state(data))

            # observations
            elif cmd == "get_observation":
                remote.send(env.get_observation())
            elif cmd == "get_images":
                remote.send(env.get_images())
            elif cmd == "get_images_and_points":
                remote.send(env.get_images_and_points())

            # misc
            elif cmd == "render_up":
                remote.send(env.render_up())

            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
        except EOFError:
            break


from stable_baselines3.common.vec_env.base_vec_env import VecEnv


class SubVecEnv(VecEnv):
    """
    Creates a multiprocess vectorized wrapper for multiple environments, distributing each environment to its own
    process, allowing significant speed up when the environment is computationally complex.

    For performance reasons, if your environment is not IO bound, the number of environments should not exceed the
    number of logical cores on your CPU.

    .. warning::

        Only 'forkserver' and 'spawn' start methods are thread-safe,
        which is important when TensorFlow sessions or other non thread-safe
        libraries are used in the parent (see issue #217). However, compared to
        'fork' they incur a small start-up cost and have restrictions on
        global variables. With those methods, users must wrap the code in an
        ``if __name__ == "__main__":`` block.
        For more information, see the multiprocessing documentation.

    :param env_fns: Environments to run in subprocesses
    :param start_method: method used to start the subprocesses.
           Must be one of the methods returned by multiprocessing.get_all_start_methods().
           Defaults to 'forkserver' on available platforms, and 'spawn' otherwise.
    """

    def __init__(self, env_fns, start_method=None):
        self.num_envs = len(env_fns)
        self.waiting = False
        self.closed = False

        if start_method is None:
            forkserver_available = "forkserver" in mp.get_all_start_methods()
            start_method = "forkserver" if forkserver_available else "spawn"
        ctx = mp.get_context(start_method)

        # Launch processes
        self.remotes, self.work_remotes = zip(
            *[ctx.Pipe() for _ in range(self.num_envs)]
        )
        self.processes = []
        for work_remote, remote, env_fn in zip(
            self.work_remotes, self.remotes, env_fns
        ):
            args = (work_remote, remote, CloudpickleWrapper(env_fn))
            # daemon=True: if the main process crashes, we should not cause things to hang
            process = ctx.Process(target=_worker, args=args, daemon=True)
            process.start()
            self.processes.append(process)
            work_remote.close()

        # Get observation and action spaces
        self.remotes[0].send(("get_spaces", None))
        observation_space, action_space = self.remotes[0].recv()

        super().__init__(len(env_fns), observation_space, action_space)

    # gym interface
    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), list(infos)

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def seed(self, seed=None):
        for idx, remote in enumerate(self.remotes):
            remote.send(("seed", seed + idx))
        return [remote.recv() for remote in self.remotes]

    def seed_sysid(self, seed=None):
        for _, remote in enumerate(self.remotes):
            remote.send(("seed", seed))
        return [remote.recv() for remote in self.remotes]

    def reset(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        results = [remote.recv() for remote in self.remotes]
        return np.stack(results)

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for process in self.processes:
            process.join()
        self.closed = True

    def render(self, data="rgb_array"):
        for pipe in self.remotes:
            pipe.send(("render", data))
        imgs = [pipe.recv() for pipe in self.remotes]
        return np.stack(imgs)

    # misc
    def get_attr(self, attr_name, indices=None):
        """Return attribute from vectorized environment (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("get_attr", attr_name))
        return [remote.recv() for remote in target_remotes]

    def set_attr(self, attr_name, value, indices=None):
        """Set attribute inside vectorized environments (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("set_attr", (attr_name, value)))
        for remote in target_remotes:
            remote.recv()

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """Call instance methods of vectorized environments."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("env_method", (method_name, method_args, method_kwargs)))
        return [remote.recv() for remote in target_remotes]

    def env_is_wrapped(self, wrapper_class, indices):
        """Check if worker environments are wrapped with a given wrapper"""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("is_wrapped", wrapper_class))
        return [remote.recv() for remote in target_remotes]

    def _get_indices(self, indices):
        """Convert a flexibly-typed reference to environment indices to an implied list of indices."""
        if indices is None:
            indices = range(self.num_envs)
        elif isinstance(indices, int):
            indices = [indices]
        return indices

    def _get_target_remotes(self, indices):
        """Get the connection object to communicate with the wanted envs that are in subprocesses."""
        indices = self._get_indices(indices)
        return [self.remotes[i] for i in indices]

    # physics parameters
    def get_parameters(self, indices=None):
        """Return (physics) parameters from vectorized environment (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("get_parameters", None))
        return [remote.recv() for remote in target_remotes]

    def set_parameters(self, values, indices=None):
        """Set (physics) parameters inside vectorized environments (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote, value in zip(target_remotes, values):
            remote.send(("set_parameters", value))
        for remote in target_remotes:
            remote.recv()

    def get_parameters_distribution(self, indices=None):
        """Return (physics) parameter dict from vectorized environment (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("get_parameters_distribution", None))
        return [remote.recv() for remote in target_remotes]

    def set_parameters_distribution(self, values, indices=None):
        """Set (physics) parameter dict inside vectorized environments (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote, value in zip(target_remotes, values):
            remote.send(("set_parameters_distribution", value))
        for remote in target_remotes:
            remote.recv()

    # pose
    def set_obj_pose(self, values, indices=None):
        """Set obj position inside vectorized environments (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote, value in zip(target_remotes, values):
            remote.send(("set_obj_pose", value))
        for remote in target_remotes:
            remote.recv()

    def get_obj_pose(self, indices=None):
        """Return workspace from vectorized environment (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("get_obj_pose", None))
        return [remote.recv() for remote in target_remotes]

    # asid reward
    def get_full_state(self, indices=None):
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("get_full_state", None))
        return [remote.recv() for remote in target_remotes]

    def set_full_state(self, values, indices=None):
        target_remotes = self._get_target_remotes(indices)
        for remote, value in zip(target_remotes, values):
            remote.send(("set_full_state", value))
        for remote in target_remotes:
            remote.recv()

    # observations
    def get_observation(self, indices=None):
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("get_observation", None))
        return [remote.recv() for remote in target_remotes]

    def get_images(self):
        for pipe in self.remotes:
            pipe.send(("render", None))
        outputs = [pipe.recv() for pipe in self.remotes]
        return outputs

    def get_images_and_points(self, indices=None):
        """Return points for a given view from vectorized environment (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("get_images_and_points", None))
        return [remote.recv() for remote in target_remotes]

    # misc
    def render_up(self, indices=None):
        """Reset robot to up, take snapshot, reset robot to init. Return snapshot."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("render_up", None))
        return [remote.recv() for remote in target_remotes]


class CloudpickleWrapper:
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)

    :param var: the variable you wish to wrap for pickling with cloudpickle
    """

    def __init__(self, var):
        self.var = var

    def __getstate__(self):
        return cloudpickle.dumps(self.var)

    def __setstate__(self, var):
        self.var = cloudpickle.loads(var)
