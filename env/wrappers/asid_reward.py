import copy
import numpy as np


class ASIDRewardWrapper:
    """
    Reward wrapper for ASID
    Args:
        envs (object): env to compute gradient
        delta (float): finite-difference step
        normalization (float): normalization factor
        max_reward (float): maximum reward
    """

    def __init__(self, envs, delta=0.1, normalization=1.0, max_reward=1000.0):
        self.envs = envs
        self.delta = delta
        self.normalization = normalization
        self.max_reward = max_reward

    def get_reward(self, full_state, action, params=None, verbose=False):
        """
        Compute reward
        Args:
            full_state (dict): full simulator state
            action (np.ndarray): action
            params (np.ndarray): physics parameters
            verbose (bool): verbose
        Returns:
            reward (float): reward
        """
        # compute gradient
        grad = self.estimate_loss_grad(full_state, action, params=params)
        if verbose:
            print("grad", "mean", np.mean(grad), "std", np.std(grad))

        # compute reward
        reward = np.trace(grad.T @ grad)
        if verbose:
            print("rew", "mean", np.mean(reward), "std", np.std(reward))

        # clip reward
        return np.clip(reward, 0.0, self.max_reward)

    def estimate_loss_grad(self, full_state, action, params=None):
        """
        Estimate gradient
        Args:
            full_state (dict): full simulator state
            action (np.ndarray): action
            params (np.ndarray): physics parameters
        Returns:
            grad (np.ndarray): gradient
        """
        # get parameters
        if params is None:
            current_param = self.envs.get_parameters()
        else:
            current_param = params

        params = copy.deepcopy(current_param)
        num_params = len(current_param)

        # compute gradient w/ finite-differences
        grad = None
        for i in range(num_params):

            params_temp = copy.deepcopy(params)
            # define +delta
            params_temp[i] += self.delta
            # apply theta_delta
            self.envs.set_parameters(params_temp)
            # set theta state
            self.envs.set_full_state(copy.deepcopy(full_state))
            # step environment
            obs1, _, _, _ = self.envs.step(action)

            params_temp = copy.deepcopy(params)
            # define -delta
            params_temp[i] -= self.delta
            # apply theta_delta
            self.envs.set_parameters(params_temp)
            # set theta state
            self.envs.set_full_state(copy.deepcopy(full_state))
            # step environment
            obs2, _, _, _ = self.envs.step(action)

            # compute gradient
            g = (obs1 - obs2) / (2 * self.delta)

            if grad is None:
                d = len(g)
                grad = np.zeros((d, num_params))
            grad[:, i] = g

            # reset theta
            self.envs.set_parameters(current_param)

        # normalize gradient
        return grad / self.normalization
