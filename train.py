import os
import hydra
import torch
import numpy as np
import io
import matplotlib.pyplot as plt
from PIL import Image as PILImage

from stable_baselines3 import SAC
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback

from env.wrappers.asid_vec import make_env, make_vec_env
from utils.experiment import hydra_to_dict, set_random_seed, setup_wandb
from utils.logger import Video, configure_logger


class LoggerCallback(BaseCallback):
    """
    A custom callback that derives from BaseCallback.
    """

    def __init__(
        self, eval_envs, eval_interval, save_dir, save_interval, verbose=False
    ):
        super(LoggerCallback, self).__init__(verbose)

        self.eval_interval = eval_interval

        self.save_dir = save_dir
        self.save_interval = save_interval

        self.eval_envs = eval_envs
        self.eval_envs.reset()

        self.reward_sum = 0.0

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.
        """
        self.reward_sum += np.mean(self.locals["rewards"])

        # log
        if np.all(self.locals["dones"]):
            self.logger.record("train/return", self.reward_sum)
            self.reward_sum = 0

        # evaluate
        if (self.n_calls - 1) % self.eval_interval == 0:
            evaluate(
                self.model, self.eval_envs, self.logger, step=self.n_calls, tag="eval"
            )
            self.compute_obj_dev(tag="eval")

        # save
        if (self.n_calls - 1) % self.save_interval == 0:
            self.model.save(self.save_dir + "_step_" + str(self.n_calls))

        return True

    def compute_obj_dev(self, tag="eval"):
        """
        This methods compute the deviation (L2) of the object pose from its initial pose.
        """
        total_reward = 0
        all_dev = np.zeros(self.eval_envs.num_envs)

        obs = self.eval_envs.reset()
        init_obj_poses = self.eval_envs.get_obj_pose()

        dones = False
        while not np.all(dones):
            actions, _state = self.model.predict(obs)
            next_obs, rewards, dones, infos = self.eval_envs.step(actions)
            obs = next_obs

            total_reward += np.sum(rewards)
            for i in range(self.eval_envs.num_envs):
                all_dev[i] += np.linalg.norm(
                    init_obj_poses[i] - self.eval_envs.get_obj_pose()[i]
                )

        self.logger.record(
            f"{tag}/current_policy_value", total_reward / self.eval_envs.num_envs
        )
        self.logger.record(f"{tag}/total_obj_displacement_l2", np.mean(all_dev))


def evaluate(policy, eval_envs, logger, step, tag="eval"):
    """
    Evaluate the policy on the environment
    Args:
        policy (object): policy to evaluate
        eval_envs (object): environment to evaluate the policy
        logger (object): logger to record the results
        tag (str): tag for the logger, e.g., train, eval
    """
    render = torch.cuda.device_count() > 0
    eval_envs.render_mode = "rgb_array"

    obs = eval_envs.reset()
    dones, infos, frames = False, [], []

    episode_returns = []

    # Assume all eval_envs terminate at the same time
    while not np.all(dones):

        if render:
            frames.append(eval_envs.render())

        # Select action
        actions, _ = policy.predict(obs, deterministic=True)
        # Take environment step
        next_obs, rewards, dones, infos = eval_envs.step(actions)
        episode_returns.append(rewards)

        obs = next_obs

    # Record episode statistics
    avg_return = np.mean(np.sum(np.stack(episode_returns).transpose(1, 0), axis=1))
    logger.record(f"{tag}/return", avg_return)

    if render:

        video = np.stack(frames)

        # Record video with return plot
        returns = np.stack(episode_returns).transpose(1, 0)
        fig_height_px, fig_width_px = video.shape[-3:-1]
        dpi = 100

        plotss = []
        for ret in returns:

            plots = []

            for l in range(len(ret)):

                # Plot return
                plt.figure(figsize=(fig_width_px / dpi, fig_height_px / dpi), dpi=dpi)
                plt.plot(ret[:l])
                plt.ylim(-10, 1000 * 1.1)
                plt.xlim(0, len(ret) - 1)
                plt.xticks(np.arange(0, len(ret), step=1))
                plt.xlabel("Steps")
                plt.ylabel("Return")

                # Figure to buffer to PIL to numpy
                buf = io.BytesIO()
                plt.savefig(buf, format="png")
                buf.seek(0)
                image = PILImage.open(buf)
                plot = np.array(image)
                plots.append(plot)
                buf.close()
                plt.close()

            plots = np.stack(plots)[..., :3]
            plotss.append(plots)

        plotss = np.stack(plotss, axis=1)
        video_ret = np.concatenate([video, plotss], axis=3)

        # Log videos
        # -> (b, t, 3, h, w)
        logger.record(
            f"{tag}/trajectory/env/camera_return",
            Video(video_ret.transpose(1, 0, 4, 2, 3), fps=20),
            exclude=["stdout"],
        )

        # # -> (b, t, 3, h, w)
        # logger.record(
        #     f"{tag}/trajectory/env/camera",
        #     Video(video.transpose(1, 0, 4, 2, 3), fps=20),
        #     exclude=["stdout"],
        # )

    logger.dump(step=step)


@hydra.main(config_path="configs/", config_name="explore_rod_sim", version_base="1.1")
def run_experiment(cfg):

    # logging
    if "wandb" in cfg.log.format_strings:
        run = setup_wandb(
            cfg,
            name=f"{cfg.exp_id}[explore][{cfg.seed}]",
            entity=cfg.log.entity,
            project=cfg.log.project,
        )
    set_random_seed(cfg.seed)
    logdir = os.path.join(cfg.log.dir, cfg.exp_id, str(cfg.seed), "explore")
    logger = configure_logger(logdir, cfg.log.format_strings)

    # train env
    envs = make_vec_env(
        robot_cfg_dict=hydra_to_dict(cfg.robot),
        env_cfg_dict=hydra_to_dict(cfg.env),
        asid_cfg_dict=hydra_to_dict(cfg.asid),
        num_workers=cfg.num_workers,
        seed=cfg.seed,
        device_id=0,
    )

    # algorithm
    device = torch.device(("cuda") if torch.cuda.is_available() else "cpu")
    n_actions = envs.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    model = SAC(
        "MlpPolicy",
        envs,
        device=device,
        learning_starts=cfg.train.algorithm.learning_starts,
        ent_coef=cfg.train.algorithm.ent_coef,
        train_freq=(cfg.train.algorithm.train_freq // cfg.num_workers, "step"),
        gradient_steps=cfg.train.algorithm.gradient_steps,
        action_noise=action_noise,
    )

    # eval env
    eval_envs = make_vec_env(
        robot_cfg_dict=hydra_to_dict(cfg.robot),
        env_cfg_dict=hydra_to_dict(cfg.env),
        asid_cfg_dict=hydra_to_dict(cfg.asid),
        num_workers=cfg.num_workers_eval,
        seed=cfg.seed + 100,
        device_id=0,
    )

    # set logger
    model.set_logger(logger)
    ckptdir = os.path.join(logdir, "policy")

    callback = LoggerCallback(
        eval_envs=eval_envs,
        eval_interval=cfg.log.eval_interval,
        save_dir=ckptdir,
        save_interval=cfg.log.save_interval,
    )

    # train
    model.learn(
        total_timesteps=cfg.train.total_timesteps,
        callback=callback,
        progress_bar=True,
    )

    # save & load policy
    model.save(ckptdir)
    model = model.load(ckptdir)

    # close envs
    envs.close()
    eval_envs.close()


if __name__ == "__main__":
    run_experiment()
