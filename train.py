import os
import time
import dmc2gym
import torch
import numpy as np
import utils
from logger import Logger
from replay_buffer import ReplayBuffer
from video import VideoRecorder
import algorithms
from arguments import parse_args

torch.backends.cudnn.benchmark = True

def make_env(cfg, pixels=True):
    # per dreamer: https://github.com/danijar/dreamer/blob/02f0210f5991c7710826ca7881f19c64a012290c/wrappers.py#L26
    camera_id = 2 if cfg.domain_name == 'quadruped' else 0

    env = dmc2gym.make(domain_name=cfg.domain_name,
                       task_name=cfg.task_name,
                       seed=cfg.seed,
                       visualize_reward=False,
                       from_pixels=pixels,
                       height=cfg.image_size if pixels else None,
                       width=cfg.image_size if pixels else None,
                       frame_skip=cfg.action_repeat,
                       camera_id=camera_id,
                       channels_first=True)

    env = utils.FrameStack(env, k=cfg.frame_stack)

    env.seed(cfg.seed)

    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1

    return env


class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.path.join(os.getcwd(), cfg.log_dir, cfg.exp_name, cfg.algorithm, str(cfg.seed))
        assert not os.path.exists(self.work_dir), f'specified working directory {self.work_dir} already exists'
        os.makedirs(self.work_dir)
        print(f'workspace: {self.work_dir}')
        self.save_dir = os.path.join(self.work_dir, "trained_models")
        utils.write_info(cfg, os.path.join(self.work_dir, 'config.log'))
        self.cfg = cfg

        self.logger = Logger(self.work_dir,
                             save_tb=False,
                             log_frequency=cfg.log_freq,
                             action_repeat=cfg.action_repeat,
                             agent=cfg.algorithm)

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        self.train_env = make_env(cfg)
        self.env = self.train_env

        # set up correlated environments
        if self.cfg.correlated_with_colour:
            original_rgb = np.copy(self.env.physics.model.mat_rgba)[:, :3]
            self.colourA = np.copy(original_rgb)
            self.colourB = np.copy(original_rgb)
            self.colourA[1, :] = [0., 0., 1.0]
            self.colourB[1, :] = [0., 1.0, 0.]
            self.probabilities = [[self.cfg.correlation_probability, 1-self.cfg.correlation_probability], [1-self.cfg.correlation_probability, self.cfg.correlation_probability]]
            self.test_probabilities = [[self.cfg.test_correlation_probability, 1-self.cfg.test_correlation_probability], [1-self.cfg.test_correlation_probability, self.cfg.test_correlation_probability]]
            xml_pathA = os.path.join("world_models", f"{cfg.domain_name}_A.xml")
            envA = make_env(cfg)
            envA.physics.reload_from_xml_path(xml_pathA)
            xml_pathB = os.path.join("world_models", f"{cfg.domain_name}_B.xml")
            envB = make_env(cfg)
            envB.physics.reload_from_xml_path(xml_pathB)
            self.envs = [envA, envB]
            object = np.random.choice([0, 1])
            self.current_colour = eval(np.random.choice(["self.colourA", "self.colourB"], p=self.probabilities[object]))
            self.env = self.envs[object]
            self.env.reset(colour=self.current_colour)
        else:
            self.envs = False
            self.env.reset()

        action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]

        self.agent = algorithms.make_agent(self.env.observation_space.shape, self.env.action_space.shape, action_range, cfg)

        self.replay_buffer = ReplayBuffer(self.env.observation_space.shape,
                                          self.env.action_space.shape,
                                          self.cfg.replay_buffer_capacity,
                                          self.cfg.image_pad, self.device,
                                          True if self.cfg.algorithm=='svea_cmid' else False)

        self.video_recorder = VideoRecorder(self.work_dir if cfg.save_video else None)

        self.step = 0

    def evaluate(self):
        eval_env = self.env
        eval_envs = self.envs
        if self.cfg.correlated_with_colour:
            eval_probs = self.probabilities

        average_episode_reward = 0

        self.video_recorder.init(enabled=True)

        for episode in range(self.cfg.num_eval_episodes):
            if eval_envs:
                object = np.random.choice(range(len(eval_envs)))
                eval_env = eval_envs[object]
            if self.cfg.correlated_with_colour:
                self.current_colour = eval(np.random.choice(["self.colourA", "self.colourB"], p=eval_probs[object]))
                obs = eval_env.reset(colour=self.current_colour)
            else:
                obs = eval_env.reset()

            done = False
            episode_reward = 0
            episode_step = 0

            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)

                obs, reward, done, info = eval_env.step(action)

                self.video_recorder.record(eval_env)
                episode_reward += reward
                episode_step += 1

            average_episode_reward += episode_reward
            self.video_recorder.save(f'{self.step}.mp4')

        average_episode_reward /= self.cfg.num_eval_episodes
        self.logger.log('eval/episode_reward', average_episode_reward, self.step)
        self.logger.dump(self.step)

    def run(self):
        episode, episode_reward, episode_step, done = 0, 0, 1, True
        total_num_steps = self.cfg.num_train_steps + self.cfg.num_test_steps
        start_time = time.time()

        while self.step <= (total_num_steps + 1):
            if done:
                if self.step > 0:
                    self.logger.log('train/duration', time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(self.step, save=(self.step > self.cfg.num_seed_steps))

                # evaluate agent periodically
                if self.step % self.cfg.eval_freq == 0:
                    self.logger.log('eval/episode', episode, self.step)
                    self.evaluate()

                self.logger.log('train/episode_reward', episode_reward, self.step)

                if self.envs:
                    object = np.random.choice(range(len(self.envs)))
                    self.env = self.envs[object]
                if self.cfg.correlated_with_colour:
                    self.current_colour = eval(np.random.choice(["self.colourA", "self.colourB"], p=self.probabilities[object]))
                    obs = self.env.reset(colour=self.current_colour)
                else:
                    obs = self.env.reset()
                prev_obs = obs.copy()

                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)

            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # run training update
            if self.step >= self.cfg.num_seed_steps:
                for _ in range(self.cfg.num_train_iters):
                    self.agent.update(self.replay_buffer, self.logger, self.step)

            if self.step > 0 and self.step % self.cfg.save_freq == 0:
                saveables = {
                    "actor": self.agent.actor.state_dict(),
                    "critic": self.agent.critic.state_dict(),
                    "critic_target": self.agent.critic_target.state_dict()
                }
                if self.cfg.algorithm == "svea_cmid":
                    saveables["cmid_discriminator"] = self.agent.cmid_discriminator.state_dict()
                elif self.cfg.algorithm == "svea_ted":
                    saveables["ted_classifier"] = self.agent.ted_classifier.state_dict()
                save_at = os.path.join(self.save_dir, f"env_step{self.step * self.cfg.action_repeat}")
                os.makedirs(save_at, exist_ok=True)
                torch.save(saveables, os.path.join(save_at, "models.pt"))

            next_obs, reward, done, info = self.env.step(action)

            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            episode_reward += reward

            self.replay_buffer.add(obs, action, reward, next_obs, done, done_no_max, episode, prev_obs)

            prev_obs = obs
            obs = next_obs
            episode_step += 1
            self.step += 1

            if self.step == self.cfg.num_train_steps:
                print("Switching to test env")
                if self.cfg.correlated_with_colour:
                    self.probabilities = self.test_probabilities
                done = True

def main(cfg):
    from train import Workspace as W
    global workspace
    workspace = W(cfg)
    workspace.run()


if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)
