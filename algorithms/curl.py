import torch
import torch.nn.functional as F
import utils
from algorithms import models
from algorithms.sac import SAC


class CURL(SAC):
    def __init__(self, obs_shape, action_shape, action_range, cfg):
        super().__init__(obs_shape, action_shape, action_range, cfg)
        self.aux_update_freq = cfg.aux_update_freq

        self.curl_head = models.CURLHead(self.critic.encoder).cuda()

        self.curl_optimizer = torch.optim.Adam(
            self.curl_head.parameters(), lr=cfg.aux_lr, betas=(cfg.aux_beta, 0.999)
        )
        self.train()

    def train(self, training=True):
        super().train(training)
        if hasattr(self, 'curl_head'):
            self.curl_head.train(training)

    def update_curl(self, x, x_pos, logger, step):
        assert x.size(-1) == 84 and x_pos.size(-1) == 84

        z_a = self.curl_head.encoder(x)
        with torch.no_grad():
            z_pos = self.critic_target.encoder(x_pos)

        logits = self.curl_head.compute_logits(z_a, z_pos)
        labels = torch.arange(logits.shape[0]).long().cuda()
        curl_loss = F.cross_entropy(logits, labels)

        self.curl_optimizer.zero_grad()
        curl_loss.backward()
        self.curl_optimizer.step()
        if logger is not None:
            logger.log('train/aux_loss', curl_loss, step)

    def update(self, replay_buffer, logger, step):
        obs, action, reward, next_obs, not_done, pos = replay_buffer.sample_curl(self.batch_size)

        logger.log('train/batch_reward', reward.mean(), step)

        self.update_critic(obs, action, reward, next_obs, not_done, logger, step)

        if step % self.actor_update_frequency == 0:
            self.update_actor_and_alpha(obs, logger, step)

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic.Q1, self.critic_target.Q1, self.critic_tau)
            utils.soft_update_params(self.critic.Q2, self.critic_target.Q2, self.critic_tau)
            utils.soft_update_params(self.critic.encoder, self.critic_target.encoder, self.encoder_tau)

        if step % self.aux_update_freq == 0:
            self.update_curl(obs, pos, logger, step)
