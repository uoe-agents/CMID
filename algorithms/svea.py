import torch
import torch.nn.functional as F
import utils
from algorithms.sac import SAC

class SVEA(SAC):
	def __init__(self, obs_shape, action_shape, action_range, cfg):
		super().__init__(obs_shape, action_shape, action_range, cfg)
		self.svea_alpha = cfg.svea_alpha
		self.svea_beta = cfg.svea_beta

	def update_critic(self, obs, action, reward, next_obs, not_done, obs_aug, logger, step):
		with torch.no_grad():
			dist = self.actor(next_obs)
			next_action = dist.rsample()
			log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
			target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
			target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob
			target_Q = reward + (not_done * self.discount * target_V)

		if self.svea_alpha == self.svea_beta:
			obs = torch.cat((obs, obs_aug), dim=0)
			action = torch.cat((action, action), dim=0)
			target_Q = torch.cat((target_Q, target_Q), dim=0)

			current_Q1, current_Q2 = self.critic(obs, action)
			critic_loss = (self.svea_alpha + self.svea_beta) * \
						  (F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q))
		else:
			current_Q1, current_Q2 = self.critic(obs, action)
			critic_loss = self.svea_alpha * \
						  (F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q))

			current_Q1_aug, current_Q2_aug = self.critic(obs_aug, action)
			critic_loss += self.svea_beta * \
						   (F.mse_loss(current_Q1_aug, target_Q) + F.mse_loss(current_Q2_aug, target_Q))

		if logger is not None:
			logger.log('train_critic/loss', critic_loss, step)

		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

	def update(self, replay_buffer, logger, step):
		obs, action, reward, next_obs, not_done, obs_aug = replay_buffer.sample_svea(self.batch_size)

		logger.log('train/batch_reward', reward.mean(), step)

		self.update_critic(obs, action, reward, next_obs, not_done, obs_aug, logger, step)

		if step % self.actor_update_frequency == 0:
			self.update_actor_and_alpha(obs, logger, step)

		if step % self.critic_target_update_frequency == 0:
			utils.soft_update_params(self.critic.Q1, self.critic_target.Q1, self.critic_tau)
			utils.soft_update_params(self.critic.Q2, self.critic_target.Q2, self.critic_tau)
			utils.soft_update_params(self.critic.encoder, self.critic_target.encoder, self.encoder_tau)