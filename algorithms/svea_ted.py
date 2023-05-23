import torch
import torch.nn.functional as F
import utils
from algorithms.svea import SVEA
from algorithms.models import TEDClassifier

class SVEA_TED(SVEA):
	def __init__(self, obs_shape, action_shape, action_range, cfg):
		super().__init__(obs_shape, action_shape, action_range, cfg)
		self.svea_alpha = cfg.svea_alpha
		self.svea_beta = cfg.svea_beta

		self.ted_coef = cfg.ted_coef
		self.ted_classifier = TEDClassifier(cfg).to(self.device)
		self.actor.encoder.copy_head_weights_from(self.critic.encoder)

		# do not update encoder params with critic optimizer (retain gradient instead)
		critic_params = list(self.critic.Q1.parameters()) + list(self.critic.Q2.parameters())
		self.critic_optimizer = torch.optim.Adam(critic_params, lr=cfg.critic_lr)
		ted_params = list(self.ted_classifier.parameters()) + list(self.critic.encoder.parameters())
		self.ted_optimizer = torch.optim.Adam(ted_params, lr=cfg.ted_lr)
		self.ted_loss = torch.nn.BCEWithLogitsLoss()
		self.ted_classifier.train()

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
		critic_loss.backward(retain_graph=True)
		self.critic_optimizer.step()

	def update_actor_and_alpha(self, obs, logger, step):
		# detach conv filters, so we don't update them with the actor loss
		dist = self.actor(obs, detach_encoder_conv=True, detach_encoder_head=True)
		action = dist.rsample()
		log_prob = dist.log_prob(action).sum(-1, keepdim=True)

		actor_Q1, actor_Q2 = self.critic(obs, action, detach_encoder_conv=True, detach_encoder_head=True)

		actor_Q = torch.min(actor_Q1, actor_Q2)

		actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

		logger.log('train_actor/loss', actor_loss, step)
		logger.log('train_actor/target_entropy', self.target_entropy, step)
		logger.log('train_actor/entropy', -log_prob.mean(), step)

		# optimize the actor
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		self.actor.log(logger, step)

		self.log_alpha_optimizer.zero_grad()
		alpha_loss = (self.alpha *
					  (-log_prob - self.target_entropy).detach()).mean()
		logger.log('train_alpha/loss', alpha_loss, step)
		logger.log('train_alpha/value', self.alpha, step)
		alpha_loss.backward()
		self.log_alpha_optimizer.step()

	def ted_update(self, obs, next_obs, obs_aug, same_episode_obs, replay_buffer, logger, step):
		if self.svea_alpha == self.svea_beta:
			obs = torch.cat((obs, obs_aug), dim=0)
			num_samples = obs.shape[0]
			obs_rep = self.critic.encoder(obs)
			with torch.no_grad():
				next_obs_rep = self.critic_target.encoder(next_obs)
				next_obs_rep = torch.cat((next_obs_rep, next_obs_rep), dim=0)

			# Stack the consecutive observations to make temporal samples
			non_iid_samples = torch.stack([obs_rep, next_obs_rep], dim=1)
			# All temporal samples are given a label of 1
			non_iid_labels = torch.ones((num_samples))

			# Create the non-temporal different episode samples
			rnd_idx = torch.randperm(num_samples)
			diff_ep_iid_samples = torch.stack([obs_rep, next_obs_rep[rnd_idx]], dim=1)
			# All non-temporal samples are given a label of 0
			diff_ep_iid_labels = torch.zeros((num_samples))

			# Create the non-temporal same episode samples
			with torch.no_grad():
				next_entry_rep = self.critic_target.encoder(same_episode_obs)
				next_entry_rep = torch.cat((next_entry_rep, next_entry_rep), dim=0)
			same_ep_iid_samples = torch.stack([obs_rep, next_entry_rep], dim=1)
			same_ep_iid_labels = torch.zeros((num_samples))

			samples = torch.cat([non_iid_samples, diff_ep_iid_samples, same_ep_iid_samples])
			labels = torch.cat([non_iid_labels, diff_ep_iid_labels, same_ep_iid_labels]).to(self.device)

			r = self.ted_classifier(samples)
			ted_loss = self.ted_loss(r, labels) * float(self.ted_coef) * (self.svea_alpha + self.svea_beta)

		else:
			num_samples = obs.shape[0]
			obs_rep = self.critic.encoder(obs)
			obs_aug_rep = self.critic.encoder(obs_aug)
			with torch.no_grad():
				next_obs_rep = self.critic_target.encoder(next_obs)

			# Stack the consecutive observations to make temporal samples
			non_iid_samples = torch.stack([obs_rep, next_obs_rep], dim=1)
			# All temporal samples are given a label of 1
			non_iid_labels = torch.ones((num_samples))

			# Create the non-temporal different episode samples
			rnd_idx = torch.randperm(num_samples)
			diff_ep_iid_samples = torch.stack([obs_rep, next_obs_rep[rnd_idx]], dim=1)
			# All non-temporal samples are given a label of 0
			diff_ep_iid_labels = torch.zeros((num_samples))

			# Create the non-temporal same episode samples
			with torch.no_grad():
				next_entry_rep = self.critic_target.encoder(same_episode_obs)
			same_ep_iid_samples = torch.stack([obs_rep, next_entry_rep], dim=1)
			same_ep_iid_labels = torch.zeros((num_samples))

			samples = torch.cat([non_iid_samples, diff_ep_iid_samples, same_ep_iid_samples])
			labels = torch.cat([non_iid_labels, diff_ep_iid_labels, same_ep_iid_labels]).to(self.device)

			# Create the same types of samples for the augmented observations
			non_iid_samples_aug = torch.stack([obs_aug_rep, next_obs_rep], dim=1)
			non_iid_labels_aug = torch.ones((num_samples))

			rnd_idx_aug = torch.randperm(num_samples)
			diff_ep_iid_samples_aug = torch.stack([obs_aug_rep, next_obs_rep[rnd_idx_aug]], dim=1)
			diff_ep_iid_labels_aug = torch.zeros((num_samples))

			same_ep_iid_samples_aug = torch.stack([obs_aug_rep, next_entry_rep], dim=1)
			same_ep_iid_labels_aug = torch.zeros((num_samples))

			samples_aug = torch.cat([non_iid_samples_aug, diff_ep_iid_samples_aug, same_ep_iid_samples_aug])
			labels_aug = torch.cat([non_iid_labels_aug, diff_ep_iid_labels_aug, same_ep_iid_labels_aug]).to(self.device)

			r = self.ted_classifier(samples)
			r_aug = self.ted_classifier(samples_aug)
			ted_loss = ((self.ted_loss(r, labels) * self.svea_alpha) + (
						self.ted_loss(r_aug, labels_aug) * self.svea_beta)) * float(self.ted_coef)

		if logger is not None:
			logger.log('train_ted/loss', ted_loss, step)

		ted_loss.backward()
		self.ted_optimizer.step()

	def update(self, replay_buffer, logger, step):
		obs, action, reward, next_obs, not_done, obs_aug, \
			same_episode_obs = replay_buffer.sample_svea_ted(self.batch_size)

		# Zero grad here as we will retain critic gradient
		self.ted_optimizer.zero_grad()

		logger.log('train/batch_reward', reward.mean(), step)

		self.update_critic(obs, action, reward, next_obs, not_done, obs_aug, logger, step)

		if step % self.actor_update_frequency == 0:
			self.update_actor_and_alpha(obs, logger, step)

		self.ted_update(obs, next_obs, obs_aug, same_episode_obs, replay_buffer, logger, step)

		if step % self.critic_target_update_frequency == 0:
			utils.soft_update_params(self.critic.Q1, self.critic_target.Q1, self.critic_tau)
			utils.soft_update_params(self.critic.Q2, self.critic_target.Q2, self.critic_tau)
			utils.soft_update_params(self.critic.encoder, self.critic_target.encoder, self.encoder_tau)