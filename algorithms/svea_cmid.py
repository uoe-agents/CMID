import numpy as np
import torch
import torch.nn.functional as F

import utils
from algorithms import models
from algorithms.svea import SVEA

class SVEA_CMID(SVEA):
    def __init__(self, obs_shape, action_shape, action_range, cfg):
        super().__init__(obs_shape, action_shape, action_range, cfg)

        self.K = cfg.feature_dim
        self.cmid_discriminator = models.CMIDDiscriminator(action_shape, cfg).to(self.device)
        self.discriminator_optimizer = torch.optim.Adam(self.cmid_discriminator.discriminator.parameters(), lr=cfg.cmid_discriminator_lr)
        self.cmid_discriminator.train()
        self.encoder_optimizer = torch.optim.Adam(self.critic.encoder.parameters(), lr=cfg.cmid_encoder_lr)
        self.cmid_update_counter = 0  # allows masks to be created only once

    def cmid_update(self, obs, obs_aug, prev_obs, prev_action, prev_masks, logger, step):
        obs = torch.cat((obs, obs_aug), dim=0)
        prev_obs = torch.cat((prev_obs, prev_obs), dim=0)
        prev_action = torch.cat((prev_action, prev_action), dim=0)
        prev_masks = torch.cat((prev_masks, prev_masks), dim=0)
        prev_action = prev_action*prev_masks

        z = self.critic.get_representation(obs)

        with torch.no_grad():
            prev_z = self.critic_target.get_representation(prev_obs)

        n = z.shape[0]
        # Only need to create masks once
        if self.cmid_update_counter == 0:
            self.mask = torch.zeros_like(z.detach().repeat(self.K, 1))
            for k in range(self.K):
                self.mask[(n * k):(n * (k + 1)), k] = 1.0

        conditioning_set = torch.cat((prev_z.repeat(self.K, 1) * self.mask,
                                      prev_action.repeat(self.K, 1)), dim=-1)

        knn_a = []
        for k in range(self.K):
            knn_a_k = torch.cat(((prev_z)[:, k].unsqueeze(-1), prev_action), dim=-1).detach()
            # scale knn inputs
            knn_a_k = (knn_a_k - knn_a_k.mean(dim=0)) / knn_a_k.std(dim=0)
            knn_a.append(knn_a_k)

        knn_a = torch.stack(knn_a, dim=0)
        dists = torch.linalg.norm(knn_a.detach().unsqueeze(2) - knn_a.detach().unsqueeze(1), dim=-1)
        knn_values, knn_indices = dists.topk(self.cfg.cmid_knn, largest=False, dim=-1)

        conditioning_set = conditioning_set.view(self.K, n, -1)

        z_repeated = z.repeat(self.K, 1, 1)
        closest_z = torch.cat([z_repeated[i, knn_indices[i]] for i in range(self.K)]).view(self.K, n, self.cfg.cmid_knn, -1)
        closest_conditioning_set = torch.cat([conditioning_set[i, knn_indices[i]] for i in range(self.K)]).view(self.K, n, self.cfg.cmid_knn, -1)
        shuffle_idx = torch.randperm(closest_z.shape[2])
        closest_z_shuffled = closest_z[:, :, shuffle_idx]

        for k in range(self.K):
            closest_z_shuffled[k, :, :, k] = closest_z[k, :, :, k]

        z_repeated = z.repeat(self.K * self.cfg.cmid_knn, 1)
        conditioning_set = conditioning_set.view(self.K * n, -1).repeat(self.cfg.cmid_knn, 1)
        closest_z_shuffled = closest_z_shuffled.view(self.K * n * self.cfg.cmid_knn, -1)
        closest_conditioning_set = closest_conditioning_set.view(self.K * n * self.cfg.cmid_knn, -1)

        true_labels = torch.ones(z_repeated.shape[0], 1).to(self.device)
        shuffled_labels = torch.zeros(closest_z_shuffled.shape[0], 1).to(self.device)
        labels = torch.cat((true_labels, shuffled_labels))

        input_z = torch.cat((z_repeated.detach(), closest_z_shuffled.detach()))
        input_conditioning_set = torch.cat((conditioning_set.detach(), closest_conditioning_set.detach()))

        preds = self.cmid_discriminator(input_z, input_conditioning_set)
        discriminator_loss = F.binary_cross_entropy_with_logits(preds, labels)

        logger.log('train_cmid/discriminator_loss', discriminator_loss, step)

        self.discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        self.discriminator_optimizer.step()

        true_preds = self.cmid_discriminator(z_repeated, conditioning_set)
        shuffled_labels = torch.zeros(z_repeated.shape[0], 1).to(self.device)
        adversarial_loss = F.binary_cross_entropy_with_logits(true_preds, shuffled_labels)

        encoder_loss = self.cfg.adversarial_loss_coef * adversarial_loss
        logger.log('train_cmid/adversarial_loss', adversarial_loss, step)

        self.encoder_optimizer.zero_grad()
        encoder_loss.backward()
        self.encoder_optimizer.step()

    def update(self, replay_buffer, logger, step):
        obs, action, reward, next_obs, not_done, obs_aug, prev_obs, \
            prev_action, prev_masks = replay_buffer.sample_svea_cmid(self.batch_size)

        logger.log('train/batch_reward', reward.mean(), step)

        self.update_critic(obs, action, reward, next_obs, not_done, obs_aug, logger, step)

        if step % self.actor_update_frequency == 0:
            self.update_actor_and_alpha(obs, logger, step)

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic.Q1, self.critic_target.Q1, self.critic_tau)
            utils.soft_update_params(self.critic.Q2, self.critic_target.Q2, self.critic_tau)
            utils.soft_update_params(self.critic.encoder, self.critic_target.encoder, self.encoder_tau)

        self.cmid_update(obs, obs_aug, prev_obs, prev_action, prev_masks, logger, step)
        self.cmid_update_counter += 1