import numpy as np
import kornia
import torch
import torch.nn as nn
from algorithms import augmentations

class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, image_pad, device, cmid):
        self.capacity = capacity
        self.device = device
        self.collect_prev_obs = cmid

        self.aug_trans = nn.Sequential(
            nn.ReplicationPad2d(image_pad),
            kornia.augmentation.RandomCrop((obs_shape[-1], obs_shape[-1])))

        if len(obs_shape)==1:
            self.obses = np.empty((capacity, *obs_shape), dtype=np.float32)
            self.next_obses = np.empty((capacity, *obs_shape), dtype=np.float32)
        else:
            self.obses = np.empty((capacity, *obs_shape), dtype=np.uint8)
            self.next_obses = np.empty((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((capacity, 1), dtype=np.float32)
        self.episode = np.empty((capacity, 1), dtype=np.float32)

        if self.collect_prev_obs:
            if len(obs_shape) == 1:
                self.prev_obses = np.empty((capacity, *obs_shape), dtype=np.float32)
            else:
                self.prev_obses = np.empty((capacity, *obs_shape), dtype=np.uint8)

        self.idx = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done, done_no_max, episode=None, prev_obs=None):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.not_dones_no_max[self.idx], not done_no_max)
        np.copyto(self.episode[self.idx], episode)

        if self.collect_prev_obs:
            np.copyto(self.prev_obses[self.idx], prev_obs)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        idxs = np.random.randint(0,
                                 (self.capacity - 1) if self.full else (self.idx - 1),
                                 size=batch_size)

        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]

        obses = torch.as_tensor(obses, device=self.device).float()
        next_obses = torch.as_tensor(next_obses, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs], device=self.device)

        return obses, actions, rewards, next_obses, not_dones_no_max, same_episode_obs, None, None, None, None

    def sample_drq(self, batch_size):
        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=batch_size)

        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]

        obses_aug = obses.copy()
        next_obses_aug = next_obses.copy()

        obses = torch.as_tensor(obses, device=self.device).float()
        next_obses = torch.as_tensor(next_obses, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs], device=self.device)

        obses_aug = torch.as_tensor(obses_aug, device=self.device).float()
        next_obses_aug = torch.as_tensor(next_obses_aug, device=self.device).float()

        obses = self.aug_trans(obses)
        next_obses = self.aug_trans(next_obses)
        obses_aug = self.aug_trans(obses_aug)
        next_obses_aug = self.aug_trans(next_obses_aug)

        return obses, actions, rewards, next_obses, not_dones_no_max, obses_aug, next_obses_aug

    def sample_svea(self, batch_size):
        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=batch_size)

        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]

        obses_aug = obses.copy()

        obses = torch.as_tensor(obses, device=self.device).float()
        next_obses = torch.as_tensor(next_obses, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs], device=self.device)

        obses_aug = torch.as_tensor(obses_aug, device=self.device).float()

        obses = self.aug_trans(obses)
        next_obses = self.aug_trans(next_obses)
        obses_aug = augmentations.random_conv(obses_aug)

        return obses, actions, rewards, next_obses, not_dones_no_max, obses_aug

    def sample_svea_ted(self, batch_size):
        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=batch_size)

        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]

        obses_aug = obses.copy()

        obses = torch.as_tensor(obses, device=self.device).float()
        next_obses = torch.as_tensor(next_obses, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs], device=self.device)

        obses_aug = torch.as_tensor(obses_aug, device=self.device).float()

        obses = self.aug_trans(obses)
        next_obses = self.aug_trans(next_obses)
        obses_aug = augmentations.random_conv(obses_aug)

        idxs_from_same_episode = []
        for i in range(batch_size):
            # Create non-temporal same episode sample
            sample_idx = idxs[i]
            sample_episode = self.episode[sample_idx]
            try:
                all_same_episode_idxs = np.nonzero(self.episode == sample_episode)[0]
                idx_to_remove = 1 - np.isin(all_same_episode_idxs, [sample_idx - 1, sample_idx, sample_idx + 1])
                x = all_same_episode_idxs * idx_to_remove
                reduced_same_episode_idxs = list(x[x != 0])
                idx_from_same_episode = np.random.choice(reduced_same_episode_idxs)
            except:
                # In the rare case there are no other samples from the same episode, we use one from a different episode
                idx_from_same_episode = np.random.choice(
                    [j for j in range(self.capacity if self.full else self.idx) if
                     j not in [sample_idx - 1, sample_idx, sample_idx + 1]])
            idxs_from_same_episode.append(idx_from_same_episode)
        same_episode_obs = torch.as_tensor(self.next_obses[idxs_from_same_episode], device=self.device).float()
        same_episode_obs = self.aug_trans(same_episode_obs)

        return obses, actions, rewards, next_obses, not_dones_no_max, obses_aug, same_episode_obs

    def sample_svea_cmid(self, batch_size):
        idxs = np.random.randint(1,
                                 self.capacity if self.full else self.idx,
                                 size=batch_size)

        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]

        obses_aug = obses.copy()

        obses = torch.as_tensor(obses, device=self.device).float()
        next_obses = torch.as_tensor(next_obses, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs], device=self.device)

        obses_aug = torch.as_tensor(obses_aug, device=self.device).float()

        obses = self.aug_trans(obses)
        next_obses = self.aug_trans(next_obses)
        obses_aug = augmentations.random_conv(obses_aug)

        prev_obses = self.prev_obses[idxs]
        prev_obses = torch.as_tensor(prev_obses, device=self.device).float()
        prev_obses = self.aug_trans(prev_obses)
        prev_actions = torch.as_tensor(self.actions[np.maximum((idxs-1), 0)], device=self.device)
        prev_masks = torch.as_tensor(self.not_dones[np.maximum((idxs-1), 0)], device=self.device)

        return obses, actions, rewards, next_obses, not_dones_no_max, obses_aug, prev_obses, \
            prev_actions, prev_masks

    def sample_curl(self, batch_size):
        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=batch_size)

        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]

        pos = obses.copy()

        obses = torch.as_tensor(obses, device=self.device).float()
        next_obses = torch.as_tensor(next_obses, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs], device=self.device)

        pos = torch.as_tensor(pos, device=self.device).float()

        obses = self.aug_trans(obses)
        next_obses = self.aug_trans(next_obses)
        pos = self.aug_trans(pos)

        return obses, actions, rewards, next_obses, not_dones_no_max, pos