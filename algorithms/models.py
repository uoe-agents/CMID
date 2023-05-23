import torch
import torch.nn as nn
import utils

class Encoder(nn.Module):
    """Convolutional encoder for image-based observations."""
    def __init__(self, obs_shape, cfg, expert_feature_dim=None):
        super().__init__()

        assert len(obs_shape) == 3
        self.num_layers = cfg.num_conv_layers
        self.num_filters = cfg.num_filters
        self.output_logits = False
        self.feature_dim = cfg.feature_dim

        self.convs = nn.ModuleList([nn.Conv2d(obs_shape[0], self.num_filters, 3, stride=2)])
        for i in range(1, self.num_layers):
            self.convs.extend([nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1)])

        # get output shape
        x = torch.randn(*obs_shape).unsqueeze(0)
        conv = torch.relu(self.convs[0](x))
        for i in range(1, self.num_layers):
            conv = self.convs[i](conv)
        conv = conv.view(conv.size(0), -1)
        self.output_shape = conv.shape[1]

        self.head = nn.Sequential(
            nn.Linear(self.output_shape, self.feature_dim),
            nn.LayerNorm(self.feature_dim))

        self.out_dim = self.feature_dim

        self.apply(utils.weight_init)

        self.outputs = dict()

    def forward_conv(self, obs):
        obs = obs / 255.
        self.outputs['obs'] = obs

        conv = torch.relu(self.convs[0](obs))
        self.outputs['conv1'] = conv

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            self.outputs['conv%s' % (i + 1)] = conv

        h = conv.view(conv.size(0), -1)
        return h

    def forward(self, obs, detach_encoder_conv=False, detach_encoder_head=False):
        h = self.forward_conv(obs)

        if detach_encoder_conv:
            h = h.detach()

        out = self.head(h)

        if not self.output_logits:
            out = torch.tanh(out)

        if detach_encoder_head:
            out = out.detach()
        self.outputs['out'] = out

        return out

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        for i in range(self.num_layers):
            utils.tie_weights(src=source.convs[i], trg=self.convs[i])

    def copy_head_weights_from(self, source):
        """Tie head layers"""
        for i in range(2):
            utils.tie_weights(src=source.head[i], trg=self.head[i])

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_encoder/{k}_hist', v, step)
            if len(v.shape) > 2:
                logger.log_image(f'train_encoder/{k}_img', v[0], step)

        for i in range(self.num_layers):
            logger.log_param(f'train_encoder/conv{i + 1}', self.convs[i], step)


class Actor(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""
    def __init__(self, obs_shape, action_shape, cfg):
        super(Actor, self).__init__()
        self.algorithm = cfg.algorithm
        self.frame_stack = cfg.frame_stack
        self.stack_representations = True if cfg.algorithm=="svea_cmid" else False

        if self.stack_representations:
            obs_shape = (obs_shape[0] // self.frame_stack, obs_shape[1], obs_shape[2])
        self.encoder = Encoder(obs_shape, cfg)

        self.log_std_bounds = [cfg.actor_log_std_min, cfg.actor_log_std_max]

        if self.stack_representations:
            self.trunk = utils.mlp(self.encoder.feature_dim * self.frame_stack, cfg.hidden_dim,
                                   2 * action_shape[0], cfg.hidden_depth)
        else:
            self.trunk = utils.mlp(self.encoder.feature_dim, cfg.hidden_dim,
                                   2 * action_shape[0], cfg.hidden_depth)

        self.outputs = dict()

        self.trunk.apply(utils.weight_init)

    def forward(self, obs, detach_encoder_conv=False, detach_encoder_head=False):

        if self.stack_representations:
            N = obs.shape[0]
            obs = obs.view((N * self.frame_stack, -1, *obs.shape[2:]))
        z = self.encoder(obs, detach_encoder_conv=detach_encoder_conv, detach_encoder_head=detach_encoder_head)

        if self.stack_representations:
            z = z.view(N, -1)

        mu, log_std = self.trunk(z).chunk(2, dim=-1)
        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
        std = log_std.exp()

        self.outputs['mu'] = mu
        self.outputs['std'] = std

        dist = utils.SquashedNormal(mu, std)
        return dist

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_actor/{k}_hist', v, step)

        for i, m in enumerate(self.trunk):
            if type(m) == nn.Linear:
                logger.log_param(f'train_actor/fc{i}', m, step)

    def copy_gru_weights_from(self, source):
        """Tie gru weights"""
        self.gru.weight_ih_l0 = source.gru.weight_ih_l0
        self.gru.weight_hh_l0 = source.gru.weight_hh_l0
        self.gru.bias_ih_l0 = source.gru.bias_ih_l0
        self.gru.bias_hh_l0 = source.gru.bias_hh_l0

class Critic(nn.Module):
    """Critic network, employs double Q-learning."""
    def __init__(self, obs_shape, action_shape, cfg):
        super().__init__()
        self.algorithm = cfg.algorithm
        self.frame_stack = cfg.frame_stack
        self.stack_representations = True if cfg.algorithm=="svea_cmid" else False

        if self.stack_representations:
            obs_shape = (obs_shape[0] // self.frame_stack, obs_shape[1], obs_shape[2])
        self.encoder = Encoder(obs_shape, cfg)

        if self.stack_representations:
            self.Q1 = utils.mlp(self.encoder.feature_dim * self.frame_stack + action_shape[0],
                                cfg.hidden_dim, 1, cfg.hidden_depth)
            self.Q2 = utils.mlp(self.encoder.feature_dim * self.frame_stack + action_shape[0],
                                cfg.hidden_dim, 1, cfg.hidden_depth)
        else:
            self.Q1 = utils.mlp(self.encoder.feature_dim + action_shape[0],
                                cfg.hidden_dim, 1, cfg.hidden_depth)
            self.Q2 = utils.mlp(self.encoder.feature_dim + action_shape[0],
                                cfg.hidden_dim, 1, cfg.hidden_depth)

        self.outputs = dict()

        self.Q1.apply(utils.weight_init)
        self.Q2.apply(utils.weight_init)

    def forward(self, obs, action, detach_encoder_conv=False, detach_encoder_head=False):

        if self.stack_representations:
            N = obs.shape[0]
            obs = obs.view((N * self.frame_stack, -1, *obs.shape[2:]))
        z = self.encoder(obs, detach_encoder_conv=detach_encoder_conv, detach_encoder_head=detach_encoder_head)

        if self.stack_representations:
            z = z.view(N, -1)

        assert z.size(0) == action.size(0)

        obs_action = torch.cat([z, action], dim=-1)

        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2


    def get_representation(self, obs, detach_encoder_conv=False, detach_encoder_head=False):

        if self.stack_representations:
            N = obs.shape[0]
            obs = obs.view((N * self.frame_stack, -1, *obs.shape[2:]))
        z = self.encoder(obs, detach_encoder_conv=detach_encoder_conv, detach_encoder_head=detach_encoder_head)

        if self.stack_representations:
            rep_dim = z.shape[-1]
            z = z.view(N, -1)
            # take only most recent representation
            z = z[:, (rep_dim * (self.frame_stack - 1)):]

        return z

    def log(self, logger, step):
        self.encoder.log(logger, step)

        for k, v in self.outputs.items():
            logger.log_histogram(f'train_critic/{k}_hist', v, step)

        assert len(self.Q1) == len(self.Q2)
        for i, (m1, m2) in enumerate(zip(self.Q1, self.Q2)):
            assert type(m1) == type(m2)
            if type(m1) is nn.Linear:
                logger.log_param(f'train_critic/q1_fc{i}', m1, step)
                logger.log_param(f'train_critic/q2_fc{i}', m2, step)


class CURLHead(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

        self.W = nn.Parameter(torch.rand(self.encoder.out_dim, self.encoder.out_dim))
    def compute_logits(self, z_a, z_pos):
        """
        Uses logits trick for CURL:
        - compute (B,B) matrix z_a (W z_pos.T)
        - positives are all diagonal elements
        - negatives are all other elements
        - to compute loss use multiclass cross entropy with identity matrix for labels
        """
        Wz = torch.matmul(self.W, z_pos.T)  # (z_dim,B)
        logits = torch.matmul(z_a, Wz)  # (B,B)
        logits = logits - torch.max(logits, 1)[0][:, None]
        return logits


class TEDClassifier(nn.Module):
    """TED classifer to predict if the input pair is temporal or non-temporal."""
    def __init__(self, cfg):
        super().__init__()

        self.W = nn.Parameter(torch.empty(2, cfg.feature_dim))
        self.b = nn.Parameter(torch.empty((1, cfg.feature_dim)))
        self.W_bar = nn.Parameter(torch.empty((1, cfg.feature_dim)))
        self.b_bar = nn.Parameter(torch.empty((1, cfg.feature_dim)))
        self.c = nn.Parameter(torch.empty((1, 1)))

        self.W.requires_grad = True
        self.b.requires_grad = True
        self.W_bar.requires_grad = True
        self.b_bar.requires_grad = True
        self.c.requires_grad = True

        nn.init.orthogonal_(self.W)
        nn.init.orthogonal_(self.b)
        nn.init.orthogonal_(self.W_bar)
        nn.init.orthogonal_(self.b_bar)
        nn.init.orthogonal_(self.c)

    def forward(self, inputs):

        x = self.W * inputs
        x = torch.sum(x, dim=1)
        x = x + self.b
        x = torch.abs(x)

        y = torch.square((self.W_bar * torch.transpose(inputs, 1, 0)[0]) + self.b_bar)

        output = (torch.sum((x-y), dim=1) + self.c).squeeze()

        return output


class CMIDDiscriminator(nn.Module):
    def __init__(self, action_shape, cfg):
        super().__init__()

        self.feature_dim = cfg.feature_dim
        num_conditional_values = action_shape[0] + self.feature_dim

        self.discriminator = utils.mlp(self.feature_dim + num_conditional_values, cfg.hidden_dim, 1, cfg.hidden_depth)
        self.apply(utils.weight_init)

    def forward(self, inputs, condition):
        x = torch.cat((inputs, condition), dim=-1)
        return self.discriminator(x)