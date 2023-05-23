import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    # environment
    parser.add_argument('--domain_name', default='cartpole')
    parser.add_argument('--task_name', default='swingup')
    parser.add_argument('--exp_name', default='cartpole_colours')
    parser.add_argument('--episode_length', default=1000, type=int)
    parser.add_argument('--device', default="cuda", type=str)
    parser.add_argument('--seed', default=0, type=int)

    # correlation
    parser.add_argument('--correlated_with_colour', default="False", type=str)
    parser.add_argument('--correlation_probability', default=0.95, type=float) # sum of probabilities on leading diagonal, see paper
    parser.add_argument('--test_correlation_probability', default=0.05, type=float) # sum of probabilities on leading diagonal, see paper

    # train
    parser.add_argument('--algorithm', default='sac', type=str)
    parser.add_argument('--action_repeat', default=4, type=int)
    parser.add_argument('--num_train_steps', default=1000, type=int)
    parser.add_argument('--num_test_steps', default=0, type=int)
    parser.add_argument('--num_train_iters', default=1, type=int)
    parser.add_argument('--replay_buffer_capacity', default=100000, type=int)
    parser.add_argument('--num_seed_steps', default=1000, type=int)

    # observation
    parser.add_argument('--image_size', default=84, type=int)
    parser.add_argument('--frame_stack', default=3, type=int)
    parser.add_argument('--image_pad', default=4, type=int)

    # eval
    parser.add_argument('--eval_freq', default=1000, type=int)
    parser.add_argument('--num_eval_episodes', default=10, type=int)

    # log
    parser.add_argument('--log_freq', default=1000, type=int)
    parser.add_argument('--save_freq', default=250000, type=int)
    parser.add_argument('--log_dir', default='runs', type=str)
    parser.add_argument('--save_video', default=False, action='store_true')

    # agent
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--hidden_dim', default=1024, type=int)
    parser.add_argument('--hidden_depth', default=2, type=int)

    # actor
    parser.add_argument('--actor_lr', default=1e-3, type=float)
    parser.add_argument('--actor_beta', default=0.9, type=float)
    parser.add_argument('--actor_log_std_min', default=-10, type=float)
    parser.add_argument('--actor_log_std_max', default=2, type=float)
    parser.add_argument('--actor_update_freq', default=2, type=int)
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--alpha_lr', default=1e-4, type=float)

    # critic
    parser.add_argument('--critic_lr', default=1e-3, type=float)
    parser.add_argument('--critic_tau', default=0.01, type=float)
    parser.add_argument('--critic_target_update_freq', default=2, type=int)

    # encoder
    parser.add_argument('--encoder_tau', default=0.05, type=float)
    parser.add_argument('--num_conv_layers', default=4, type=int)
    parser.add_argument('--feature_dim', default=56, type=int)
    parser.add_argument('--num_filters', default=32, type=int)

    # SVEA
    parser.add_argument('--svea_alpha', default=0.5, type=float)
    parser.add_argument('--svea_beta', default=0.5, type=float)

    # CMID
    parser.add_argument('--cmid_encoder_lr', default=1e-3, type=float)
    parser.add_argument('--cmid_discriminator_lr', default=1e-2, type=float)
    parser.add_argument('--adversarial_loss_coef', default=100, type=float)
    parser.add_argument('--cmid_knn', default=5, type=int)

    # TED
    parser.add_argument('--ted_coef', default=1)
    parser.add_argument('--ted_lr', default=1e-3, type=float)

    # CURL
    parser.add_argument('--aux_lr', default=1e-3, type=float)
    parser.add_argument('--aux_beta', default=0.9, type=float)
    parser.add_argument('--aux_update_freq', default=2, type=int)

    args = parser.parse_args()

    assert args.algorithm in {'sac', 'rad', 'curl', 'drq', 'svea', 'svea_ted', 'svea_cmid'}, f'specified algorithm "{args.algorithm}" is not supported'
    assert args.seed is not None, 'must provide seed for experiment'
    assert args.log_dir is not None, 'must provide a log directory for experiment'

    args.correlated_with_colour = eval(args.correlated_with_colour)

    return args
