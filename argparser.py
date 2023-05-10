import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Training agent on Catch")

    parser.add_argument("--run_name", type=str, default="train",
                        help="Name of the run")
    parser.add_argument("--algorithm", type=str, default="DQN",
                        choices=["DQN", "Dueling_architecture", "DQV_max"],
                        help="Type of algorithm to use for training")
    parser.add_argument("--log_video", action="store_true",
                        help="Whether to log video of agent's performance")
    parser.add_argument("--max_steps", type=int, default=50000,
                        help="Maximum number of steps to train for")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Learning rate for training")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--epsilon_start", type=float, default=0.1,
                        help="Initial epsilon")
    parser.add_argument("--epsilon_end", type=float, default=0.01,
                        help="Final epsilon")
    parser.add_argument("--epsilon_decay_rate", type=float, default=1000,
                        help="Number of steps to decay epsilon over")
    parser.add_argument("--buffer_capacity", type=int, default=1000,
                        help="Capacity of replay buffer")
    parser.add_argument("--replay_warmup_steps", type=int, default=100,
                        help="Number of steps to warm up replay buffer")
    parser.add_argument("--prioritized_replay", action="store_true",
                        help="Whether to use prioritized replay")
    parser.add_argument("--prioritized_replay_alpha", type=float, default=0.6,
                        help="Alpha parameter for prioritized replay")
    parser.add_argument("--prioritized_replay_beta", type=float, default=0.4,
                        help="Beta parameter for prioritized replay")
    parser.add_argument("--target_net_update_freq", type=int, default=None,
                        help="Number of steps between target network updates")
    parser.add_argument("--soft_update_tau", type=float, default=1e-3,
                        help="Tau for soft target network updates")
    parser.add_argument("--double_q_learning", action="store_true",
                        help="Whether to use double Q-learning")
    # parser.add_argument("--dueling_architecture", action="store_true",
    #                     help="Whether to use dueling architecture.")
    parser.add_argument("--hidden_size", type=int, default=128,
                        help="Number of hidden units in feedforward network.")
    parser.add_argument("--n_filters", type=int, default=32,
                        help="Number of filters in convolutional network.")

    args = parser.parse_args()
    return args
