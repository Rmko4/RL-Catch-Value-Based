import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Training agent on Catch")

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
    parser.add_argument("--epsilon_decay_steps", type=float, default=1000,
                        help="Number of steps to decay epsilon over")
    parser.add_argument("--buffer_capacity", type=int, default=1000,
                        help="Capacity of replay buffer")
    parser.add_argument("--replay_warmup_steps", type=int, default=10,
                        help="Number of steps to warm up replay buffer")
    parser.add_argument("--target_net_update_freq", type=int, default=100,
                        help="Number of steps between target network updates")

    args = parser.parse_args()
    return args
