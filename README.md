# Deep Value-Based Reinforcement Learning
An implementation of the following methods: Deep Q-Network (DQN), Double Deep Q-Network (DDQN), Dueling Architecture, Deep Quality-Value (DQV), and DQV-max. The implementation is based on the following papers:
- [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)
- [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)
- [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581)
- [Deep Quality-Value (DQV) Learning](https://arxiv.org/abs/1810.00368)
- [Approximating two value functions instead of one: towards characterizing a new family of Deep Reinforcement Learning algorithms](https://arxiv.org/abs/1909.01779)

The methods are trained and evaluated on the Catch game.
## Example of trained agent
![output](https://github.com/Rmko4/RL-Catch-Value-Based/assets/55834815/5022a617-f350-41b8-8414-f0fd33dd6b83)


## Running the code
### Installation
To install all dependencies, run the following command:
```bash
pip install -r requirements.txt
```

### Training
To train the agent, run the following command:
```bash
python source/train_agent.py [Training Options]
```

#### Training Options:
- *--run_name* (*str*): Name of the run.
- *--algorithm* (*{DQN,Dueling_architecture,DQV,DQV_max}*) : Type of algorithm to use for training.
<br><br>
- *--log_video*: *Whether* to log video of agent's performance.
<br><br>
- *--max_epochs* (*int*): Maximum number of steps to train for.
- *--batch_size* (*int*): Batch size for training.
- *--batches_per_step* (*int*): Number of batches to sample from replay buffer per agent step.
- *--optimizer* (*{Adam,RMSprop,SGD}*): Optimizer to use for training.
- *--learning_rate* (*float*): Learning rate for training.
- *--gamma* (*float*): Discount factor.
- *--epsilon_start* (*float*): Initial epsilon.
- *--epsilon_end* (*float*): Final epsilon.
- *--epsilon_decay_rate* (*int*): Number of steps to decay epsilon over.
- *--buffer_capacity* (*int*): Capacity of replay buffer.
- *--replay_warmup_steps* (*int*): Number of steps to warm up the replay buffer.
<br><br>
- *--target_net_update_freq* (*int*): Number of steps between target network updates.
- *--soft_update_tau* (*float*): Tau for soft target network updates.
- *--double_q_learning*: Whether to use double Q-learning.
<br><br>
- *--hidden_size* (*int*): Number of hidden units in the feedforward network.
- *--n_filters* (*int*): Number of filters in the convolutional network.
<br><br>
- *--prioritized_replay*: Whether to use prioritized replay.
- *--prioritized_replay_alpha* (*float*): Alpha parameter for prioritized replay.
- *--prioritized_replay_beta* (*float*): Beta parameter for prioritized replay.
