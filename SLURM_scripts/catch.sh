#!/usr/bin/env bash
#SBATCH --time=2:00:00
#SBATCH --gpus-per-node=a100.20gb:1
#SBATCH --job-name=RL-Catch
#SBATCH --mem=200GB
#SBATCH --profile=task

module load Python
module load cuDNN
module load CUDA

source ~/envs/drlenv/bin/activate

# Copy git repo to local
cp -r ~/Deep-Reinforcement-Learning/ $TMPDIR
# cd to working directory (repo)
cd $TMPDIR/Deep-Reinforcement-Learning/

python train_agent.py \
--run_name train \
--max_steps 20000 \
--batch_size 32 \
--learning_rate 0.0005 \
--gamma 0.99 \
--epsilon_start 1.0 \
--epsilon_end 0.01 \
--epsilon_decay_rate 2000 \
--buffer_capacity 10000 \
--replay_warmup_steps 10 \
--soft_update_tau 0.02 \
--hidden_size 256 \
--n_filters 8 \
--double_q_learning