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
--max_epochs 50 \
--batch_size 128 \
--batches_per_step 1 \
--learning_rate 0.001 \
--gamma 0.99 \
--epsilon_start 0.5 \
--epsilon_end 0.01 \
--epsilon_decay_rate 100 \
--buffer_capacity 10000 \
--replay_warmup_steps 10 \
--soft_update_tau 0.05 \
--hidden_size 128 \
--n_filters 16 \
--algorithm DQV
