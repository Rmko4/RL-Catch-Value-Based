#! /bin/bash
module load Python
python -m venv ~/envs/drlenv
git clone git@github.com:Rmko4/Deep-Reinforcement-Learning.git ~/Deep-Reinforcement-Learning
cd ~/Deep-Reinforcement-Learning
source ~/envs/drlenv/bin/activate
pip install -r requirements.txt

cd ..
rm -rf ~/Deep-Reinforcement-Learning
git clone git@github.com:Rmko4/Deep-Reinforcement-Learning.git ~/Deep-Reinforcement-Learning
cp -r ~/Deep-Reinforcement-Learning/ $TMPDIR
# cd to working directory (repo)
cd $TMPDIR/Deep-Reinforcement-Learning/


srun --partition=gpushort --gpus-per-node=a100.20gb:1 --mem=200GB --time=3:00:00 --job-name=RL-Catch --pty /bin/bash

python train_agent.py --run_name train --max_steps 20000 --batch_size 32 --learning_rate 0.001 --gamma 0.99 --epsilon_start 1.0 --epsilon_end 0.01 --epsilon_decay_rate 2000 --buffer_capacity 5000 --replay_warmup_steps 10 --soft_update_tau 0.01 --hidden_size 32 --n_filters 16 --double_q_learning --dueling_architecture
