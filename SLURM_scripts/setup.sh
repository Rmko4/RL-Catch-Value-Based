#! /bin/bash
module load Python
python -m venv ~/envs/drlenv
git clone git@github.com:Rmko4/Deep-Reinforcement-Learning.git ~/Deep-Reinforcement-Learning
cd ~/Deep-Reinforcement-Learning
source ~/envs/drlenv/bin/activate
pip install -r requirements.txt

srun --partition=gpushort --gpus-per-node=a100.20gb:1 --mem=200GB --time=3:00:00 --job-name=RL-Catch --pty /bin/bash