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