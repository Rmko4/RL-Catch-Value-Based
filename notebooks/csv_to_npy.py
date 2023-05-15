# %%
from pathlib import Path

import numpy as np
import pandas as pd


# %%
GROUP_NUMBER = 29

results_dir = Path("../results")
file_name_prefix = f"group_{GROUP_NUMBER}_catch_rewards"

# %%
files = list(results_dir.glob("*.csv"))

# %%
for i, file in enumerate(files):
    df = pd.read_csv(file)
    rewards = df['train_DQN - test/total_reward'] / 10
    rewards_array = rewards.to_numpy()
    filename = f'{file_name_prefix}_{i}.npy'
    np.save(results_dir/filename, rewards_array)