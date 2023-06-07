# %%
from matplotlib import style
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import re

# %%
results_dir = Path("../results")
all_result_path = results_dir/"Catch_test_all.csv"

pattern = r"train\d_(.*) - .*"

name_map = {"DQN": "DQN",
            "DDQN": "DDQN",
            "prio_DQN": "Prioritized Replay DQN",
            "DQV": "DQV",
            "DQV_max": "DQV-max",
            "Dueling_architecture": "Dueling Architecture",
            }

algs = ["DQN", "DDQN", "prio_DQN", "DQV", "DQV_max", "Dueling_architecture"]

# %%
df = pd.read_csv(all_result_path)


# %%
# Extract the columns that end with 'test/total_reward'
col_names = [col for col in df.columns if col.endswith('test/total_reward')]

# Extract the alg_name from the column names
alg_names = [re.match(pattern, col).group(1) for col in col_names]


# %%
# Create a new DataFrame with the extracted columns
grouped_data = df[col_names].groupby(
    alg_names, axis=1).apply(lambda x: x.values)


# %%
style.use(["cleanplot", "font_libertine"])


# %%
fig_width = 3.26
fig_height = 3

# Create the figure object with the custom size
fig = plt.figure(figsize=(fig_width, fig_height))

for alg_name, data in list(grouped_data.items())[0:4]:
    data = grouped_data[alg_name]
    mu = np.mean(data, axis=1)
    sigma = np.std(data, axis=1)
    epochs = np.arange(len(mu))
    plt.plot(epochs, mu, label=name_map[alg_name], lw=1)
    plt.fill_between(epochs, mu + 0.5*sigma, mu - 0.5*sigma, alpha=0.3)

plt.xlabel("Test Epoch")
plt.ylabel("Average Reward")
plt.legend()
fig.savefig(results_dir/"results_all.pdf", dpi=300, bbox_inches='tight')
plt.show()


# %%
# Plot by moving average
fig_width = 3.26
fig_height = 3

# Create the figure object with the custom size
fig = plt.figure(figsize=(fig_width, fig_height))

# Set the window size for the moving average
window_size = 10

for alg_name in algs:
    data = grouped_data[alg_name]/10
    mu = np.mean(data, axis=1)
    sigma = np.std(data, axis=1)
    epochs = np.arange(len(mu))

    # Calculate the moving average using convolution
    padded_mu = np.pad(mu, (window_size-1, 0), mode='edge')
    padded_sigma = np.pad(sigma, (window_size-1, 0), mode='edge')
    ma = np.convolve(padded_mu, np.ones(window_size)/window_size, mode='valid')
    ma_sd = np.convolve(padded_sigma, np.ones(
        window_size)/window_size, mode='valid')

    plt.plot(epochs, ma, linestyle='-', label=f"{name_map[alg_name]}", lw=1)

    plt.fill_between(epochs, ma + 0.5*ma_sd, ma - 0.5*ma_sd, alpha=0.3)

plt.hlines(0.25, epochs[0], epochs[-1],
           linestyle='--', color='k', label="Random")

plt.xlabel("Test Epoch")
plt.ylabel("Average Reward")
plt.legend()
plt.savefig(results_dir / "results_all_catch_ma.pdf", dpi=300)
plt.show()

# %%
