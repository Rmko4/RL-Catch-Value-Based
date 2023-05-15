# %%
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

# %%
results_dir = Path("../results")

# %%
files = list(results_dir.glob("*.npy"))

# %%
data = [np.load(file) for file in files]
data = np.array(data)

# %%
data = np.mean(data, axis=0)

# %%
from matplotlib import style
style.use(["cleanplot", "font_libertine"])
# %%
epochs = np.arange(len(data))
plt.plot(epochs, data)
plt.xlabel("Test Epoch")
plt.ylabel("Average Reward")
plt.savefig(results_dir/"results.png", dpi=300)
plt.show()
# %%
