# %%
from catch import CatchEnv
import numpy as np
import random

# Create the environment
env = CatchEnv()


# %%
episode_rewards = []
# Run a few episodes
for episode in range(10000):
    terminal = False
    total_reward = 0.

    obs = env.reset()

    while not terminal:
        # time.sleep(0.01)
        # Choose a random action
        action = random.randint(0, 2)

        # Perform the action in the environment
        state, reward, terminal = env.step(action)

        # Update the total reward
        total_reward += float(reward)

    # Print the total reward for the episode
    print("Episode:", episode + 1, "Total Reward:", total_reward)
    episode_rewards.append(total_reward)

episode_rewards = np.array(episode_rewards)
print("Average reward:", np.mean(episode_rewards))
print("Standard deviation:", np.std(episode_rewards))

# %%
# Close the environment
env.close()
