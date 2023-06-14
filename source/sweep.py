import wandb
import numpy as np 
import random

from argparser import get_args
from train_agent import PROJECT_NAME, train

sweep_configuration = {}

hparams = get_args()

# Initialize sweep by passing in config. 
# (Optional) Provide a name of the project.
sweep_id = wandb.sweep(
  sweep=sweep_configuration, 
  project=PROJECT_NAME
  )

def train_start():
    # run = wandb.init()
    train(hparams, config=wandb.config)

# Start sweep job.
wandb.agent(sweep_id, function=train_start, count=10)