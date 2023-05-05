from os import name
import random
from pathlib import Path
from typing import List

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger, WandbLogger

from argparser import get_args
from catch_module import CatchRLModule

PROJECT_NAME = "RL-Catch"
LOGS_DIR = Path("logs/")

def train(hparams):
    logger = WandbLogger(name=hparams.run_name, 
                         project=PROJECT_NAME,
                         save_dir=LOGS_DIR,
                        log_model=True,)
    csv_logger = CSVLogger(save_dir=LOGS_DIR)

    catch_module = CatchRLModule(**vars(hparams))
    trainer = Trainer(max_steps=1e4,
                      logger=[logger, csv_logger],
                      )
    trainer.fit(catch_module)


if __name__ == "__main__":
    hparams = get_args()
    train(hparams)
