from os import name
import random
from pathlib import Path
from typing import List

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger, WandbLogger

from argparser import get_args
from catch_module import CatchRLModule
from video_logger import VideoLoggerCallback

PROJECT_NAME = "RL-Catch"
LOGS_DIR = Path("logs/")


def train(hparams):
    logger = WandbLogger(name=hparams.run_name,
                         project=PROJECT_NAME,
                         save_dir=LOGS_DIR,
                         log_model=True,
                         anonymous="allow",)
    csv_logger = CSVLogger(save_dir=LOGS_DIR)

    hparams: dict = vars(hparams)
    hparams.pop('run_name')
    max_epochs = hparams.pop('max_epochs')
    log_video = hparams.pop('log_video')

    callbacks = []
    if log_video:
        callbacks.append(VideoLoggerCallback())

    catch_module = CatchRLModule(**hparams)
    trainer = Trainer(max_epochs=max_epochs,
                      logger=[logger, csv_logger],
                      log_every_n_steps=1,
                      callbacks=callbacks,
                      )
    trainer.fit(catch_module)


if __name__ == "__main__":
    hparams = get_args()
    train(hparams)
