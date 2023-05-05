import random
from pathlib import Path
from typing import List

import cv2
import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger

from catch import CatchEnv
from argparser import get_args
from catch_module import CatchRLModule

VIDEO_PATH = Path("videos")


def writeVideo(history: List[np.ndarray],
               output_file: str = 'output.mp4'):
    # Define the video codec and output video dimensions
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_shape = (640, 640)

    video_writer = cv2.VideoWriter(output_file, fourcc, 4, frame_shape, False)

    # Iterate over the frames in history and write them to the output video
    for frame in history:
        write_frame = (255*frame).astype(np.uint8)
        write_frame = cv2.resize(
            write_frame, frame_shape, interpolation=cv2.INTER_NEAREST)
        video_writer.write(write_frame)

    video_writer.release()


def run_environment():
    env = CatchEnv()
    number_of_episodes = 1

    for ep in range(number_of_episodes):
        env.reset()

        history = []

        state, reward, terminal = env.step(1)

        while not terminal:
            state, reward, terminal = env.step(random.randint(0, 2))
            print("Reward obtained by the agent: {}".format(reward))
            state = np.squeeze(state)

            # history.append(state[:, :, -1])
            history.append(env.image.copy())

        writeVideo(history, str(VIDEO_PATH/"output.mp4"))
        print("End of the episode")


def train(hparams):
    catch_module = CatchRLModule(**vars(hparams))
    trainer = Trainer(max_steps=1e4, logger=CSVLogger(save_dir="logs/"),)
    trainer.fit(catch_module)


if __name__ == "__main__":
    hparams = get_args()
    train(hparams)
