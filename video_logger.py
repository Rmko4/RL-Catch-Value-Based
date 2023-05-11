from pathlib import Path
from typing import List

import cv2
import imageio
import numpy as np
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback

from catch_module import CatchRLModule

VIDEO_PATH = Path("videos")/"output.gif"


def writeMP4(history: List[np.ndarray],
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


def writeGIF(history: List[np.ndarray],
             output_file: str = 'output.gif'):
    # Define the output image dimensions
    frame_shape = (640, 640)

    # Create a list of image frames
    frames = []
    for frame in history:
        write_frame = (255*frame).astype(np.uint8)
        write_frame = cv2.resize(
            write_frame, frame_shape, interpolation=cv2.INTER_NEAREST)
        frames.append(write_frame)

    # Save the frames as an animated GIF
    imageio.mimsave(output_file, frames, duration=250)


class VideoLoggerCallback(Callback):
    def __init__(self, save_every_n_epochs=5) -> None:
        self.interval = save_every_n_epochs
        self.epoch = 0

    def on_train_epoch_end(self, trainer: Trainer, pl_module: CatchRLModule) -> None:
        if self.epoch % self.interval != 0:
            self.epoch += 1
            return

        history = []

        agent = pl_module.agent
        agent.reset()
        history.append(agent.env.image.copy())

        terminal = False
        while not terminal:
            _, terminal = agent.step(
                freeze_time=True, epsilon=0.)
            history.append(agent.env.image.copy())
            
        history.append(agent.env.image.copy())

        writeGIF(history, str(VIDEO_PATH))

        wandb.log(
            {"test/play_video": wandb.Video(str(VIDEO_PATH))}
        )

        self.epoch += 1
