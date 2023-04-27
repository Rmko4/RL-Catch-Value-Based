from catch import CatchEnv
import cv2
import numpy as np
import random

from typing import List

from pathlib import Path

video_path = Path("videos")

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

        writeVideo(history, str(video_path/"output.mp4"))
        print("End of the episode")


if __name__ == "__main__":
    run_environment()
