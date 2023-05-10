from catch import CatchEnv
import cv2
import numpy as np
import random

class CatchRender(CatchEnv):
    def __init__(self):
        super().__init__()
        self.renderer = None
        self.fps = 2
    
    def step(self, action):
        step_return = super().step(action)
        self.render()
        return step_return

    def render(self):
        if self.renderer is None:   
            self.renderer = cv2.namedWindow("Catch", cv2.WINDOW_NORMAL)
        self._render_frame()
    
    def _render_frame(self):
        cv2.imshow("Catch", self.image)
        cv2.waitKey(1000 // self.fps)
        

def run_environment():
    env = CatchRender()
    number_of_episodes = 1

    for ep in range(number_of_episodes):
        env.reset()

        state, reward, terminal = env.step(1)

        while not terminal:
            state, reward, terminal = env.step(random.randint(0, 2))
            print("Reward obtained by the agent: {}".format(reward))
            state = np.squeeze(state)

        print("End of the episode")


if __name__ == "__main__":
    run_environment()
