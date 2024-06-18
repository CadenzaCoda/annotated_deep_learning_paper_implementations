import numpy as np

import gym
# import barc_gym.envs
import multiprocessing
import cv2
from collections import deque
from mars_explorer.envs.settings import DEFAULT_CONFIG as conf
from labml_nn.rl.game import Game


class MyGame:
    def __init__(self, seed=None):
        self.env = gym.make('exploConf-v1', conf=conf)
        self.env.reset()

        self.obs_4 = deque(maxlen=4)


    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        observation = self._process_obs(observation)
        self.obs_4.append(observation)
        return np.asarray(self.obs_4), reward, done, info
        # return self.env.step(action)
        # return observation, reward, done, truncated, info

    def reset(self, seed=None):
        obs = self.env.reset()
        obs = self._process_obs(obs)
        for _ in range(4):
            self.obs_4.append(np.copy(obs))
        return np.asarray(self.obs_4)
        # return observation

    @staticmethod
    def _process_obs(obs):
        obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
        return obs


def worker_process(remote: multiprocessing.connection.Connection, seed: int):
    """
    ##Worker Process

    Each worker process runs this method
    """

    # create game
    game = MyGame(seed)

    # wait for instructions from the connection and execute them
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            remote.send(game.step(data))
        elif cmd == "reset":
            remote.send(game.reset())
        elif cmd == "close":
            remote.close()
            break
        else:
            raise NotImplementedError


class Worker:
    """
    Creates a new worker and runs it in a separate process.
    """

    def __init__(self, seed):
        self.child, parent = multiprocessing.Pipe()
        self.process = multiprocessing.Process(target=worker_process, args=(parent, seed))
        self.process.start()


if __name__ == '__main__':
    env = gym.make('exploConf-v1', conf=conf)
    from matplotlib import pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    obs = env.reset()
    # ax1.imshow(obs)
    action = env.action_space.sample()
    obs, rew, done, info = env.step(action)
    # ax2.imshow(obs)
    print(action)
    print(obs.shape)
    # plt.show()
    print(rew)
    print(done)
    print(info)

    # game = MyGame(seed=0)
    # obs, reward, done, truncated, info = game.step(np.zeros(2, ))
    # print(obs, reward, done, info)
