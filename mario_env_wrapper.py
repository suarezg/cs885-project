from torchvision import transforms
import gym
from gym.spaces import Box
import gym_super_mario_bros
import numpy as np
import torch
from gym.wrappers import FrameStack
from nes_py.wrappers import JoypadSpace
from torch import nn
from torch.distributions import Categorical
from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT, COMPLEX_MOVEMENT

'''
Environment class wrappers taken from:
    https://github.com/dredwardhyde/reinforcement-learning
'''

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(low=0, high=255, shape=self.observation_space.shape[:2], dtype=np.uint8)

    def observation(self, observation):
        transform = transforms.Grayscale()
        return transform(torch.tensor(np.transpose(observation, (2, 0, 1)).copy(), dtype=torch.float))


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        self.shape = (shape, shape)
        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transformations = transforms.Compose([transforms.Resize(self.shape), transforms.Normalize(0, 255)])
        return transformations(observation).squeeze(0)


def create_mario_env(world=1, stage=1, version=0, action_type=RIGHT_ONLY, shape=84,frame_skip=4, frame_stack=4, seed=None):
    '''
    create a mario OpenAI environment and utilize the wrappers above
    '''
    
    mario_env_string = 'SuperMarioBros-%d-%d-v%d' %(world, stage, version)
    env = gym_super_mario_bros.make(mario_env_string)
    env = JoypadSpace(env, action_type)
    env = FrameStack(ResizeObservation(GrayScaleObservation(SkipFrame(env, skip=frame_skip)), shape=shape), num_stack=frame_stack)
    if seed is not None:
        env.seed(seed)
        env.action_space.seed(seed)
        torch.manual_seed(seed)
        torch.random.manual_seed(seed)
        np.random.seed(seed)

    return env
