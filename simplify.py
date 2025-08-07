import gym
import os
import numpy as np
from gym_pybullet_drones.envs.single_agent_rl.TakeoffAviary import TakeoffAviary
from scipy.spatial.transform import Rotation
from PIL import Image

env = gym.make('landing-aviary-v0', gui = True)
env2 = gym.make('landing-aviary-v0')

#os.system('rm -rf /home/user/landing/landing_rl/test_images')
#os.system('mkdir /home/user/landing/landing_rl/test_images')
max_step = 42452424242424
state=env.reset()
action = np.array((0.0, 0.0, 0.0))

for i in range(0,10):
    env.reset()
    env2.reset()
    for step in range(max_step):
        img, reward, done, info = env.step(action)
    
        im = Image.fromarray(img.squeeze())
        im = im.convert("L")
        im.save("test_images/drone_view_{0}.png".format(step))