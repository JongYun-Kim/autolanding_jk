# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from collections import deque
from typing import Any, NamedTuple

#import dm_env
import numpy as np
import enum
#from dm_control import manipulation, suite
#from dm_control.suite.wrappers import action_scale, pixels
#from dm_env import StepType, specs
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.specs import BoundedArray, Array
from gym_pybullet_drones.envs.single_agent_rl.TakeoffAviary import TakeoffAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
import gym
import copy


class StepType(enum.IntEnum):
    """Defines the status of a `TimeStep` within a sequence."""
    # Denotes the first `TimeStep` in a sequence.
    FIRST = 0
    # Denotes any `TimeStep` in a sequence that is not FIRST or LAST.
    MID = 1
    # Denotes the last `TimeStep` in a sequence.
    LAST = 2

    def first(self) -> bool:
        return self is StepType.FIRST

    def mid(self) -> bool:
        return self is StepType.MID

    def last(self) -> bool:
        return self is StepType.LAST

class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any
    landing_info: Any
    drone_state: Any
    position_error: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        if isinstance(attr, str):
            return getattr(self, attr)
        else:
            return tuple.__getitem__(self, attr)


class ActionRepeatWrapper():
    def __init__(self, env, num_repeats):
        self._env = env
        self._num_repeats = num_repeats

    def step(self, action):
        reward = 0.0
        discount = 1.0
        for i in range(self._num_repeats):
            time_step = self._env.step(action)
            reward += (time_step.reward or 0.0) * discount
            discount *= time_step.discount
            if time_step.last():
                break

        return time_step._replace(reward=reward, discount=discount)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class FrameStackWrapper(gym.Wrapper):
    def __init__(self, env, num_frames, pixels_key='pixels'):
        super().__init__(env)
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)
        self._drone_states = deque([], maxlen=num_frames)

    def observation_spec(self):
        return BoundedArray((self._num_frames,84,84), int, minimum = 0, maximum = 255, name = 'observation')

    def action_spec(self):
        return BoundedArray((3,), np.float32, minimum = -1, maximum = 1, name = 'action')

    def drone_state_spec(self):
        return BoundedArray((self._num_frames,7), np.float32, minimum = -1, maximum = 1, name = 'drone_state')
    
    def reward_spec(self):
        return Array((1,), np.float32, 'reward')

    def multi_reward_spec(self):
        return Array((3,), np.float32, 'reward')

    def discount_spec(self):
        return Array((1,), np.float32, 'discount')

    def _transform_observation(self, time_step):
        assert len(self._frames) == self._num_frames
        obs = np.concatenate(list(self._frames), axis=0)
        drone_state = np.concatenate(list(self._drone_states), axis=0)
        time_step = list(time_step)
        time_step[0] = obs
        time_step.append(drone_state)
        return time_step

    def _extract_pixels(self, time_step):
        pixels = time_step[0].astype(int)
        # remove batch dim
        if len(pixels.shape) == 4:
            pixels = pixels[0]
        return pixels#.transpose(2, 0, 1).copy()

    def reset(self):
        time_step = self.env.reset().astype(int)
        pixels = time_step#.transpose(2, 0, 1).copy()
        #pixels = np.pad(pixels, ((0, 0), (0, 1), (0, 1)), mode='constant')
        next_velocity = copy.deepcopy(self.env.vel[0,:])/self.env.SPEED_LIMIT
        next_attitude = copy.deepcopy(self.env.quat[0,:])
        #next_delay = np.array([copy.deepcopy(self.env.AGGR_PHY_STEPS)/11])
        #drone_state = np.concatenate((next_velocity,next_attitude,next_delay))[None,:].astype(np.float32)
        drone_state = np.concatenate((next_velocity,next_attitude))[None,:].astype(np.float32)
        #pixels[0,84,0:7] = drone_state[0,:]*255
        for _ in range(self._num_frames):
            self._frames.append(pixels)
            self._drone_states.append(drone_state)
        return self._transform_observation(time_step)

    #def get_sim_time(self):
    #    return self.env.get_sim_time()

    def step(self, action):
        time_step = self.env.step(action)
        pixels = self._extract_pixels(time_step)
        #pixels = np.pad(pixels, ((0, 0), (0, 1), (0, 1)), mode='constant')
        next_velocity = copy.deepcopy(self.env.vel[0,:])/self.env.SPEED_LIMIT
        next_attitude = copy.deepcopy(self.env.quat[0,:])
        drone_state = np.concatenate((next_velocity,next_attitude))[None,:].astype(np.float32)
        #pixels[0,84,0:7] = drone_state[0,:]*255
        self._frames.append(pixels)
        self._drone_states.append(drone_state)
        return self._transform_observation(time_step)

    def get_vehicle_position(self):
        return self.env.get_vehicle_position()[0]

    def get_vehicle_velocity(self):
        return self.env.get_vehicle_velocity()[0]

    def get_uav_position(self):
        return self.env.pos[0,:]

    def get_uav_velocity(self):
        return self.env.vel[0,:]

    # def __getattr__(self, name):
    #     return getattr(self._env, name)

    def __getattr__(self, name): # TODO: 지피티가 준거니까 고쳐라 다시 위에 코맨트 해둠
        return getattr(self.env, name)

class FrameStackWrapperWithGimbalState(FrameStackWrapper):
    def __init__(self, env, num_frames, pixels_key='pixels'):
        super().__init__(env, num_frames, pixels_key)

    def observation_spec(self):
        return BoundedArray((self._num_frames,84,84), int, minimum = 0, maximum = 255, name = 'observation')

    def action_spec(self):
        return BoundedArray((5,), np.float32, minimum = -1, maximum = 1, name = 'action')

    def drone_state_spec(self):
        return BoundedArray((self._num_frames, 11), np.float32, minimum = -1, maximum = 1, name = 'drone_state')

    def reset(self):
        time_step = self.env.reset().astype(int)
        pixels = time_step

        next_velocity = copy.deepcopy(self.env.vel[0,:])/self.env.SPEED_LIMIT
        next_attitude = copy.deepcopy(self.env.quat[0,:])
        next_gimbal = copy.deepcopy(self.gimbal_state_quat) # (4,)
        drone_state = np.concatenate((next_velocity,next_attitude,next_gimbal))[None,:].astype(np.float32)

        for _ in range(self._num_frames):
            self._frames.append(pixels)
            self._drone_states.append(drone_state)

        return self._transform_observation(time_step)

    #def get_sim_time(self):
    #    return self.env.get_sim_time()

    def step(self, action):
        time_step = self.env.step(action)
        pixels = self._extract_pixels(time_step)

        next_velocity = copy.deepcopy(self.env.vel[0,:])/self.env.SPEED_LIMIT
        next_attitude = copy.deepcopy(self.env.quat[0,:])
        next_gimbal = copy.deepcopy(self.gimbal_state_quat) # (4,)
        drone_state = np.concatenate((next_velocity,next_attitude,next_gimbal))[None,:].astype(np.float32)

        self._frames.append(pixels)
        self._drone_states.append(drone_state)

        return self._transform_observation(time_step)


class FrameStackWrapperWithGimbalOracle(FrameStackWrapperWithGimbalState):
    def __init__(self, env, num_frames, pixels_key='pixels'):
        super().__init__(env, num_frames, pixels_key)

    def action_spec(self):
        return BoundedArray((3,), np.float32, minimum = -1, maximum = 1, name = 'action')

class ExtendedTimeStepWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        pass

    def reset(self):
        time_step = self.env.reset()
        return self._augment_time_step(time_step, step_type = StepType.FIRST)

    def step(self, action):
        time_step = self.env.step(action)
        if time_step[2] == True:
            step_type = StepType.LAST
        else:
            step_type = StepType.MID
        return self._augment_time_step(time_step, action, step_type= step_type)

    def _augment_time_step(self, time_step, action=None, step_type = None):
        if  step_type == StepType.FIRST:
            discount = 1.0
            landing_info = False#time_step[3]["landing"]
            position_error = [0.0, 0.0]
        elif time_step[3]["episode end flag"] == True:
            discount = 0.0
            landing_info = time_step[3]["landing"]
            position_error = [time_step[3]['x error'], time_step[3]['y error']]
        else:
            discount = 1.0
            landing_info = time_step[3]["landing"]
            position_error = [time_step[3]['x error'], time_step[3]['y error']]

        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        if step_type == StepType.FIRST:
            reward = 0.0
        else:
            reward = time_step[1]

        #add landing info

        return ExtendedTimeStep(observation=time_step[0],
                                step_type=step_type,
                                action=action,
                                reward= reward,#time_step.reward or 0.0,
                                discount=discount,
                                landing_info = landing_info,
                                drone_state = time_step[-1],
                                position_error = position_error)#time_step.discount or 1.0)


class MultiRewardExtendedTimeStepWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self):
        time_step = self.env.reset()
        return self._augment_time_step(time_step, step_type = StepType.FIRST)

    def step(self, action):
        time_step = self.env.step(action)
        if time_step[2] == True:
            step_type = StepType.LAST
        else:
            step_type = StepType.MID
        return self._augment_time_step(time_step, action, step_type= step_type)

    def _augment_time_step(self, time_step, action=None, step_type = None):
        if  step_type == StepType.FIRST:
            discount = 1.0
            landing_info = False#time_step[3]["landing"]
            position_error = [0.0, 0.0]
        elif time_step[3]["episode end flag"] == True:
            discount = 0.0
            landing_info = time_step[3]["landing"]
            position_error = [time_step[3]['x error'], time_step[3]['y error']]
        else:
            discount = 1.0
            landing_info = time_step[3]["landing"]
            position_error = [time_step[3]['x error'], time_step[3]['y error']]

        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        if step_type == StepType.FIRST:
            reward = np.zeros((3,), dtype=np.float32)
        else:
            reward = time_step[1]

        return ExtendedTimeStep(observation=time_step[0],
                                step_type=step_type,
                                action=action,
                                reward= reward,
                                discount=discount,
                                landing_info = landing_info,
                                drone_state = time_step[-1],
                                position_error = position_error)


def make(name, frame_stack, action_repeat, seed):
    env = gym.make(name) 
    env = FrameStackWrapper(env, frame_stack)
    env = ExtendedTimeStepWrapper(env)
    return env

def make_with_gimbal(name, frame_stack, action_repeat, seed, env_config=None):
    for _ in range(5):
        print(name)
    env = gym.make(name, **env_config) if env_config is not None else gym.make(name)
    env = FrameStackWrapperWithGimbalState(env, frame_stack)
    env = ExtendedTimeStepWrapper(env)
    return env

def make_with_gimbal_oracle(name, frame_stack, action_repeat, seed, env_config=None):
    env = gym.make(name, **env_config) if env_config is not None else gym.make(name)
    env = FrameStackWrapperWithGimbalOracle(env, frame_stack)
    env = ExtendedTimeStepWrapper(env)
    return env

def make_toddler_train(name, frame_stack, action_repeat, seed):
    env = gym.make(name)
    env = FrameStackWrapper(env, frame_stack)
    env = ExtendedTimeStepWrapper(env)
    return env

def make_reflective_train(name, frame_stack, action_repeat, seed):
    env = gym.make(name)
    env = FrameStackWrapper(env, frame_stack)
    env = MultiRewardExtendedTimeStepWrapper(env)
    return env
