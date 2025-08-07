# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path

import hydra
import numpy as np
import torch
from gym_pybullet_drones.utils import specs

import dmc
import utils
from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader
from video import TrainVideoRecorder, VideoRecorder
from pathlib import Path
import time
import random
from PIL import Image
from decomposed_networks import Actor_Decomposed, Encoder_Decomposed

import matplotlib
import matplotlib.pyplot as plt

torch.backends.cudnn.benchmark = True


def make_agent(obs_spec, action_spec, cfg):
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg)


class Workspace:
    def __init__(self, cfg, eval_dir):
        self.work_dir = eval_dir
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.setup()

        self.agent = make_agent(self.eval_env.observation_spec(),
                                self.eval_env.action_spec(),
                                self.cfg.agent)
        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

    def setup(self):
        # create logger
        #self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)
        # create envs
        #self.train_env = dmc.make(self.cfg.task_name, self.cfg.frame_stack,
        #                          self.cfg.action_repeat, self.cfg.seed)
        self.eval_env = dmc.make(self.cfg.task_name, self.cfg.frame_stack,
                                self.cfg.action_repeat, self.cfg.seed)
        #self.eval_env = self.train_env 
        # create replay buffer
        data_specs = (self.eval_env.observation_spec(),
                      self.eval_env.action_spec(),
                      self.eval_env.reward_spec(),
                      self.eval_env.discount_spec())


        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None)
        self.train_video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_train_video else None)


    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def visualise(self):
        os.system('rm -rf /home/user/landing/test_images')
        os.system('mkdir /home/user/landing/test_images')
        step, episode, total_reward = 0, 0, 0
        #eval_until_episode = utils.Until(self.cfg.num_post_eval_episodes)
        eval_until_episode = utils.Until(1)
        Landing_flags = []
        Landing_errors = []
        action_array = []
        velocity_array = []
        uav_vels = []
        uav_poss = []
        gv_vels = []
        gv_poss = []
        while eval_until_episode(episode):
            time_step = self.eval_env.reset()
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            Drone_states_list = []
            #self.video_recorder.init(self.eval_env, enabled=False)
            #indexes = np.load("/home/user/landing/test_images_tSNE_bad/indexes.npy")
            while not time_step.last():
                #print(time_step.observation.shape)
                #print(time_step.observation[1].squeeze().shape)

                #np.save("/home/user/landing/simulation_data_quant/state_{0}_eps_{1}".format(step,episode),time_step.drone_state)
                #np.save("/home/user/landing/simulation_data_quant/image_{0}_eps_{1}".format(step,episode),time_step.observation)
                #observation = np.load("/home/user/landing/experimental_data_new_reward_3_fail/image_{}.npy".format(step)).astype(int)

                #drone_state = np.load("/home/user/landing/experimental_data_new_reward_3_fail/state_{}.npy".format(step))
                #print(indexes[step])
                #observation = np.load("/home/user/landing/experimental_data_after_cropping/image_{}.npy".format(indexes[step])).astype(int)

                #drone_state = np.load("/home/user/landing/experimental_data_after_cropping/state_{}.npy".format(indexes[step]))
                #drone_state[:,3] = 0.0
                #drone_state[:,4] = 0.0
                #drone_state[:,5] = 0.0
                #drone_state[:,6] = 1.0
                
                #im = Image.fromarray(time_step.observation[2].squeeze().astype(float))
                #im = im.convert("L")
                #im.save("/home/user/landing/test_images/drone_view_{0}.png".format(step))
                with torch.no_grad(), utils.eval_mode(self.agent):
                    #print(time_step.observation.max())
                    #print(time_step.observation.min())x

                    action = self.agent.act(time_step.observation,
                                            self.global_step,
                                            time_step.drone_state,
                                            eval_mode=True)
                    velocity = time_step.drone_state[2,0:3].flatten()
                    #print(action)
                    #print(action)
                    #action[0] = 0.0#0.05/(2.91666667/2)
                    #action[1] = 0.0
                    #action[2] = -1.0
                    #print(action)
                    #if step > 80:
                    #    print(action)
                    # if action[2] > 0:
                    #     im = Image.fromarray(observation[2].squeeze().astype(float))
                    #     im = im.convert("L")
                    #     im.save("/home/u`ser/landing/test_images_up_2/drone_view_{0}.png".format(step))
                    # elif action[2] < 0:
                    #     im = Image.fromarray(observation[2].squeeze().astype(float))
                    #     im = im.convert("L")
                    #     im.save("/homeuser/landing/test_images_down_2/drone_view_{0}.png".format(step))
                    #if step < 40:
                    #    print(step)
                    #    action[0] = random.uniform(-0.05,0.05)
                    #    action[1] = random.uniform(-0.05,0.05)
                    #    action[2] = 0.0
                    #curr_time = self.eval_env.get_sim_time()
                    #action_array.append(np.append(action,[curr_time]))
                    #velocity_array.append(np.append(velocity,[curr_time]))
                uav_vel = self.eval_env.get_uav_velocity()
                uav_pos = self.eval_env.get_uav_position()
                gv_pos = self.eval_env.get_vehicle_position()
                gv_vel = self.eval_env.get_vehicle_velocity()
                uav_vels.append(uav_vel.copy())
                uav_poss.append(uav_pos.copy())
                gv_vels.append(gv_vel)
                gv_poss.append(gv_pos)
                action_array_np = np.array(action_array)
                velocity_array_np = np.array(velocity_array)
                #time.sleep(0.1)
                #np.save("/home/user/landing/experimental_data/sim_actions_{0}".format(step),action)
                time_step = self.eval_env.step(action)
                self.video_recorder.record(self.eval_env)
                total_reward += time_step.reward

                #print('reward is {0}'.format(time_step.reward))
                step += 1
            #np.save('/home/user/landing/sim_controller_test/action.npy', action_array)
            #np.save('/home/user/landing/sim_controller_test/velocity.npy', velocity_array)
            #print(velocity_array)
            Landing_flags.append(time_step.landing_info)
            Landing_errors.append(time_step.position_error)
            #exit()
            print(episode)
            episode += 1
            #time.sleep(120)
            self.video_recorder.save(f'{self.global_frame}_eval.mp4')
            #exit()
        Landing_flags = np.array(Landing_flags)
        print('Succesful landings')
        successes = np.sum(Landing_flags)/episode
        print(successes)
        print('episode_reward')
        print(total_reward / episode)
        print('positional error')
        Landing_errors = np.array(Landing_errors)
        Landing_distances = np.linalg.norm(Landing_errors, axis = 1)
        average_distance_error = np.sum(Landing_distances)/episode
        print(average_distance_error)
        #plotters
        uav_vels = np.array(uav_vels)
        uav_poss = np.array(uav_poss)
        gv_poss = np.array(gv_poss)
        gv_vels = np.array(gv_vels)
        #this is done to account for initial offset of vehicles
        uav_poss[:,0] = uav_poss[:,0] - gv_poss[0,0]
        uav_poss[:,1] = uav_poss[:,1] - gv_poss[0,1]
        gv_poss[:,0] = gv_poss[:,0]  - gv_poss[0,0]
        gv_poss[:,1] = gv_poss[:,1]  - gv_poss[0,1]
        positions_error = uav_poss - gv_poss
        velocities_error = uav_vels - gv_vels
        plt.figure(1)
        #plt.plot(uav_vels[:,0],label = "x-velocity",linewidth=4.0)
        #plt.plot(uav_vels[:,1],label = "y-velocity",linewidth=4.0)
        plt.plot(uav_vels[:,2],label = "z-velocity",linewidth=4.0)
        # naming the x axis
        plt.xlabel('steps')
        # naming the y axis
        plt.ylabel('vertical velocity (m/s)')
        #plt.legend()
        plt.savefig('UAV_vel.png', bbox_inches='tight')
        plt.figure(2)
        plt.plot(positions_error[:,0],label = "x-position",linewidth=4.0, linestyle = ':')
        plt.plot(positions_error[:,1],label = "y-position",linewidth=4.0, linestyle = '--')
        plt.plot(positions_error[:,2],label = "z-position",linewidth=4.0)
        plt.xlabel('steps')
        # naming the y axis
        plt.ylabel('positions error (m)')
        plt.legend(fontsize = 16)
        #plot 3d position of UGV and UAV
        plt.savefig('UAV_pos_error_2D.png', bbox_inches='tight')
        fig = plt.figure()
        ax = plt.axes(projection ='3d')
        ax.plot3D(uav_poss[:,0], uav_poss[:,1], uav_poss[:,2], 'green', label = "UAS trajectory",linewidth=4.0)
        ax.plot3D(gv_poss[:,0], gv_poss[:,1], gv_poss[:,2], 'red', label = "Landing platform trajectory",linewidth=4.0, linestyle = '--')
        ax.set_xlabel('x-position (m)')
        # naming the y axis
        ax.set_ylabel('y-position (m)')
        ax.set_zlabel('z-position (m)')
        ax.legend(fontsize = 16)
        plt.savefig('UAV_pos_3D.png', bbox_inches='tight')
        #plot 2d position
        plt.figure(4)
        plt.plot(uav_poss[:,0], uav_poss[:,1], 'green',label = "UAS trajectory",linewidth=4.0)
        plt.plot(gv_poss[:,0], gv_poss[:,1], 'red', label = "Landing platform trajectory",linewidth=4.0, linestyle = '--')
        plt.xlabel('x-position (m)')
        # naming the y axis
        plt.ylabel('y-position (m)')
        plt.legend(fontsize = 16)
        plt.savefig('UAV_pos_2D.png', bbox_inches='tight')
        print('plotter done')



    def load_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        for k, v in payload.items():
            print(k)
            print(v)
            self.__dict__[k] = v
        #self.agent.encoder = Encoder_Decomposed([3])


@hydra.main(config_path='cfgs', config_name='config')
def main(cfg):
    from Visualise import Workspace as W

    eval_dir = '/home/user/landing/exp_local/2023.07.13/031012_task=landing-aviary-v0/'
    eval_dir = Path(eval_dir)
    root_dir = eval_dir
    workspace = W(cfg, eval_dir)
    snapshot = root_dir / 'snapshot.pt'
    print(f'resuming: {snapshot}')
    workspace.load_snapshot()
    workspace.visualise()


if __name__ == '__main__':
    main()