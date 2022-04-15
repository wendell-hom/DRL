import time
import argparse
import os, inspect


currentdir = os.getcwd() #os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import os
import numpy as np
import torch

from env import MinitaurExtendedEnv
from pybullet_envs.minitaur.envs import minitaur_gym_env
# from pybullet_envs.minitaur.envs import minitaur_gym_env
from env_randomizer import MinitaurEnvRandomizer, StaticEnvRandomizer, SimPramRandomizer
from spm.SPM import SimParamModel

from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3 import PPO
# import spm.SPM as SPM
log_path = './logs'
batch_size = 128

def get_frame_captures(env, actions):
    obs = []
    frames = []
    env.reset()
    for act in actions:
        next_obs, _, _, _ = env.step(act)
        obs.append(next_obs)
        frames.append(env.render())
    return np.asarray(frames)

def main_1():

    static_env_randomizer = StaticEnvRandomizer()
    dynamic_env_randomizer = MinitaurEnvRandomizer()
    

    # real_enviornment = MinitaurExtendedEnv(
    #       history_length=1,
    #       history_include_actions=True,
    #       history_include_states=True,
    #       include_state_difference=True,
    #       include_second_state_difference=True,
    #       include_base_position=True,
    #       include_leg_model=True,
    #       never_terminate=True,
    #       action_scale=0.5,
    #       urdf_version=minitaur_gym_env.DERPY_V0_URDF_VERSION,
    #       render=False,
    #       env_randomizer=static_env_randomizer,
    # )

    sim_environment = MinitaurExtendedEnv(
         history_length=1,
          history_include_actions=True,
          history_include_states=True,
          include_state_difference=True,
          include_second_state_difference=True,
          include_base_position=True,
          include_leg_model=True,
          never_terminate=False,
          action_scale=0.5,
          urdf_version=minitaur_gym_env.DERPY_V0_URDF_VERSION,
          render=False,
          
    )




    # #warmup
    # warmup_buffer = []
    # # SPM.SimParamModel(None, None)
    # sim_enviornment.add_env_randomizer(dynamic_env_randomizer)

    # for i in range(1):
    #     param_dict = dynamic_env_randomizer.randomize_env(sim_enviornment)

    #     for param_model, param_obj in param_dict.items():
    #         print(param_model, param_obj)
    #     param_shape = {}
    #     param_elem = []
    #     for param_name, param_obj in param_dict.items():
    #         if isinstance(param_obj, np.ndarray):
    #             param_shape[param_name] = param_obj.shape
    #             param_dict[param_name] = param_obj.flatten()
    #             param_elem += param_obj.flatten().tolist()
    #         else:
    #             param_shape[param_name] = 1
    #             param_elem.append(param_obj)
        
    #     action_space = sim_enviornment.action_space.sample()
    #     next_obs_space, reward, done, info = sim_enviornment.step(action_space)
    #     env_img = sim_enviornment.render()



    #     new_param_dict = {}
    #     counter = 0
    #     for param_name, param_s in param_shape.items():
    #         end_counter = counter + np.prod(param_s)
    #         new_param_dict[param_name] = param_elem[counter: end_counter]
    #         counter = end_counter

    #     # SPM goes here

    #     # print(action_space.shape, env_action_obs.shape, env_img.shape, param_elem) 


    # Where to save the model
    output_dir = "models_SPM_t1"
    save_path = os.path.join(output_dir, "model")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    callbacks = []
    callbacks.append(CheckpointCallback(save_freq=int(1e5), save_path=save_path, name_prefix='SPM_t1'))

    model = PPO("MlpPolicy", sim_environment, verbose=1, n_steps=256, 
                batch_size=batch_size, gae_lambda=0.95, gamma=0.99, n_epochs=1, ent_coef=0.0, 
                learning_rate=2.5e-4, clip_range=0.2, device='cuda')

    sim_param_model = SimPramRandomizer(sim_environment, model, batch_size)
    sim_environment.add_env_randomizer(sim_param_model)
    
    # FOR 1: K
    for _ in range(10):
        model.learn(total_timesteps=1e2, callback=callbacks)
        sim_param_model.spm_train()

    # model.learn(total_timesteps=2e6, callback=callbacks, eval_env=real_environment)



if __name__ == "__main__":
  main_1()
