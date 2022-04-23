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



def parameter_error(sim_param_model, static_env_randomizer):

    error = dict()

    for param_name, param_obj in sim_param_model.param_dict.items():
        real_params = static_env_randomizer.param_dict[param_name][0]
        sim_param_means = (param_obj[1] + param_obj[2])/2
        if not np.isscalar(real_params):
            assert(real_params.shape == sim_param_means.shape)
        else:
            assert(np.isscalar(sim_param_means))

        error[param_name] = np.linalg.norm(real_params - sim_param_means)

    return error

def main_1():

    static_environment_randomizer = StaticEnvRandomizer()
    real_environment = MinitaurExtendedEnv(
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
          env_randomizer=static_environment_randomizer,
         render=False,
    )
    

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
    spm_loss = []

    #init spm
    for _ in range(20):
        print(".", end='')
        sim_param_model.randomize_env(sim_environment)
        actions = [sim_environment.action_space.sample() for _ in range(256)]
        loss = sim_param_model.spm_train(actions)
        spm_loss.append(loss)

    print("Done with pre-training")

    spm_loss = []
    # FOR 1: K  
    fp = open("error.txt", "w")
    for _ in range(100):
        #TRAIN RL AGENT AND THE SPM MODEL AGAINST THE ENVIORNMENT PARAMETERS
        model.learn(total_timesteps=1e2, callback=callbacks)
        spm_loss += [sim_param_model.spm_train(model.rollout_buffer.actions)]
        sim_param_model.update_params(real_environment)

        # Check difference between mean of simulation parameters vs. parameters of real environment
        distance = parameter_error(sim_param_model, static_environment_randomizer)
        print(f"Distance: {distance}")
        fp.write(f"Distance: {distance} \n")

    







if __name__ == "__main__":
  main_1()

