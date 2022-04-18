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
    
    # FOR 1: K
    for _ in range(10):
        #TRAIN RL AGENT AND THE SPM MODEL AGAINST THE ENVIORNMENT PARAMETERS
        model.learn(total_timesteps=1e2, callback=callbacks)
        sim_param_model.spm_train()
        sim_param_model.update_params(real_environment)
        # EVALUATE PARAMS AGAINST REAL WORLD OBSERVATIONS
        # evaluate(real_env, sim_env, agent, sim_param_model, video_real, video_sim,
        #              args.num_eval_episodes, L, step, args, use_policy, update_distribution, training_phase)

        # collect rollouts using PPO model running on real world env, i.e., collect real world trajectories.
        # potentially 3 episodes 

        # for each episode, we call update_sim_params to figure which parameters (129) should be updated.
        # values of 0 from forward_classifier, should be decremented, values of 0.5 leave it alone, values of 1 increment

        # pred_sim_params is a list of 3 items, but we take mean so it becomes a list of len = number of parameters (129)
        
        # Loop through all parameters,
        # figure out current simulation mean value
        # figure out new updated mean based on 
        #     scale_factor = max(prev_mean, args.alpha)
        #     new_update = - alpha * (np.mean(pred_mean) - 0.5) * scale_factor
        #     curr_update = args.momentum * args.update[i] + (1 - args.momentum) * new_update
        #     new_mean = prev_mean + curr_update
        # new_mean = max(new_mean, 1e-3)
        # sim_env.dr[param] = new_mean


        

    # model.learn(total_timesteps=2e6, callback=callbacks, eval_env=real_environment)








if __name__ == "__main__":
  main_1()

