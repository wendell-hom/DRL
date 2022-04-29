import time
import argparse
import os, inspect
from tqdm import tqdm, trange


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
    total = 0

    for param_name, param_obj in sim_param_model.param_dict.items():
        real_params = static_env_randomizer.param_dict[param_name][0]
        sim_param_means = (param_obj[1] + param_obj[2])/2
        if not np.isscalar(real_params):
            assert(real_params.shape == sim_param_means.shape)
        else:
            assert(np.isscalar(sim_param_means))

        error[param_name] = np.linalg.norm(real_params - sim_param_means)
        total += error[param_name]

    return error, total

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

    model = PPO("MlpPolicy", sim_environment, verbose=1, n_steps=500, 
                batch_size=batch_size, gae_lambda=0.95, gamma=0.99, n_epochs=10, ent_coef=0.0, 
                learning_rate=2.5e-4, clip_range=0.2, device='cuda', tensorboard_log=log_path)

    sim_param_model = SimPramRandomizer(sim_environment, model, batch_size)
    sim_environment.add_env_randomizer(sim_param_model)
    spm_loss = []

    #init spm
    # for _ in range(50):
    #     print(".", end='')
    #     loss = sim_param_model.spm_train()

    print("Done with pre-training")

    spm_loss = []
    # FOR 1: K  
    fp = open("log.txt", "w")

    for _ in range(1000):

        #TRAIN RL AGENT AND THE SPM MODEL AGAINST THE ENVIORNMENT PARAMETERS
        model.learn(total_timesteps=1e5, callback=callbacks)

		# Evaluate in real environment after training
        rewards, falls = sim2sim(model, real_environment)
        print(f"Rewards, falls: {rewards}, {falls}")
        fp.write(f"Rewards, falls: {rewards}, {falls}")

        # Train the SPM model
        with trange(25) as t:
            for i in t:
                loss, acc = sim_param_model.spm_train()
                spm_loss.append(loss)
                dev_loss, dev_acc = sim_param_model.evaluate_spm(real_environment)
                t.set_postfix(acc=acc, loss=loss, dev_loss=dev_loss, dev_acc=dev_acc )
                fp.write(f"Accuracy, loss: {loss}, {acc}, {dev_loss}, {dev_acc}\n")
                print(f"Accuracy, loss: {loss}, {acc}, {dev_loss}, {dev_acc}")

        # Do real world rollouts to update simulation parameters
        for _ in range(3):
            sim_param_model.update_params(real_environment)

        
        # Check difference between mean of simulation parameters vs. parameters of real environment
        distance, total = parameter_error(sim_param_model, static_environment_randomizer)
        print(f"Total error: {total}")
        print(f"Individual error: {distance}")
        fp.write(f"Total error: {total} \n")
        fp.write(f"Individual error: {distance} \n")

    
def sim2sim(model, environment, episodes=10, steps=500):
    rewards = []
    fallen = []

    for _ in range(episodes):
        sum_reward = 0
        has_fallen = 0
        observation = environment.reset()

        for _ in range(steps):
            # Sleep to prevent serial buffer overflow on microcontroller.
            time.sleep(0.002)
            action, _states = model.predict(observation)
            observation, reward, done, _ = environment.step(action)

            # Seems like minitaur always falls during my runs, need to debug to see why  
            if environment.is_fallen():
                has_fallen = 1

            sum_reward += reward
            if done:
                break
        rewards.append(sum_reward)
        fallen.append(has_fallen)

    return np.mean(rewards), np.sum(fallen)





if __name__ == "__main__":
  main_1()

