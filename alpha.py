import time
import argparse
import os, inspect
currentdir = os.getcwd() #os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import os
import numpy as np

from env import MinitaurExtendedEnv
from pybullet_envs.minitaur.envs import minitaur_gym_env
# from pybullet_envs.minitaur.envs import minitaur_gym_env
from env_randomizer import MinitaurEnvRandomizer, StaticEnvRandomizer

from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3 import PPO

log_path = './logs'

def main_1():
    steps = 1000
    episodes = 5

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n-timesteps", help="number of timesteps", default=1000, type=int)
    parser.add_argument("-R", "--render", help="Render", default=0, type=int)
    parser.add_argument("-e", "--episodes", help="Number of episodes", default=5, type=int)
    args = parser.parse_args()

    episodes = args.episodes
    steps = args.n_timesteps
    render = args.render
    static_env_randomizer = StaticEnvRandomizer()
    dynamic_env_randomizer = MinitaurEnvRandomizer()

    real_environment = MinitaurExtendedEnv(
          history_length=1,
          history_include_actions=True,
          history_include_states=False,
          include_state_difference=False,
          include_second_state_difference=False,
          include_base_position=False,
          include_leg_model=False,
          never_terminate=True,
          action_scale=0.5,
          urdf_version=minitaur_gym_env.DERPY_V0_URDF_VERSION,
          render=render,
          env_randomizer=static_env_randomizer,
    )

    sim_enviornment = MinitaurExtendedEnv(
         history_length=1,
          history_include_actions=True,
          history_include_states=False,
          include_state_difference=False,
          include_second_state_difference=False,
          include_base_position=False,
          include_leg_model=False,
          never_terminate=True,
          action_scale=0.5,
          urdf_version=minitaur_gym_env.DERPY_V0_URDF_VERSION,
          render=render,
          env_randomizer=dynamic_env_randomizer,
    )

    # Where to save the model
    output_dir = "models_oscar_t1"

    save_path = os.path.join(output_dir, "model.zip")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    callbacks = []
    callbacks.append(CheckpointCallback(save_freq=int(1e5), save_path=save_path, name_prefix='oscar_t1'))
	
    model = PPO("MlpPolicy", sim_enviornment, verbose=1, n_steps=2048,
                batch_size=256, gae_lambda=0.95, gamma=0.99, n_epochs=10, ent_coef=0.0,
                learning_rate=2.5e-4, clip_range=0.2, device='cuda')
    model.learn(total_timesteps=int(2e6), callback=callbacks, eval_env=real_environment)
  


if __name__ == "__main__":
  main_1()
