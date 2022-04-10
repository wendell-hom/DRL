import time

import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import os
import numpy as np

from env import MinitaurExtendedEnv
from pybullet_envs.minitaur.envs import minitaur_gym_env
# from pybullet_envs.minitaur.envs import minitaur_gym_env
from env_randomizer import MinitaurEnvRandomizer

from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3 import PPO

log_path = './logs'

def main():
    randomizer = MinitaurEnvRandomizer('all_params')
  
    environment = MinitaurExtendedEnv(
        history_length=1,
        history_include_actions=True,
        history_include_states=False,
        include_state_difference=False,
        include_second_state_difference=False,
        include_base_position=False,
        include_leg_model=False,
        never_terminate=False,
        action_scale=0.5,
        env_randomizer=randomizer,
        urdf_version=minitaur_gym_env.DERPY_V0_URDF_VERSION,
     #   render=False,
    )

    # Where to save the model
    output_dir = "models"

    save_path = os.path.join(output_dir, "model.zip")
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    callbacks = []
    callbacks.append(CheckpointCallback(save_freq=1e5, save_path=output_dir,
                                          name_prefix='model'))

	# Training parameters from stable-baselines3-zoo
    # We are not using make_vec_env, so don't have a place for n_envs and normalize
    # I think those two settings are for make_vec_env
    model = PPO("MlpPolicy", environment, verbose=1, n_steps=2048, 
                batch_size=64, gae_lambda=0.95, gamma=0.99, n_epochs=10, ent_coef=0.0, 
                learning_rate=2.5e-4, clip_range=0.2)
    model.learn(total_timesteps=2e6, callback=callbacks)

    


if __name__ == "__main__":
  main() 
