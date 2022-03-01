import time

import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import os
import numpy as np
import tensorflow as tf
from pybullet_envs.minitaur.envs import minitaur_extended_env
from pybullet_envs.minitaur.envs import minitaur_gym_env
# from pybullet_envs.minitaur.envs import minitaur_gym_env
from pybullet_envs.minitaur.envs.env_randomizers import minitaur_env_randomizer_from_config 
from pybullet_envs.minitaur.envs.env_randomizers import minitaur_env_randomizer_config
from stable_baselines3 import PPO

log_path = './logs'

def main():
  steps = 1000
  episodes = 5

  randomizer = minitaur_env_randomizer_from_config.MinitaurEnvRandomizerFromConfig('all_params')
  environment = minitaur_gym_env.MinitaurGymEnv(
      urdf_version=minitaur_gym_env.DERPY_V0_URDF_VERSION,
      render=True,
      leg_model_enabled=False,
      motor_velocity_limit=np.inf,
      pd_control_enabled=True,
      accurate_motor_model_enabled=True,
      motor_overheat_protection=True,
      hard_reset=False,
      env_randomizer=randomizer,
      log_path=log_path)

  # environment = minitaur_gym_env.MinitaurGymEnv(
  #     urdf_version=minitaur_gym_env.DERPY_V0_URDF_VERSION,
  #     render=True,
  #     num_steps_to_log=steps,
  #     pd_latency=0.002,
  #     control_latency=0.02,
  #     remove_default_joint_damping=True,
  #     on_rack=False,
  #     env_randomizer=randomizer,
  #     log_path=log_path)
  # np.random.seed(100)
  model = PPO("MlpPolicy", environment, verbose=1)

  observation = environment.reset()
  print(observation)
  for i in range(episodes):
    sum_reward = 0
    observation = environment.reset()
    for _ in range(steps):
      # Sleep to prevent serial buffer overflow on microcontroller.
      time.sleep(0.002)
      action = environment.action_space.sample()
      observation, reward, done, _ = environment.step(action)
      sum_reward += reward
      if done:
        break
 


if __name__ == "__main__":
  main() 
