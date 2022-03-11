import time
import argparse

import os, inspect
currentdir = os.getcwd() #os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import os
import numpy as np

from pybullet_envs.minitaur.envs import minitaur_extended_env
from pybullet_envs.minitaur.envs import minitaur_gym_env
#from pybullet_envs.bullet import minitaur_gym_env
from pybullet_envs.bullet import minitaur_env_randomizer
from env_randomizer import MinitaurEnvRandomizer

from stable_baselines3 import PPO

log_path = './logs'

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n-timesteps", help="number of timesteps", default=1000, type=int)
    parser.add_argument("-r", "--random_env", help="Randomize environment", default=1, type=int)
    parser.add_argument("-m", "--model", help="Model to load", default="minitaur.zip", type=str)
    parser.add_argument("-R", "--render", help="Render", default=1, type=int)
    parser.add_argument("-e", "--episodes", help="Number of episodes", default=1, type=int)
    args = parser.parse_args()

    episodes = args.episodes
    steps = args.n_timesteps
    randomize = args.random_env
    model_file = args.model
    render = args.render


    if randomize:
        randomizer = MinitaurEnvRandomizer('all_params')

        environment = minitaur_extended_env.MinitaurExtendedEnv(
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
          env_randomizer=randomizer,
          render=render,
        )
    else:
        environment = minitaur_extended_env.MinitaurExtendedEnv(
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
        )


    #randomizer = (minitaur_env_randomizer.MinitaurEnvRandomizer())
    # environment = minitaur_gym_env.MinitaurBulletEnv(render=True,
    #                                                leg_model_enabled=False,
    #                                                motor_velocity_limit=np.inf,
    #                                                pd_control_enabled=True,
    #                                                accurate_motor_model_enabled=True,
    #                                                motor_overheat_protection=True,
    #                                                #env_randomizer=randomizer,
    #                                                hard_reset=False)
    model = PPO.load(model_file)


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

    print("Average reward:", np.mean(sum_reward))
    print("Times fallen:", np.sum(fallen))

if __name__ == "__main__":
  main() 
