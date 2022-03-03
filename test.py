import time
import argparse

import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
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
    parser.add_argument("-n", "--n-timesteps", help="number of timesteps", default=5000, type=int)
    parser.add_argument("-r", "--random_env", help="Randomize environment", default=1, type=int)
    parser.add_argument("-m", "--model", help="Model to load", default="minitaur.zip", type=str)
    args = parser.parse_args()

    steps = args.n_timesteps
    randomize = args.random_env
    model_file = args.model


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
        render=True,
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

    sum_reward = 0
    observation = environment.reset()
    for _ in range(steps):
        # Sleep to prevent serial buffer overflow on microcontroller.
        time.sleep(0.002)
        action, _states = model.predict(observation)
        observation, reward, done, _ = environment.step(action)
        sum_reward += reward
        if done:
            break

    print("Total reward:", sum_reward)

if __name__ == "__main__":
  main() 
