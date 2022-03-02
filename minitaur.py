import time

import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import os
import numpy as np

from pybullet_envs.minitaur.envs import minitaur_extended_env
from pybullet_envs.minitaur.envs import minitaur_gym_env
# from pybullet_envs.minitaur.envs import minitaur_gym_env
from env_randomizer import MinitaurEnvRandomizer

from stable_baselines3 import PPO

log_path = './logs'

def main():
    steps = 1000
    episodes = 5

    randomizer = MinitaurEnvRandomizer('all_params')
    # environment = minitaur_gym_env.MinitaurGymEnv(
    #     urdf_version=minitaur_gym_env.DERPY_V0_URDF_VERSION,
    #     render=True,
    #     leg_model_enabled=False,
    #     motor_velocity_limit=np.inf,
    #     pd_control_enabled=True,
    #     accurate_motor_model_enabled=True,
    #     motor_overheat_protection=True,
    #     hard_reset=False,
    #     env_randomizer=randomizer,
    #     log_path=log_path)

    environment = minitaur_extended_env.MinitaurExtendedEnv(
        history_length=1,
        history_include_actions=True,
        history_include_states=False,
        include_state_difference=False,
        include_second_state_difference=False,
        include_base_position=False,
        include_leg_model=False,
        never_terminate=False,
        action_scale=0.5,
        urdf_version=minitaur_gym_env.DERPY_V0_URDF_VERSION,
        render=True,
        # leg_model_enabled=False,
        # motor_velocity_limit=np.inf,
        # pd_control_enabled=True,
        # accurate_motor_model_enabled=True,
        # motor_overheat_protection=True,
        # hard_reset=False,
        #senv_randomizer=randomizer,
    )

	# Training parameters from stable-baselines3-zoo
    # We are not using make_vec_env, so don't have a place for n_envs and normalize
    # I think those two settings are for make_vec_env
    model = PPO("MlpPolicy", environment, verbose=1, n_steps=2048, 
                batch_size=64, gae_lambda=0.95, gamma=0.99, n_epochs=10, ent_coef=0.0, 
                learning_rate=2.5e-4, clip_range=0.2)
    model.learn(total_timesteps=2e6)

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
