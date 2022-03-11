from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import functools
import random
import abc


import os, inspect
currentdir = os.getcwd() #os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
parentdir = os.path.dirname(os.path.dirname(parentdir))
os.sys.path.insert(0, parentdir)

import numpy as np


PARAM_RANGE = {
    # The following ranges are in percentage. e.g. 0.8 means 80%.
    "mass": [0.8, 1.2],
    "inertia": [0.5, 1.5],
    "motor strength": [0.8, 1.2],

    # The following ranges are the physical values, in SI unit.
    "motor friction": [0, 0.05],  # Viscous damping (Nm s/rad).
    "control step": [0.003, 0.02],  # Time inteval (s).
    "latency": [0.0, 0.04],  # Time inteval (s).
    "lateral friction": [0.5, 1.25],  # Friction coefficient (dimensionless).
    "battery": [14.0, 16.8],  # Voltage (V).
    "joint friction": [0, 0.05],  # Coulomb friction torque (Nm).
}


def all_params():
  """Randomize all the physical parameters."""
  return PARAM_RANGE


class EnvRandomizerBase(object):
    __metaclass__ = abc.ABCMeta
    @abc.abstractmethod
    def randomize_env(self, env):
        pass

    def randomize_step(self, env):
        pass


class MinitaurEnvRandomizer(EnvRandomizerBase):
        
    def __init__(self, config=None):
        if config is None:
            config = "all_params"
        try:
            config = all_params
        except AttributeError:
            raise ValueError("Config {} is not found.".format(config))
        self._randomization_param_dict = config()
    
    def randomize_env(self, env):
        """Randomize various physical properties of the environment.
        It randomizes the physical parameters according to the input configuration.
        Args:
        env: A minitaur gym environment.
        """
        print("Randomizing the ENV")
        self._randomization_function_dict = self._build_randomization_function_dict(env)
        for param_name, random_range in iter(self._randomization_param_dict.items()):
            self._randomization_function_dict[param_name](lower_bound=random_range[0],
                                                        upper_bound=random_range[1])

    def _build_randomization_function_dict(self, env):
        func_dict = {}
        func_dict["mass"] = functools.partial(self._randomize_masses, minitaur=env.minitaur)
        func_dict["inertia"] = functools.partial(self._randomize_inertia, minitaur=env.minitaur)
        func_dict["latency"] = functools.partial(self._randomize_latency, minitaur=env.minitaur)
        func_dict["joint friction"] = functools.partial(self._randomize_joint_friction,
                                                        minitaur=env.minitaur)
        func_dict["motor friction"] = functools.partial(self._randomize_motor_friction,
                                                        minitaur=env.minitaur)
        func_dict["restitution"] = functools.partial(self._randomize_contact_restitution,
                                                    minitaur=env.minitaur)
        func_dict["lateral friction"] = functools.partial(self._randomize_contact_friction,
                                                        minitaur=env.minitaur)
        func_dict["battery"] = functools.partial(self._randomize_battery_level, minitaur=env.minitaur)
        func_dict["motor strength"] = functools.partial(self._randomize_motor_strength,
                                                        minitaur=env.minitaur)
        # Settinmg control step needs access to the environment.
        func_dict["control step"] = functools.partial(self._randomize_control_step, env=env)
        return func_dict

    def _randomize_control_step(self, env, lower_bound, upper_bound):
        randomized_control_step = random.uniform(lower_bound, upper_bound)
        env.set_time_step(randomized_control_step)
    
    def _randomize_masses(self, minitaur, lower_bound, upper_bound):
        base_mass = minitaur.GetBaseMassesFromURDF()
        random_base_ratio = random.uniform(lower_bound, upper_bound)
        randomized_base_mass = random_base_ratio * np.array(base_mass)
        minitaur.SetBaseMasses(randomized_base_mass)

        leg_masses = minitaur.GetLegMassesFromURDF()
        random_leg_ratio = random.uniform(lower_bound, upper_bound)
        randomized_leg_masses = random_leg_ratio * np.array(leg_masses)
        minitaur.SetLegMasses(randomized_leg_masses)
    
    def _randomize_inertia(self, minitaur, lower_bound, upper_bound):
        base_inertia = minitaur.GetBaseInertiasFromURDF()
        random_base_ratio = random.uniform(lower_bound, upper_bound)
        randomized_base_inertia = random_base_ratio * np.array(base_inertia)
        minitaur.SetBaseInertias(randomized_base_inertia)
        leg_inertia = minitaur.GetLegInertiasFromURDF()
        random_leg_ratio = random.uniform(lower_bound, upper_bound)
        randomized_leg_inertia = random_leg_ratio * np.array(leg_inertia)
        minitaur.SetLegInertias(randomized_leg_inertia)
    
    def _randomize_latency(self, minitaur, lower_bound, upper_bound):
        randomized_latency = random.uniform(lower_bound, upper_bound)
        minitaur.SetControlLatency(randomized_latency)
    
    def _randomize_joint_friction(self, minitaur, lower_bound, upper_bound):
        num_knee_joints = minitaur.GetNumKneeJoints()
        randomized_joint_frictions = np.random.uniform([lower_bound] * num_knee_joints,
                                                    [upper_bound] * num_knee_joints)
        minitaur.SetJointFriction(randomized_joint_frictions)
    
    def _randomize_motor_friction(self, minitaur, lower_bound, upper_bound):
        randomized_motor_damping = random.uniform(lower_bound, upper_bound)
        minitaur.SetMotorViscousDamping(randomized_motor_damping)
    
    def _randomize_contact_restitution(self, minitaur, lower_bound, upper_bound):
        randomized_restitution = random.uniform(lower_bound, upper_bound)
        minitaur.SetFootRestitution(randomized_restitution)
    
    def _randomize_contact_friction(self, minitaur, lower_bound, upper_bound):
        randomized_foot_friction = random.uniform(lower_bound, upper_bound)
        minitaur.SetFootFriction(randomized_foot_friction)
    
    def _randomize_battery_level(self, minitaur, lower_bound, upper_bound):
        randomized_battery_voltage = random.uniform(lower_bound, upper_bound)
        minitaur.SetBatteryVoltage(randomized_battery_voltage)
    
    def _randomize_motor_strength(self, minitaur, lower_bound, upper_bound):
        randomized_motor_strength_ratios = np.random.uniform([lower_bound] * minitaur.num_motors,
                                                            [upper_bound] * minitaur.num_motors)
        minitaur.SetMotorStrengthRatios(randomized_motor_strength_ratios)
    

class StaticEnvRandomizer(MinitaurEnvRandomizer):
    def __init__(self):
        self._randomization_param_dict = all_params()
        self.step = 0
    def randomize_env(self, env):
        if self.step == 0:
            self._randomization_function_dict = self._build_randomization_function_dict(env)
            for param_name, random_range in iter(self._randomization_param_dict.items()):
                self._randomization_function_dict[param_name](lower_bound=random_range[0],
                                                            upper_bound=random_range[1])
            self.step += 1
    def randomize_step(self, env):
        pass

class BayesianEnvRandomizer(MinitaurEnvRandomizer):
    def __init__(self):       
        self._randomization_param_dict = all_params()
    
    def randomize_env(self, env):
        if self.step == 0:
            self._randomization_function_dict = self._build_randomization_function_dict(env)
            for param_name, random_range in iter(self._randomization_param_dict.items()):
                self._randomization_function_dict[param_name](lower_bound=random_range[0],
                                                            upper_bound=random_range[1])

    def randomize_step(self, env):
        pass