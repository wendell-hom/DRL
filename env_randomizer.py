from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import functools
import random
import abc
import os, inspect
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import torch

from spm.SPM import SimParamModel
currentdir = os.getcwd() #os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
parentdir = os.path.dirname(os.path.dirname(parentdir))
os.sys.path.insert(0, parentdir)

import numpy as np


PARAM_RANGE = {
    # The following ranges are in percentage. e.g. 0.8 means 80%.
    "mass": [0.8, 1.2],
    'leg mass': [0.8, 1.2],
    "inertia": [0.5, 1.5],
    'leg inertia': [0.5, 1.5],
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
        self.randomization_param_dict = config()
    
    def randomize_env(self, env):
        """Randomize various physical properties of the environment.
        It randomizes the physical parameters according to the input configuration.
        Args:
        env: A minitaur gym environment.
        """
        param_dict = {}
        self._randomization_function_dict = self._build_randomization_function_dict(env)
        for param_name, random_range in iter(self.randomization_param_dict.items()):
            param_dict[param_name] = self._randomization_function_dict[param_name](lower_bound=random_range[0],
                                                        upper_bound=random_range[1])
        return param_dict



    def _build_randomization_function_dict(self, env):
        func_dict = {}
        func_dict["mass"] = functools.partial(self._randomize_masses, minitaur=env.minitaur)
        func_dict["leg mass"] = functools.partial(self._randomize_leg_mass, minitaur=env.minitaur)
        func_dict["inertia"] = functools.partial(self._randomize_inertia, minitaur=env.minitaur)
        func_dict['leg inertia'] = functools.partial(self._randomize_leg_inertia, minitaur=env.minitaur)

        func_dict["latency"] = functools.partial(self._randomize_latency, minitaur=env.minitaur)
        func_dict["joint friction"] = functools.partial(self._randomize_joint_friction, minitaur=env.minitaur)
        func_dict["motor friction"] = functools.partial(self._randomize_motor_friction, minitaur=env.minitaur)
        func_dict["restitution"] = functools.partial(self._randomize_contact_restitution, minitaur=env.minitaur)
        func_dict["lateral friction"] = functools.partial(self._randomize_contact_friction, minitaur=env.minitaur)
        func_dict["battery"] = functools.partial(self._randomize_battery_level, minitaur=env.minitaur)
        func_dict["motor strength"] = functools.partial(self._randomize_motor_strength, minitaur=env.minitaur)
        # Settinmg control step needs access to the environment.
        func_dict["control step"] = functools.partial(self._randomize_control_step, env=env)
        return func_dict

    def _randomize_control_step(self, env, lower_bound, upper_bound):
        randomized_control_step = random.uniform(lower_bound, upper_bound)
        env.set_time_step(randomized_control_step)
        return randomized_control_step

    def _randomize_masses(self, minitaur, lower_bound, upper_bound):
        base_mass = minitaur.GetBaseMassesFromURDF()
        random_base_ratio = random.uniform(lower_bound, upper_bound)
        randomized_base_mass = random_base_ratio * np.array(base_mass)
        minitaur.SetBaseMasses(randomized_base_mass)
        return randomized_base_mass
    
    def _randomize_leg_mass(self, minitaur, lower_bound, upper_bound):
        leg_masses = minitaur.GetLegMassesFromURDF()
        random_leg_ratio = random.uniform(lower_bound, upper_bound)
        randomized_leg_masses = random_leg_ratio * np.array(leg_masses)
        minitaur.SetLegMasses(randomized_leg_masses)
        return randomized_leg_masses 

    def _randomize_inertia(self, minitaur, lower_bound, upper_bound):
        base_inertia = minitaur.GetBaseInertiasFromURDF()
        random_base_ratio = random.uniform(lower_bound, upper_bound)
        randomized_base_inertia = random_base_ratio * np.array(base_inertia)
        minitaur.SetBaseInertias(randomized_base_inertia)       
        return randomized_base_inertia
    
    def _randomize_leg_inertia(self, minitaur, lower_bound, upper_bound):
        leg_inertia = minitaur.GetLegInertiasFromURDF()
        random_leg_ratio = random.uniform(lower_bound, upper_bound)
        randomized_leg_inertia = random_leg_ratio * np.array(leg_inertia)
        minitaur.SetLegInertias(randomized_leg_inertia)
        return randomized_leg_inertia
    
    def _randomize_latency(self, minitaur, lower_bound, upper_bound):
        randomized_latency = random.uniform(lower_bound, upper_bound)
        minitaur.SetControlLatency(randomized_latency)
        return randomized_latency
    
    def _randomize_joint_friction(self, minitaur, lower_bound, upper_bound):
        num_knee_joints = minitaur.GetNumKneeJoints()
        randomized_joint_frictions = np.random.uniform([lower_bound] * num_knee_joints,
                                                    [upper_bound] * num_knee_joints)
        minitaur.SetJointFriction(randomized_joint_frictions)
        return randomized_joint_frictions
    
    def _randomize_motor_friction(self, minitaur, lower_bound, upper_bound):
        randomized_motor_damping = random.uniform(lower_bound, upper_bound)
        minitaur.SetMotorViscousDamping(randomized_motor_damping)
        return randomized_motor_damping
    
    def _randomize_contact_restitution(self, minitaur, lower_bound, upper_bound):
        randomized_restitution = random.uniform(lower_bound, upper_bound)
        minitaur.SetFootRestitution(randomized_restitution)
        return randomized_restitution

    def _randomize_contact_friction(self, minitaur, lower_bound, upper_bound):
        randomized_foot_friction = random.uniform(lower_bound, upper_bound)
        minitaur.SetFootFriction(randomized_foot_friction)
        return randomized_foot_friction
    
    def _randomize_battery_level(self, minitaur, lower_bound, upper_bound):
        randomized_battery_voltage = random.uniform(lower_bound, upper_bound)
        minitaur.SetBatteryVoltage(randomized_battery_voltage)
        return randomized_battery_voltage
    
    def _randomize_motor_strength(self, minitaur, lower_bound, upper_bound):
        randomized_motor_strength_ratios = np.random.uniform([lower_bound] * minitaur.num_motors,
                                                            [upper_bound] * minitaur.num_motors)
        minitaur.SetMotorStrengthRatios(randomized_motor_strength_ratios)
        return randomized_motor_strength_ratios

class StaticEnvRandomizer(MinitaurEnvRandomizer):
    def __init__(self):
        self.randomization_param_dict = all_params()
        self.step = 0
    def randomize_env(self, env):
        if self.step == 0:
            self._randomization_function_dict = self._build_randomization_function_dict(env)
            for param_name, random_range in iter(self.randomization_param_dict.items()):
                self._randomization_function_dict[param_name](lower_bound=random_range[0],
                                                            upper_bound=random_range[1])
            self.step += 1
    def randomize_step(self, env):
        pass

def crop_center(img,cropx,cropy):
    y,x,c = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx, :]
    
def get_frame_captures(env, actions):
    obs = []
    frames = []
    env.reset()
    for act in actions:
        next_obs, _, _, _ = env.step(act)
        obs.append(next_obs)
        frames.append(env.render())
        # imgplot = plt.imshow(frames[-1])
        # plt.show()
    frames = np.asarray(frames)
    frames = np.transpose(frames, (0, 3, 1, 2))
    return frames, obs


class SimPramRandomizer(MinitaurEnvRandomizer):
    def __init__(self, env, model, batch_size):
        self.randomization_param_dict = all_params()
        self.randomize_env(env)
        self.env = env
        self.model = model
        param_elems, param_shape = self.get_params()
        frame, obs = get_frame_captures(env, env.action_space.sample())
       
        
        self.SPM = SimParamModel(shape=len(param_elems), action_space=env.action_space, state_dim=obs[0].size, layers=2, units=400, device=torch.device('cuda'), obs_shape=frame[0].shape, 
            encoder_type='pixel', encoder_feature_dim=50, encoder_num_layers=4, encoder_num_filters=32, batch_size=batch_size )

    def spm_train(self):
        param_elems, param_shape = self.get_params()
        actions = self.model.rollout_buffer.actions
        frames, obs = get_frame_captures(self.env, actions)

        x = []
        for i in range(len(frames)):
            x.append([frames[i], obs[i], actions[i]])

        self.SPM.update(x, param_elems, self.distribution_mean)

        with torch.no_grad():
            pred = self.SPM.forward_classifier([x], param_elems).cpu().numpy()
            mask = (pred > 0.3) & (pred < 1-0.3)
            preds = np.round(pred)
            confidence_pred = np.mean(preds, axis=0)
        print(confidence_pred)

    def randomize_env(self, env):    
        self.param_dict = super(SimPramRandomizer, self).randomize_env(env)

    def randomize_step(self, env):
        pass

    @property
    def distribution_mean(self):
        param_mean_elems = {}
        for param_name, elem in self.randomization_param_dict.items():
            param_mean_elems[param_name] = (elem[0] + elem[1]) / 2
        return param_mean_elems

    def set_param_distribution(self, elem_classification):
        param_mean_elems = self.distribution_mean
        for i, (param_name, elem) in enumerate(self.randomization_param_dict.items()):
            idx = elem_classification[i]
            self.randomization_param_dict[param_name][idx] = param_mean_elems[param_name]
        return param_mean_elems

    def get_params(self):
        param_shape = {}
        param_elem = []
        for param_name, param_obj in self.param_dict.items():
            if isinstance(param_obj, np.ndarray):
                param_shape[param_name] = param_obj.shape
                self.param_dict[param_name] = param_obj.flatten()
                param_elem += param_obj.flatten().tolist()
            else:
                param_shape[param_name] = 1
                param_elem.append(param_obj)

        return param_elem, param_shape
