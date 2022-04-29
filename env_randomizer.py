from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import functools
import random
import abc
import os, inspect
from collections import OrderedDict
from shutil import ExecError
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

import torch
import torch.nn as nn

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
    "motor friction": [0.0, 0.05],  # Viscous damping (Nm s/rad).
    "control step": [0.003, 0.02],  # Time inteval (s).
    "latency": [0.0, 0.04],  # Time inteval (s).
    "lateral friction": [0.5, 1.25],  # Friction coefficient (dimensionless).
    "battery": [14.0, 16.8],  # Voltage (V).
    "joint friction": [0.0, 0.05],  # Coulomb friction torque (Nm).
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
        self.param_dict = {}
    
    def randomize_env(self, env):
        """Randomize various physical properties of the environment.
        It randomizes the physical parameters according to the input configuration.
        Args:
        env: A minitaur gym environment.
        """
    
        self._randomization_function_dict = self._build_randomization_function_dict(env)
        for param_name, random_range in iter(self.randomization_param_dict.items()):
            self.param_dict[param_name] = self._randomization_function_dict[param_name](lower_bound=random_range[0], 
                                                  upper_bound=random_range[1])
        
        #return param_dict
    @property
    def distribution_mean(self):
        param_mean_elems = []
        for param_name, param_obj in self.param_dict.items():
            _, lb, ub = param_obj
            mean = (lb + ub)/2

            if np.isscalar(mean):
                param_mean_elems.append(mean)
            else:
                param_mean_elems.extend( mean.flatten() )

        return np.array(param_mean_elems)


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
        if "control_step" in self.param_dict:
            _, lb, ub = self.param_dict["control_step"]
            randomized_control_step = np.random.uniform(lb, ub)
        else:    
            randomized_control_step = np.random.uniform(lower_bound, upper_bound)
            lb, ub = lower_bound, upper_bound
        
        env.set_time_step(randomized_control_step)
        return randomized_control_step, lb, ub
    
    def _randomize_masses(self, minitaur, lower_bound, upper_bound):
        if "mass" in self.param_dict:
            _, lb, ub = self.param_dict["mass"]
            randomized_base_mass = np.random.uniform(lb, ub)
        else:
            base_mass = minitaur.GetBaseMassesFromURDF()
            random_base_ratio = np.random.uniform(lower_bound, upper_bound)
            randomized_base_mass = random_base_ratio * np.array(base_mass)
            lb, ub = (lower_bound*np.array(base_mass), upper_bound*np.array(base_mass))

        minitaur.SetBaseMasses(randomized_base_mass)
        return randomized_base_mass, lb, ub
    
    def _randomize_leg_mass(self, minitaur, lower_bound, upper_bound):
        if "leg mass" in self.param_dict:
            _, lb, ub = self.param_dict["leg mass"]
            randomized_leg_masses = np.random.uniform(lb, ub)
        else:
            leg_masses = minitaur.GetLegMassesFromURDF()
            random_leg_ratio = np.random.uniform(lower_bound, upper_bound)
            randomized_leg_masses = random_leg_ratio * np.array(leg_masses)
            lb, ub = (lower_bound*np.array(leg_masses), upper_bound*np.array(leg_masses))

        minitaur.SetLegMasses(randomized_leg_masses)
        return randomized_leg_masses, lb, ub 

    def _randomize_inertia(self, minitaur, lower_bound, upper_bound):
        if "inertia" in self.param_dict:
            _, lb, ub = self.param_dict["inertia"]
            randomized_base_inertia = np.random.uniform(lb, ub)
        else:
            base_inertia = minitaur.GetBaseInertiasFromURDF()
            random_base_ratio = np.random.uniform(lower_bound, upper_bound)
            randomized_base_inertia = random_base_ratio * np.array(base_inertia)
            lb, ub = (lower_bound*np.array(base_inertia), upper_bound*np.array(base_inertia))

        minitaur.SetBaseInertias(randomized_base_inertia)       
        return randomized_base_inertia, lb, ub
    
    def _randomize_leg_inertia(self, minitaur, lower_bound, upper_bound):
        if "leg inertia" in self.param_dict:
            _, lb, ub = self.param_dict["leg inertia"]
            randomized_leg_inertia = np.random.uniform(lb, ub)
        else:
            leg_inertia = minitaur.GetLegInertiasFromURDF()
            random_leg_ratio = np.random.uniform(lower_bound, upper_bound)
            randomized_leg_inertia = random_leg_ratio * np.array(leg_inertia)
            lb, ub = (lower_bound*np.array(leg_inertia), upper_bound*np.array(leg_inertia))

        minitaur.SetLegInertias(randomized_leg_inertia)
        return randomized_leg_inertia, lb, ub
    
    def _randomize_latency(self, minitaur, lower_bound, upper_bound):
        if "latency" in self.param_dict:
            _, lb, ub = self.param_dict["latency"]
            randomized_latency = np.random.uniform(lb, ub)
        else:
            randomized_latency = np.random.uniform(lower_bound, upper_bound)
            lb, ub = lower_bound, upper_bound

        minitaur.SetControlLatency(randomized_latency)
        return randomized_latency, lb, ub
    
    def _randomize_joint_friction(self, minitaur, lower_bound, upper_bound):
        if "joint friction" in self.param_dict:
            _, lb, ub = self.param_dict["joint friction"]
            randomized_joint_frictions = np.random.uniform(lb, ub)
        else:
            num_knee_joints = minitaur.GetNumKneeJoints()
            randomized_joint_frictions = np.random.uniform([lower_bound] * num_knee_joints,
                                                    [upper_bound] * num_knee_joints)
            lb, ub = (np.array([lower_bound]*num_knee_joints), np.array([upper_bound]*num_knee_joints))

        minitaur.SetJointFriction(randomized_joint_frictions)
        return randomized_joint_frictions, lb, ub
    
    def _randomize_motor_friction(self, minitaur, lower_bound, upper_bound):
        if "motor friction" in self.param_dict:
            _, lb, ub = self.param_dict["motor friction"]
            randomized_motor_damping = np.random.uniform(lb, ub)
        else:
            randomized_motor_damping = np.random.uniform(lower_bound, upper_bound)
            lb, ub = (lower_bound, upper_bound)

        minitaur.SetMotorViscousDamping(randomized_motor_damping)
        return randomized_motor_damping, lb, ub
    
    def _randomize_contact_restitution(self, minitaur, lower_bound, upper_bound):
        if "restitution" in self.param_dict:
            _, lb, ub = self.param_dict["restitution"]
            randomized_restitution = np.random.uniform(lb, ub)
        else:
            randomized_restitution = np.random.uniform(lower_bound, upper_bound)
            lb, ub = (lower_bound, upper_bound)

        minitaur.SetFootRestitution(randomized_restitution)
        return randomized_restitution, lb, ub

    def _randomize_contact_friction(self, minitaur, lower_bound, upper_bound):
        if "lateral friction" in self.param_dict:
            _, lb, ub = self.param_dict["lateral friction"]
            randomized_foot_friction = np.random.uniform(lb, ub)
        else:
            randomized_foot_friction = np.random.uniform(lower_bound, upper_bound)
            lb, ub = (lower_bound, upper_bound)

        minitaur.SetFootFriction(randomized_foot_friction)
        return randomized_foot_friction, lb, ub
    
    def _randomize_battery_level(self, minitaur, lower_bound, upper_bound):
        if "battery" in self.param_dict:
            _, lb, ub = self.param_dict["battery"]
            randomized_battery_voltage = np.random.uniform(lb, ub)
        else:
            randomized_battery_voltage = np.random.uniform(lower_bound, upper_bound)
            lb, ub = (lower_bound, upper_bound)

        minitaur.SetBatteryVoltage(randomized_battery_voltage)
        return randomized_battery_voltage, lb, ub
    
    def _randomize_motor_strength(self, minitaur, lower_bound, upper_bound):
        if "motor strength" in self.param_dict:
            _, lb, ub = self.param_dict["motor strength"]
            randomized_motor_strength_ratios = np.random.uniform(lb, ub)
        else:
            randomized_motor_strength_ratios = np.random.uniform([lower_bound] * minitaur.num_motors,
                                                            [upper_bound] * minitaur.num_motors)
            lb, ub = (np.array([lower_bound] * minitaur.num_motors), np.array([upper_bound] * minitaur.num_motors))

        minitaur.SetMotorStrengthRatios(randomized_motor_strength_ratios)
        return randomized_motor_strength_ratios, lb, ub

class StaticEnvRandomizer(MinitaurEnvRandomizer):
    def __init__(self):
        self.randomization_param_dict = all_params()
        self.step = 0
        self.param_dict = {}
    def randomize_env(self, env):
        if self.step == 0:
            self._randomization_function_dict = self._build_randomization_function_dict(env)
            for param_name, random_range in iter(self.randomization_param_dict.items()):
                self.param_dict[param_name] = self._randomization_function_dict[param_name](lower_bound=random_range[0],
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

        img = env.render()
        img = img[125:300, 100:350, :]
        img = cv2.resize(img, dsize=(84,84), interpolation=cv2.INTER_CUBIC)
        frames.append(img)

        # imgplot = plt.imshow(frames[-1])
        # plt.show()
    frames = np.asarray(frames)
    frames = np.transpose(frames, (0, 3, 1, 2))
    return frames, obs


class SimPramRandomizer(MinitaurEnvRandomizer):
    def __init__(self, env, agent, batch_size):
        self.randomization_param_dict = all_params()
        self.param_dict = {}
        self.randomize_env(env)
        self.env = env
        self.agent = agent
        self.param_elems, self.param_shape = self._get_params()
        self.param_mean = []

        frame, obs = get_frame_captures(env, env.action_space.sample())       
        self.SPM = SimParamModel(shape=len(self.param_elems), action_space=env.action_space, state_dim=obs[0].size, layers=2, units=50, device=torch.device('cuda'), obs_shape=frame[0].shape, 
            encoder_type='pixel', encoder_feature_dim=50, encoder_num_layers=4, encoder_num_filters=32, batch_size=batch_size, normalize_features=True, clip_positive=True,  )

    def _get_params(self):
        param_shape = OrderedDict()
        param_elem = []
        for param_name, param_obj in self.param_dict.items():
            # param_obj is a 3-tuple (parameter_value, lower_bound, upper_bound)
            param_obj = param_obj[0]
            if isinstance(param_obj, np.ndarray):
                param_shape[param_name] = param_obj.shape
                #self.param_dict[param_name] = param_obj.flatten()
                param_elem += param_obj.flatten().tolist()
            else:
                param_shape[param_name] = 1
                param_elem.append(param_obj)
        return param_elem, param_shape


    def collect_rollout(self, n_steps = 1000):
        obs = self.env.reset() 

        traj = []
        for i in range(n_steps):
            action, state = self.agent.predict(obs)
            obs, _, _, _ = self.env.step(action)
            traj.append([[], obs, action])

        return traj

    def spm_train(self):

        x = self.collect_rollout()

        self.SPM.train() 
        loss, acc = self.SPM.update(x, self.param_elems, self.distribution_mean)
        return loss, acc

    def randomize_env(self, env):    
        super(SimPramRandomizer, self).randomize_env(env)

    def randomize_step(self, env):
        pass

    @property
    def distribution_mean(self):
        param_mean_elems = []
        for param_name, param_obj in self.param_dict.items():
            _, lb, ub = param_obj
            mean = (lb + ub)/2

            if np.isscalar(mean):
                param_mean_elems.append(mean)
            else:
                param_mean_elems.extend( mean.flatten() )

        return np.array(param_mean_elems)



    def evaluate_spm(self, real_env):

        traj = self.collect_rollout()
        full_traj = [traj]

        with torch.no_grad():
            pred_class = self.SPM.forward_classifier(full_traj, [self.param_elems])

            pred_class_flat = pred_class.flatten().unsqueeze(0).detach().cpu()

            real_params = real_env._env_randomizers[0].distribution_mean
            ground_truth = self.param_elems > real_params
            ground_truth = torch.Tensor(ground_truth.reshape(1, -1))

            loss = nn.BCELoss()(pred_class_flat, ground_truth).detach().cpu().item()
            accuracy = np.mean((torch.round(pred_class_flat) == ground_truth).float().detach().cpu().numpy())

        return loss, accuracy

    def update_params(self, real_env, alpha=0.1):
        self.env
        self.agent

        obs = real_env.reset()

        actions = []

        traj = self.collect_rollout()

        full_traj = [traj]
        with torch.no_grad():
            preds = self.SPM.forward_classifier(full_traj, [self.param_elems]).detach().cpu().numpy()

        mask = (preds > 0.3) & (preds < 0.7)
        preds = np.round(preds)
        preds[mask] = 0.5
        confidence_preds = np.mean(preds, axis=0)

        alpha = 0.1
        new_update = -alpha * (confidence_preds - 0.5)

        # Shift the lower bound and upper bounds of the distribution in the predicted direction
        idx = 0
        for param_name, param_obj in self.param_dict.items():
            param_vals, lb, ub = param_obj

            # Pick the corresponding elements from new_update and reshape it to the same shape as this parameter
            if np.isscalar(lb):
                param_updates = new_update[idx]
                idx += 1
            else:
                param_updates = new_update[idx:idx+lb.size].reshape(lb.shape)
                idx += lb.size

            # At least some parameters must be non-negative
            lb = np.fmax(lb + param_updates, 0)
            ub = np.fmax(ub + param_updates, 0)

            self.param_dict[param_name] = (param_vals, lb, ub)

            # For debugging, report any changes to parameters if parameters whose shape isn't too large
            if np.any(param_updates) and lb.size < 10:
                print(f"Updating bounds for {param_name} parameter to")
                print(lb)
                print(ub)

        return 
