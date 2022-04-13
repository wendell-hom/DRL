import collections
import math
import pybullet as pb
from gym import spaces
import numpy as np
from pybullet_envs.minitaur.envs import minitaur_gym_env

INIT_EXTENSION_POS = 2.0
INIT_SWING_POS = 0.0
NUM_LEGS = 4
NUM_MOTORS = 2 * NUM_LEGS

MinitaurPose = collections.namedtuple(
    "MinitaurPose", "swing_angle_1, swing_angle_2, swing_angle_3, swing_angle_4, "
    "extension_angle_1, extension_angle_2, extension_angle_3, "
    "extension_angle_4")

def get_camera():
    width = 128
    height = 128

    fov = 60
    aspect = width / height
    near = 0.02
    far = 1

    view_matrix = pb.computeViewMatrix([0, 0, 0.5], [0, 0, 0], [1, 0, 0])
    projection_matrix = pb.computeProjectionMatrixFOV(fov, aspect, near, far)

    # Get depth values using the OpenGL renderer
    images = pb.getCameraImage(width,
                            height,
                            view_matrix,
                            projection_matrix,
                            shadow=True,
                            renderer=pb.ER_BULLET_HARDWARE_OPENGL)
    # NOTE: the ordering of height and width change based on the conversion
    rgb_opengl = np.reshape(images[2], (height, width, 4)) * 1. / 255.
    depth_buffer_opengl = np.reshape(images[3], [width, height])
    depth_opengl = far * near / (far - (far - near) * depth_buffer_opengl)
    seg_opengl = np.reshape(images[4], [width, height]) * 1. / 255.

    return (rgb_opengl, depth_opengl, seg_opengl)


class MinitaurReactiveEnv(minitaur_gym_env.MinitaurGymEnv):
    """The gym environment for the minitaur.

    It simulates the locomotion of a minitaur, a quadruped robot. The state space
    include the angles, velocities and torques for all the motors and the action
    space is the desired motor angle for each motor. The reward function is based
    on how far the minitaur walks in 1000 steps and penalizes the energy
    expenditure.

    """
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 166}

    def __init__(self,
                urdf_version=None,
                energy_weight=0.005,
                control_time_step=0.006,
                action_repeat=6,
                control_latency=0.02,
                pd_latency=0.003,
                on_rack=False,
                motor_kp=1.0,
                motor_kd=0.015,
                remove_default_joint_damping=True,
                render=False,
                num_steps_to_log=1000,
                accurate_motor_model_enabled=True,
                use_angle_in_observation=True,
                hard_reset=False,
                env_randomizer=None,
                log_path=None):

        self._use_angle_in_observation = use_angle_in_observation

        super(MinitaurReactiveEnv, self).__init__(urdf_version=urdf_version,
                            energy_weight=energy_weight,
                            accurate_motor_model_enabled=accurate_motor_model_enabled,
                            motor_overheat_protection=True,
                            motor_kp=motor_kp,
                            motor_kd=motor_kd,
                            remove_default_joint_damping=remove_default_joint_damping,
                            control_latency=control_latency,
                            pd_latency=pd_latency,
                            on_rack=on_rack,
                            render=render,
                            hard_reset=hard_reset,
                            num_steps_to_log=num_steps_to_log,
                            env_randomizer=env_randomizer,
                            log_path=log_path,
                            control_time_step=control_time_step,
                            action_repeat=action_repeat)

        action_dim = 8
        action_low = np.array([-0.5] * action_dim)
        action_high = -action_low
        self.action_space = spaces.Box(action_low, action_high)  # type: ignore
        self._cam_dist = 1.0
        self._cam_yaw = 30
        self._cam_pitch = -30

    def reset(self):
        # TODO(b/73666007): Use composition instead of inheritance.
        # (http://go/design-for-testability-no-inheritance).
        init_pose = MinitaurPose(swing_angle_1=INIT_SWING_POS,
                                swing_angle_2=INIT_SWING_POS,
                                swing_angle_3=INIT_SWING_POS,
                                swing_angle_4=INIT_SWING_POS,
                                extension_angle_1=INIT_EXTENSION_POS,
                                extension_angle_2=INIT_EXTENSION_POS,
                                extension_angle_3=INIT_EXTENSION_POS,
                                extension_angle_4=INIT_EXTENSION_POS)
        # TODO(b/73734502): Refactor input of _convert_from_leg_model to namedtuple.
        initial_motor_angles = self._convert_from_leg_model(list(init_pose))
        super(MinitaurReactiveEnv, self).reset(initial_motor_angles=initial_motor_angles,
                                            reset_duration=0.5)
        return self._get_observation()

    def _convert_from_leg_model(self, leg_pose):
        motor_pose = np.zeros(NUM_MOTORS)
        for i in range(NUM_LEGS):
            motor_pose[int(2 * i)] = leg_pose[NUM_LEGS + i] - (-1)**int(i / 2) * leg_pose[i]
            motor_pose[int(2 * i + 1)] = (leg_pose[NUM_LEGS + i] + (-1)**int(i / 2) * leg_pose[i])
        return motor_pose

    def _signal(self, t):
        initial_pose = np.array([
            INIT_SWING_POS, INIT_SWING_POS, INIT_SWING_POS, INIT_SWING_POS, INIT_EXTENSION_POS,
            INIT_EXTENSION_POS, INIT_EXTENSION_POS, INIT_EXTENSION_POS
        ])
        return initial_pose

    def _transform_action_to_motor_command(self, action):
        # Add the reference trajectory (i.e. the trotting signal).
        action += self._signal(self.minitaur.GetTimeSinceReset())
        return self._convert_from_leg_model(action)

    def is_fallen(self):
        """Decides whether the minitaur is in a fallen state.

        If the roll or the pitch of the base is greater than 0.3 radians, the
        minitaur is considered fallen.

        Returns:
        Boolean value that indicates whether the minitaur has fallen.
        """
        roll, pitch, _ = self.minitaur.GetTrueBaseRollPitchYaw()
        return math.fabs(roll) > 0.3 or math.fabs(pitch) > 0.3

    def _get_true_observation(self):
        """Get the true observations of this environment.

        It includes the roll, the pitch, the roll dot and the pitch dot of the base.
        If _use_angle_in_observation is true, eight motor angles are added into the
        observation.

        Returns:
        The observation list, which is a numpy array of floating-point values.
        """
        roll, pitch, _ = self.minitaur.GetTrueBaseRollPitchYaw()
        roll_rate, pitch_rate, _ = self.minitaur.GetTrueBaseRollPitchYawRate()
        observation = [roll, pitch, roll_rate, pitch_rate]
        if self._use_angle_in_observation:
            observation.extend(self.minitaur.GetMotorAngles().tolist())
        self._true_observation = np.array(observation)
        return self._true_observation

    def _get_observation(self):
        roll, pitch, _ = self.minitaur.GetBaseRollPitchYaw()
        roll_rate, pitch_rate, _ = self.minitaur.GetBaseRollPitchYawRate()
        observation = [roll, pitch, roll_rate, pitch_rate]
        if self._use_angle_in_observation:
            observation.extend(self.minitaur.GetMotorAngles().tolist())
        self._observation = np.array(observation)
        return self._observation

    def _get_observation_upper_bound(self):
        """Get the upper bound of the observation.

        Returns:
        The upper bound of an observation. See _get_true_observation() for the
        details of each element of an observation.
        """
        upper_bound_roll = 2 * math.pi
        upper_bound_pitch = 2 * math.pi
        upper_bound_roll_dot = 2 * math.pi / self._time_step
        upper_bound_pitch_dot = 2 * math.pi / self._time_step
        upper_bound_motor_angle = 2 * math.pi
        upper_bound = [
            upper_bound_roll, upper_bound_pitch, upper_bound_roll_dot, upper_bound_pitch_dot
        ]

        if self._use_angle_in_observation:
            upper_bound.extend([upper_bound_motor_angle] * NUM_MOTORS)
        return np.array(upper_bound)

    def _get_observation_lower_bound(self):
        lower_bound = -self._get_observation_upper_bound()
        return lower_bound


class MinitaurExtendedEnv(MinitaurReactiveEnv):
    """The 'extended' environment for Markovian property.

    This class implements to include prior actions and observations to the
    observation vector, thus making the environment "more" Markovian. This is
    especially useful for systems with latencies.

    Args:
        history_length: the length of the historic data
        history_include_actions: a flag for including actions as history
        history_include_states: a flag for including states as history
        include_state_difference: a flag for including the first-order differences
        as history
        include_second_state_difference: a flag for including the second-order state
        differences as history.
        include_base_position: a flag for including the base as observation,
        never_terminate: if this is on, the environment unconditionally never
        terminates.
        action_scale: the scale of actions,
    """
    MAX_BUFFER_SIZE = 1001
    ACTION_DIM = 8
    PARENT_OBSERVATION_DIM = 12
    INIT_EXTENSION_POS = 2.0
    INIT_SWING_POS = 0.0

    metadata = {
        "render.modes": ["human", "rgb_array"],
        "video.frames_per_second": 50,
    }

    def __init__(self,
                history_length=1,
                history_include_actions=True,
                history_include_states=True,
                include_state_difference=True,
                include_second_state_difference=True,
                include_base_position=True,
                include_leg_model=True,
                never_terminate=True,
                action_scale=0.5,
                **kwargs):
        self._kwargs = kwargs

        self._history_length = history_length
        self._history_include_actions = history_include_actions
        self._history_include_states = history_include_states
        self._include_state_difference = include_state_difference
        self._include_second_state_difference = include_second_state_difference
        self._include_base_position = include_base_position
        self._include_leg_model = include_leg_model

        self._never_terminate = never_terminate
        self._action_scale = action_scale

        self._past_parent_observations = np.zeros((self.MAX_BUFFER_SIZE + 1,
                                                self.PARENT_OBSERVATION_DIM))
        self._past_motor_angles = np.zeros((self.MAX_BUFFER_SIZE + 1, 8))
        self._past_actions = np.zeros((self.MAX_BUFFER_SIZE, self.ACTION_DIM))
        self._counter = 0

        super(MinitaurExtendedEnv, self).__init__(**kwargs)
        self.action_space = spaces.Box(-1.0, 1.0, self.action_space.shape) # type: ignore
        self.observation_space = spaces.Box(-np.inf, np.inf,
                                            self._get_observation().shape)
        # This is mainly for the TF-Agents compatibility
        self.action_space.flat_dim = len(self.action_space.low) # type: ignore
        self.observation_space.flat_dim = len(self.observation_space.low)  # type: ignore

    def _get_observation(self):
        """Maybe concatenate motor velocity and torque into observations."""
        parent_observation = super(MinitaurExtendedEnv, self)._get_observation()
        parent_observation = np.array(parent_observation)
        # Base class might require this.
        self._observation = parent_observation
        self._past_parent_observations[self._counter] = parent_observation
        num_motors = self.minitaur.num_motors
        self._past_motor_angles[self._counter] = parent_observation[-num_motors:]

        history_states = []
        history_actions = []
        for i in range(self._history_length):
            t = max(self._counter - i - 1, 0)

        if self._history_include_states:
            history_states.append(self._past_parent_observations[t]) # type: ignore

        if self._history_include_actions:
            history_actions.append(self._past_actions[t]) # type: ignore

        t = self._counter
        tm, tmm = max(0, self._counter - 1), max(0, self._counter - 2)

        state_difference, second_state_difference = [], []
        if self._include_state_difference:
            state_difference = [
                self._past_motor_angles[t] - self._past_motor_angles[tm]
            ]
        if self._include_second_state_difference:
            second_state_difference = [
                self._past_motor_angles[t] - 2 * self._past_motor_angles[tm] +
                self._past_motor_angles[tmm]
            ]

        base_position = []
        if self._include_base_position:
            base_position = np.array((self.minitaur.GetBasePosition()))

        leg_model = []
        if self._include_leg_model:
            raw_motor_angles = self.minitaur.GetMotorAngles()
            leg_model = self.convert_to_leg_model(raw_motor_angles)

        observation_list = ( [parent_observation] + history_states + history_actions + state_difference + second_state_difference + [base_position] +  [leg_model])  # type: ignore

        full_observation = np.concatenate(observation_list)
        return full_observation

    def reset(self):
        """Resets the time and history buffer."""
        self._counter = 0
        self._signal(self._counter)  # This sets the current phase
        self._past_parent_observations = np.zeros((self.MAX_BUFFER_SIZE + 1,
                                                self.PARENT_OBSERVATION_DIM))
        self._past_motor_angles = np.zeros((self.MAX_BUFFER_SIZE + 1, 8))
        self._past_actions = np.zeros((self.MAX_BUFFER_SIZE, self.ACTION_DIM))
        self._counter = 0

        return np.array(super(MinitaurExtendedEnv, self).reset())

    def step(self, action):
        """Step function wrapper can be used to add shaping terms to the reward.

        Args:
        action: an array of the given action

        Returns:
        next_obs: the next observation
        reward: the reward for this experience tuple
        done: the terminal flag
        info: an additional information
        """

        action *= self._action_scale

        self._past_actions[self._counter] = action
        self._counter += 1

        next_obs, _, done, info = super(MinitaurExtendedEnv, self).step(action)

        reward = self.reward()
        info.update(base_reward=reward)

        return next_obs, reward, done, info

    def terminate(self):
        """The helper function to terminate the environment."""
        super(MinitaurExtendedEnv, self)._close() # type: ignore

    def _termination(self):
        """Determines whether the env is terminated or not.

        checks whether 1) the front leg is bent too much or 2) the time exceeds
        the manually set weights.

        Returns:
        terminal: the terminal flag whether the env is terminated or not
        """
        if self._never_terminate:
            return False

        leg_model = self.convert_to_leg_model(self.minitaur.GetMotorAngles())
        swing0 = leg_model[0] # type: ignore
        swing1 = leg_model[2] # type: ignore
        maximum_swing_angle = 0.8
        if swing0 > maximum_swing_angle or swing1 > maximum_swing_angle:
            return True

        if self._counter >= 500:
            return True

        return False

    def reward(self):
        """Compute rewards for the given time step.

        It considers two terms: 1) forward velocity reward and 2) action
        acceleration penalty.

        Returns:
        reward: the computed reward.
        """
        current_base_position = self.minitaur.GetBasePosition()
        dt = self.control_time_step
        velocity = (current_base_position[0] - self._last_base_position[0]) / dt
        velocity_reward = np.clip(velocity, -0.5, 0.5)

        action = self._past_actions[self._counter - 1]
        prev_action = self._past_actions[max(self._counter - 2, 0)]
        prev_prev_action = self._past_actions[max(self._counter - 3, 0)]
        acc = action - 2 * prev_action + prev_prev_action
        action_acceleration_penalty = np.mean(np.abs(acc))

        reward = 0.0
        reward += 1.0 * velocity_reward
        reward -= 0.1 * action_acceleration_penalty

        return reward

    @staticmethod
    def convert_to_leg_model(motor_angles):
        """A helper function to convert motor angles to leg model.

        Args:
        motor_angles: raw motor angles:

        Returns:
        leg_angles: the leg pose model represented in swing and extension.
        """
        # TODO(sehoonha): clean up model conversion codes
        num_legs = 4
        # motor_angles = motor_angles / (np.pi / 4.)
        leg_angles = np.zeros(num_legs * 2)
        for i in range(num_legs):
            motor1, motor2 = motor_angles[2 * i:2 * i + 2]
            swing = (-1)**(i // 2) * 0.5 * (motor2 - motor1)
            extension = 0.5 * (motor1 + motor2)

            leg_angles[i] = swing
            leg_angles[i + num_legs] = extension

        return leg_angles

    def __getstate__(self):
        """A helper get state function for pickling."""
        return {"kwargs": self._kwargs}

    def __setstate__(self, state):
        """A helper set state function for pickling."""
        self.__init__(**state["kwargs"])
