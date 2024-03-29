import numpy
import pybullet
import gym

from pybullet_envs.bullet import minitaur_gym_env
from pybullet_envs.bullet import minitaur_env_randomizer
#from pybullet_envs.minitaur.envs import minitaur_randomize_terrain_gym_env
 

from pybullet_envs.bullet import racecarGymEnv

from pybullet_envs.bullet import KukaGymEnv


class MaxStepsWrapper(gym.Wrapper):
    def __init__(self, env, max_steps):
        gym.Wrapper.__init__(self, env)

        self.steps      = 0
        self.max_steps  = max_steps

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        self.steps+= 1
        if self.steps >= self.max_steps:
            self.steps = 0
            done = True
 
        return obs, reward, done, info

    def reset(self):
        self.steps = 0
        return self.env.reset()
 
def WrapperMinitaurBulletEnv(name = "none", render = False, max_steps = 1000): 
    randomizer = minitaur_env_randomizer.MinitaurEnvRandomizer()

    env = minitaur_gym_env.MinitaurBulletEnv(
        render=render,
        motor_velocity_limit=numpy.inf,
        pd_control_enabled=True,
        hard_reset=False,
        env_randomizer=randomizer,
        shake_weight=0.0,
        drift_weight=0.0,
        energy_weight=0.0,
        on_rack=False)

    env = MaxStepsWrapper(env, max_steps)

    if render:
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_SHADOWS,0)
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI,0)


    return env

def WrapperMinitaurBulletEnvGUI(name = "none"): 
    return WrapperMinitaurBulletEnv(name, True)


def WrapperMinitaurRandomizeTerrainGymEnv(name = "none"): 

    render = True
    randomizer = minitaur_env_randomizer.MinitaurEnvRandomizer()

    env =minitaur_randomize_terrain_gym_env.MinitaurRandomizeTerrainGymEnv(
        render=render,
        motor_velocity_limit=numpy.inf,
        pd_control_enabled=True,
        hard_reset=False,
        env_randomizer=randomizer,
        shake_weight=0.0,
        drift_weight=0.0,
        energy_weight=0.0,
        on_rack=False)

    env = MaxStepsWrapper(env, max_steps)

    if render:
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_SHADOWS,0)
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI,0)


    return env


def WrapperRacecarGymEnv(name = "none", max_steps = 1000): 
    render = False

    env = racecarGymEnv.RacecarGymEnv(renders=render)

    env = MaxStepsWrapper(env, max_steps)

    if render:
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_SHADOWS,0)
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI,0)

    return env

def WrapperKukaGymEnv(name = "none", max_steps = 1000):  
    render = False

    env = KukaGymEnv(renders=render)

    env = MaxStepsWrapper(env, max_steps)

    if render:
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_SHADOWS,0)
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI,0)

    return env