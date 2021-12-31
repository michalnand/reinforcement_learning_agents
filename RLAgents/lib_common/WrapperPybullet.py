import numpy
import pybullet

from pybullet_envs.bullet import minitaur_gym_env
from pybullet_envs.bullet import minitaur_env_randomizer
#rom pybullet_envs.minitaur.envs import minitaur_alternating_legs_env_example
 

from pybullet_envs.bullet import racecarGymEnv

from pybullet_envs.bullet import KukaGymEnv
 
def WrapperMinitaurBulletEnv(name = "none"): 

    render = False
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

    if render:
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_SHADOWS,0)
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI,0)


    return env


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

    if render:
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_SHADOWS,0)
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI,0)


    return env


def WrapperRacecarGymEnv(name = "none"): 
    render = False

    env = racecarGymEnv.RacecarGymEnv(renders=render)

    if render:
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_SHADOWS,0)
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI,0)

    return env

def WrapperKukaGymEnv(name = "none"):  
    render = False

    env = KukaGymEnv(renders=render)

    if render:
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_SHADOWS,0)
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI,0)

    return env