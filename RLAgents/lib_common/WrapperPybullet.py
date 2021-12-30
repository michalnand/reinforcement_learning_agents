import numpy
import pybullet

from pybullet_envs.bullet import minitaur_gym_env
from pybullet_envs.bullet import minitaur_env_randomizer

from pybullet_envs.bullet import racecarGymEnv

from pybullet_envs.bullet import KukaGymEnv
 
def WrapperMinitaurBulletEnv(render = False): 
    randomizer = minitaur_env_randomizer.MinitaurEnvRandomizer()

    env = minitaur_gym_env.MinitaurBulletEnv(
        render=(render == True),
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


def WrapperRacecarGymEnv(render = False): 

    env = racecarGymEnv.RacecarGymEnv(renders=(render == True))

    if render:
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_SHADOWS,0)
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI,0)

    return env

def WrapperKukaGymEnv(render = False): 

    env = KukaGymEnv(renders=(render == True))

    if render:
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_SHADOWS,0)
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI,0)

    return env