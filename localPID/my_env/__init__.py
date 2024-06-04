from gym.envs.registration import register

register(
    id='CameraControlLocal-v0',
    entry_point='my_env.CameraControlLocal:CameraControlLocalEnv',
)