from gym.envs.registration import register

register(
    id='CameraControl-v0',
    entry_point='my_env.CameraControl:CameraControlEnv',
)