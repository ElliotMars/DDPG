from gym.envs.registration import register

register(
    id='CameraControlnophase-v0',
    entry_point='my_env.CameraControlnophase:CameraControlnophaseEnv',
)