from gym.envs.registration import register

register(
    id='DirectControl-v0',
    entry_point='my_env.DirectControl:DirectControlEnv',
)