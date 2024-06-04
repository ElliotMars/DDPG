from gym.envs.registration import register

register(
    id='Kalman-v0',
    entry_point='my_env.Kalman:KalmanEnv',
)