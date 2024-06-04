import numpy as np
STEP_LEN = [0.001, 0.001, 0.001]

class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.002, theta=0.01, dt=0.05):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt

    def noise(self, Kp, Ki, Kd):
        self.x_prev = np.array([Kp, Ki, Kd])
        dx = self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.random.normal(size=self.mu.shape) * np.sqrt(self.dt)
        #dx[2] *= (STEP_LEN[2] / STEP_LEN[0])
        for i in range(len(dx)):
            dx[i] = min(max(dx[i], -STEP_LEN[i]), STEP_LEN[i])
        return dx

OUActionNoise = OrnsteinUhlenbeckActionNoise(mu=np.array([0.02, 0.01, 0.01]))
for i in range(10):
    print(OUActionNoise.noise(0.02, 0.01, 0.01))