import gym
from gym import spaces
import numpy as np
import math
import random

from typing import Optional

BINGO_RANGE = 10
ROU = 0.5
AMPLIFY = 1
TOP_DIS = 276
TOP_PixelRANGE = [3000, 2010]

from DirectControl.DC_agent import STEP_LEN, ANGLEX0, ANGLEY0

class DirectControlEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        #super(CustomEnv, self).__init__()

        # 使用指示
        self.isopen = True

        # 高度和速度
        self.Height = 120

        # 迭代间隔时间
        self.t_plot = 0.05

        # 像素中心坐标
        self.x0 = 320
        self.y0 = 240

        # #每段循环次数
        # self.iterations = 100

        # 卡尔曼滤波参数
        self.Q = np.diag([0.001, 0.0001, 0.001, 0.0001])
        self.R = np.diag([2, 2])
        self.F = np.array([[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]])
        self.H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
        self.P = np.eye(4)

        #self.xt1 = None
        #self.yt1 = None


        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:影响Actor Net | step | train random action OU噪声
        self.action_space = spaces.Box(low=np.array([-1, -1]),
                                       high=np.array([1, 1]), shape=(2,), dtype=np.float64)
        # Example for using image as input:影响step
        self.observation_space = spaces.Box(low=np.array([0, 0, -1, -1, -1, -1]),
                                            high=np.array([1, 1, 1, 1, 1, 1]),
                                            shape=(6,), dtype=np.float64)

    def step(self, action):
        #print(action)
        self.aim_times = 0
        self.distance = 0

        # if self.step_count == 1:
        #     self.Initial()

        self.xt1 = self.trajectoryx[(self.step_count)]
        self.yt1 = self.trajectoryy[(self.step_count)]

        self.CameraCoord()

        # 添加观测噪声
        self.x1_O = self.x1 + np.random.normal(0, 2)
        self.y1_O = self.y1 + np.random.normal(0, 2)
            #self.Coordx_O.append(self.xt1_O)
            #self.Coordy_O.append(self.yt1_O)

        Z = np.array([[self.x1_O], [self.y1_O]])
        self.Kalman_Filter(Z)
        self.x1_F, self.y1_F = self.ux[0, 0], self.ux[2, 0]

        self.control(action)

        self.t += self.t_plot
        self.step_count += 1

        if (abs(self.xt1 - self.xt) <= BINGO_RANGE and abs(self.yt1 - self.yt) <= BINGO_RANGE):
            self.aim_times = 1
        else:
            self.aim_times = 0
        self.distance = abs(self.xt1 - self.xt) + abs(self.yt1 - self.yt)


        observation = self.get_obs()
        #print('observation = ', observation)
        reward = self.get_RE()  # TODO
        #print('reward = ',reward)
        done = False
        truncation = False
        info = {}

        return observation, reward, done, truncation, info

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):

        # # 继承/初始PID
        # self.Kp = KP0
        # self.Ki = KI0
        # self.Kd = KD0
        self.anglex = ANGLEX0
        self.angley = ANGLEY0


        self.step_count = 0

        trajectory = random.randint(1,3)
        #trajectory = 3

        if trajectory == 1:
            print('diagonal')
            #对角线
            self.trajectoryx = np.concatenate([np.linspace(63.63, -63.63, 200),
                                               np.linspace(-63.63, 63.63, 200),
                                               np.linspace(63.63, -63.63, 200)])
            self.trajectoryy = np.concatenate([np.linspace(-63.63, 63.63, 200),
                                               np.linspace(63.63, -63.63, 200),
                                               np.linspace(-63.63, 63.63, 200)])

        elif trajectory == 2:
            print('square')
            #沿着边界的方块型
            self.trajectoryx = np.concatenate([np.linspace(67.5, 67.5, 150),
                                           np.linspace(67.5, -67.5, 150),
                                           np.linspace(-67.5, -67.5, 150),
                                           np.linspace(-67.5, 67.5, 150)])
            self.trajectoryy = np.concatenate([np.linspace(-67.5, 67.5, 150),
                                           np.linspace(67.5, 67.5, 150),
                                           np.linspace(67.5, -67.5, 150),
                                           np.linspace(-67.5, -67.5, 150)])

        else:
            print('lemniscate')
            # 8字型曲线
            # 每组PID迭代次数ying
            t = np.linspace(0, 2 * np.pi, 600)
            a = 60
            self.trajectoryx = a * np.cos(t)
            self.trajectoryy = a * np.sin(2 * t)

        self.t = 0

        self.xt1 = self.trajectoryx[0]
        self.yt1 = self.trajectoryy[0]
        self.CameraCoord()
        self.ux = np.array([[self.x1], [0], [self.y1], [0]])
        self.x1_O = self.x1 + np.random.normal(0, 2)
        self.y1_O = self.y1 + np.random.normal(0, 2)
        Z = np.array([[self.x1_O], [self.y1_O]])
        self.Kalman_Filter(Z)
        self.x1_F, self.y1_F = self.ux[0, 0], self.ux[2, 0]
        self.xt = self.transform(self.anglex)
        self.yt = self.transform(self.angley)


        if (abs(self.xt1 - self.xt) <= BINGO_RANGE and abs(self.yt1 - self.yt) <= BINGO_RANGE):
            self.aim_times = 1
        else:
            self.aim_times = 0
        self.distance = abs(self.xt1 - self.xt) + abs(self.yt1 - self.yt)


        observation = self.get_obs()

        return observation  # reward, done, info can't be included

    # def render(self, mode='human'):
    #     ...

    def close(self):
        self.isopen = False

    def get_obs(self):
        obs = np.array([self.distance / TOP_DIS, self.aim_times,
                        self.anglex / STEP_LEN[0], self.angley / STEP_LEN[1],
                        self.x1_F / TOP_PixelRANGE[0], self.y1_F / TOP_PixelRANGE[1]], dtype=np.float64)
        return obs

    def get_RE(self):
        re1 = -ROU * self.distance / TOP_DIS
        re2 = (1 - ROU) * self.aim_times
        reward = AMPLIFY * (re1 + re2)
        return reward

    def CameraCoord(self):
        xw = self.xt1
        yw = self.yt1
        zw = self.Height
        theta = self.deg2rad(self.anglex)
        alpha = self.deg2rad(self.angley)
        f = 1.8e-3
        dx, dy = 3.14e-6, 3.75e-6
        u0, v0 = 320, 240

        # Transformation
        x_temp = xw * np.cos(theta) - zw * np.sin(theta)
        y_temp = yw
        z_temp = xw * np.sin(theta) + zw * np.cos(theta)
        xc = x_temp
        yc = y_temp * np.cos(alpha) - z_temp * np.sin(alpha)
        zc = y_temp * np.sin(alpha) + z_temp * np.cos(alpha)

        x = (f * xc) / zc
        y = (f * yc) / zc

        # 像素坐标
        self.x1 = x / dx + u0
        self.y1 = y / dy + v0


    def control(self, action):
        # global integralx, integraly, lasterrorx, lasterrory
        # errorx = self.x1 - self.x0
        # errory = self.y1 - self.y0
        # self.integralx += errorx
        # self.integraly += errory
        # Vx = self.Kp * errorx + self.Ki * self.integralx + self.Kd * (errorx - self.lasterrorx)
        # Vy = self.Kp * errory + self.Ki * self.integraly + self.Kd * (errory - self.lasterrory)
        self.anglex = action[0]
        self.angley = action[1]
        self.xt = self.transform(self.anglex)
        self.yt = self.transform(self.angley)
        # self.lasterrorx, self.lasterrory = errorx, errory

    def Kalman_Filter(self, Z):
        # Prediction
        ux_pred = self.F.dot(self.ux)
        P_pred = self.F.dot(self.P).dot(self.F.T) + self.Q
        # Updateself.
        S = self.H.dot(P_pred).dot(self.H.T) + self.R
        K = P_pred.dot(self.H.T).dot(np.linalg.inv(S))
        ux_update = ux_pred + K.dot(Z - self.H.dot(ux_pred))
        P_update = P_pred - K.dot(self.H).dot(P_pred)
        self.ux = ux_update
        self.P = P_update
        # return ux_update, P_update

    def deg2rad(self, deg):
        return deg * np.pi / 180

    def transform(self, angle):
        radians = self.deg2rad(angle)
        return self.Height * np.tan(radians)