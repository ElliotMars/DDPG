import gym
from gym import spaces
import numpy as np
import math
import random
import csv

from typing import Optional

BINGO_RANGE = 10
ROU = 0.5
AMPLIFY = 1
PQR_HIGH = 100
PQR_LOW = 0
TOP_DIS = 19517
TOP_AIM = 100

from Kalman.Kalman_agent import STEP_LEN, P0, Q0, R0

class KalmanEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, ):
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

        #每段循环次数
        self.iterations = 100

        # PID参数
        self.Kp = 0.1
        self.Ki = 0.1
        self.Kd = 0.1

        self.F = np.array([[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]])
        self.H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])

        self.xt1 = None
        self.yt1 = None

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:影响Actor Net | step | train random action OU噪声
        self.action_space = spaces.Box(low=np.array([-1, -1, -1]),
                                       high=np.array([1, 1, 1]), shape=(3,), dtype=np.float64)
        # Example for using image as input:影响step
        self.observation_space = spaces.Box(low=np.array([0, 0, 0, 0, 0]),
                                            high=np.array([1, 1, 1, 1, 1]),
                                            shape=(5,), dtype=np.float64)

    def step(self, action):
        self.step_count += 1
        self.aim_times = 0
        self.distance = np.empty(shape=self.iterations)

        if self.step_count == 1:
            self.Initial()

        p_new = min(max(self.P[0][0] + action[0], PQR_LOW), PQR_HIGH)
        q_new = min(max(self.Q[0][0] + action[1], PQR_LOW), PQR_HIGH)
        r_new = min(max(self.R[0][0] + action[2], PQR_LOW), PQR_HIGH)

        self.P = np.diag([p_new, p_new, p_new, p_new])
        self.Q = np.diag([q_new, q_new, q_new, q_new])
        self.R = np.diag([r_new, r_new])

        #print(self.P[0][0], self.Q[0][0], self.R[0][0])

        for i in range(self.iterations):


            self.xt1 = self.trajectoryx[(self.step_count - 1) * self.iterations + i]
            self.yt1 = self.trajectoryy[(self.step_count - 1) * self.iterations + i]
            #self.Coordx_R.append(self.xt1)
            #self.Coordy_R.append(self.yt1)

            self.CameraCoord()

            # 添加观测噪声
            self.x1_O = self.x1 + np.random.normal(0, 2)
            self.y1_O = self.y1 + np.random.normal(0, 2)
            #self.Coordx_O.append(self.xt1_O)
            #self.Coordy_O.append(self.yt1_O)

            Z = np.array([[self.x1_O], [self.y1_O]])
            self.Kalman_Filter(Z)
            self.x1_F, self.y1_F = self.ux[0, 0], self.ux[2, 0]
            print(self.ux)
            #self.x1_F, self.y1_F = self.x1_O, self.y1_O

            self.control()
            #self.Coordx_F.append(self.xt1_F)
            #self.Coordy_F.append(self.yt1_F)

            #self.outx_values.append(self.xt1 - self.xt)
            #self.outy_values.append(self.yt1 - self.yt)
            #self.time_values.append(i * self.t_plot)
            self.t += self.t_plot

            if self.step_count == 6:
                self.step_count = 0

            #print('xt = ', self.xt, ' yt = ', self.yt, 'xt1 = ', self.xt1, 'yt1 = ', self.yt1)

            # if (abs(self.xt1 - self.xt) <= BINGO_RANGE and abs(self.yt1 - self.yt) <= BINGO_RANGE and self.tr_notdone):
            #     self.tr = self.t
            #     self.tr_notdone = False
            if (abs(self.xt1 - self.xt) <= BINGO_RANGE and abs(self.yt1 - self.yt) <= BINGO_RANGE):
                self.aim_times += 1
            self.distance[i] = abs(self.xt1 - self.xt) + abs(self.yt1 - self.yt)

        observation = self.get_obs()
        #print('observation = ', observation)
        reward = self.get_RE()  # TODO
        #print('reward = ',reward)
        done = False
        truncation = False
        info = {}

        return observation, reward, done, truncation, info

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):

        # 继承/初始Kalman
        self.P = P0
        self.Q = Q0
        self.R = R0

        self.step_count = 0

        trajectory = random.randint(1, 3)
        # trajectory = 3
        # 获取当前工作目录

        if trajectory == 1:
            print('diagonal')
            self.trajectoryx = []
            self.trajectoryy = []
            # 打开csv文件
            with open('../UAVroute/diagonal.csv', newline='') as csvfile:
                # 创建csv文件阅读器
                csvreader = csv.reader(csvfile)
                # 跳过标题行（如果有的话）
                # next(csvreader)  # 如果第一行是标题，可以跳过

                # 逐行读取csv文件内容
                for row in csvreader:
                    # 将数据存入相应的列表
                    self.trajectoryx.append(float(row[0]))  # 第一列数据，转为浮点数并存入列表
                    self.trajectoryy.append(float(row[1]))  # 第二列数据，转为浮点数并存入列表

        elif trajectory == 2:
            print('square')
            self.trajectoryx = []
            self.trajectoryy = []
            # 打开csv文件
            with open('../UAVroute/square.csv', newline='') as csvfile:
                # 创建csv文件阅读器
                csvreader = csv.reader(csvfile)
                # 跳过标题行（如果有的话）
                # next(csvreader)  # 如果第一行是标题，可以跳过

                # 逐行读取csv文件内容
                for row in csvreader:
                    # 将数据存入相应的列表
                    self.trajectoryx.append(float(row[0]))  # 第一列数据，转为浮点数并存入列表
                    self.trajectoryy.append(float(row[1]))  # 第二列数据，转为浮点数并存入列表

        else:
            print('lemniscate')
            self.trajectoryx = []
            self.trajectoryy = []
            # 打开csv文件
            with open('../UAVroute/lemniscate.csv', newline='') as csvfile:
                # 创建csv文件阅读器
                csvreader = csv.reader(csvfile)
                # 跳过标题行（如果有的话）
                # next(csvreader)  # 如果第一行是标题，可以跳过

                # 逐行读取csv文件内容
                for row in csvreader:
                    # 将数据存入相应的列表
                    self.trajectoryx.append(float(row[0]))  # 第一列数据，转为浮点数并存入列表
                    self.trajectoryy.append(float(row[1]))  # 第二列数据，转为浮点数并存入列表

        self.t = 0

        #self.tr = 0
        self.aim_times = 0
        self.distance = np.zeros(shape=self.iterations)


        observation = self.get_obs()

        return observation  # reward, done, info can't be included

    # def render(self, mode='human'):
    #     ...

    def Initial(self):
        # self.tr = 0
        # self.tr_notdone = True
        self.aim_times = 0
        self.distance = np.empty(shape=self.iterations)

        # 无人机初始位置
        self.x3 = self.trajectoryx[0]
        self.y3 = self.trajectoryy[0]

        # 起始时间
        self.t = 0

        # 舵机初始角度
        self.anglex = 0
        self.angley = 0
        # 初始摄像头指向坐标
        self.xt = 0
        self.yt = 0

        # 观测坐标
        self.xt1_O = 0
        self.yt1_O = 0

        # 滤波后坐标
        self.xt1_F = 0
        self.yt1_F = 0

        self.nx = 0
        self.ny = 0

        # 无人机移动坐标初始化
        self.xt1 = self.x3
        self.yt1 = self.y3

        # 积分项
        self.integralx = 0
        self.integraly = 0

        # 上次偏差
        self.lasterrorx = 0
        self.lasterrory = 0

        # 输出值
        #self.outx_values = 0
        #self.outy_values = 0

        # 数据存储列表
        #self.Coordx_R = []
        #self.Coordy_R = []
        #self.Coordx_O = []
        #self.Coordy_O = []
        #self.Coordx_F = []
        #self.Coordy_F = []
        #self.outx_values = []
        #self.outy_values = []
        #self.time_values = []
        # 像素坐标系坐标
        self.CameraCoord()

        # 卡尔曼滤波初始化参数
        self.ux = np.array([[self.x1], [0], [self.y1], [0]])

    def close(self):
        self.isopen = False

    def get_obs(self):
        obs = np.array([np.sum(self.distance) / TOP_DIS, self.aim_times / TOP_AIM, self.P[0][0] / PQR_HIGH,
                        self.Q[0][0] / PQR_HIGH, self.R[0][0] / PQR_HIGH], dtype=np.float64)
        return obs

    def get_RE(self):
        # c_tr = 12
        # c_d = ROU
        # c_taim = 1 - ROU
        # re1 = -c_tr * self.tr
        # re2 = -c_d * np.sum(self.distance)
        # re3 = c_taim * self.aim_times
        # reward = re1 + re2 + re3
        # reward = AMPLIFY * (re2 + re3)
        # re3 = self.aim_times * self.t_plot
        re1 = -ROU * np.sum(self.distance) / TOP_DIS
        re2 = (1 - ROU) * self.aim_times / TOP_AIM
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

    def Control_servo(self, axis, value):

        if axis == 'x':
            self.nx += value
            self.anglex = max(min(self.nx, 30), -30)
        elif axis == 'y':
            self.ny += value
            self.angley = max(min(self.ny, 30), -30)

    def control(self):
        # global integralx, integraly, lasterrorx, lasterrory
        errorx = self.x1 - self.x0
        errory = self.y1 - self.y0
        self.integralx += errorx
        self.integraly += errory
        Vx = self.Kp * errorx + self.Ki * self.integralx + self.Kd * (errorx - self.lasterrorx)
        Vy = self.Kp * errory + self.Ki * self.integraly + self.Kd * (errory - self.lasterrory)
        self.Control_servo('x', Vx)
        self.Control_servo('y', Vy)
        self.xt = self.transform(self.anglex)
        self.yt = self.transform(self.angley)
        self.lasterrorx, self.lasterrory = errorx, errory

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