import numpy as np
import matplotlib.pyplot as plt

# 初始化变量
x0, y0 = 320, 240
nx, ny = 0, 0
h = 100
x3, y3 = 10, 10
vx, vy = 5, 5
t = 0
anglex, angley = 0, 0
p, k, id = 0.02, 0.01, 0.01
xt, yt = 0, 0
xt1, yt1 = x3, y3
integralx, integraly = 0, 0
lasterrorx, lasterrory = 0, 0
iterations = 100
t_plot = 0.05  # 每次迭代的时间间隔

# 数据存储列表
Coordx_R = []
Coordy_R = []
Coordx_O = []
Coordy_O = []
Coordx_F = []
Coordy_F = []
outx_values = []
outy_values = []
time_values = []

# 卡尔曼滤波初始化参数
ux = np.array([[xt1], [0], [yt1], [0]])
Q = np.diag([0.001, 0.0001, 0.001, 0.0001])
R = np.diag([2, 2])
F = np.array([[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]])
H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
P = np.eye(4)


def deg2rad(deg):
    return deg * np.pi / 180


def CameraCoord(xw, yw, zw, theta, alpha):
    theta = deg2rad(theta)
    alpha = deg2rad(alpha)
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

    global x1, y1
    x1 = x / dx + u0
    y1 = y / dy + v0


def Control_servo(axis, value):
    global nx, ny, anglex, angley
    if axis == 'x':
        nx += value
        anglex = max(min(nx, 40), -40)
    elif axis == 'y':
        ny += value
        angley = max(min(ny, 40), -40)


def control():
    global integralx, integraly, lasterrorx, lasterrory
    errorx = x1 - x0
    errory = y1 - y0
    integralx += errorx
    integraly += errory
    Vx = p * errorx + id * integralx + k * (errorx - lasterrorx)
    Vy = p * errory + id * integraly + k * (errory - lasterrory)
    Control_servo('x', Vx)
    Control_servo('y', Vy)
    lasterrorx, lasterrory = errorx, errory


def Kalman_Filter(ux, P, Z, Q, R, F, H):
    # Prediction
    ux_pred = F.dot(ux)
    P_pred = F.dot(P).dot(F.T) + Q
    # Update
    S = H.dot(P_pred).dot(H.T) + R
    K = P_pred.dot(H.T).dot(np.linalg.inv(S))
    ux_update = ux_pred + K.dot(Z - H.dot(ux_pred))
    P_update = P_pred - K.dot(H).dot(P_pred)
    return ux_update, P_update


# 主循环
for i in range(iterations):
    CameraCoord(xt1, yt1, h, anglex, angley)
    control()
    xt1 = x3 + vx * t
    yt1 = y3 + vy * t
    Coordx_R.append(xt1)
    Coordy_R.append(yt1)

    # 添加观测噪声
    xt1_O = xt1 + np.random.normal(0, 2)
    yt1_O = yt1 + np.random.normal(0, 2)
    Coordx_O.append(xt1_O)
    Coordy_O.append(yt1_O)

    Z = np.array([[xt1_O], [yt1_O]])
    ux, P = Kalman_Filter(ux, P, Z, Q, R, F, H)
    xt1_F, yt1_F = ux[0, 0], ux[2, 0]
    Coordx_F.append(xt1_F)
    Coordy_F.append(yt1_F)

    outx_values.append(xt1 - xt)
    outy_values.append(yt1 - yt)
    time_values.append(i * t_plot)
    t += t_plot

# 绘制路径和误差图
plt.figure()
plt.plot(Coordx_R, Coordy_R, 'r-', label='Real Path')
plt.plot(Coordx_O, Coordy_O, 'g.', label='Observed Path')
plt.plot(Coordx_F, Coordy_F, 'b-', label='Filtered Path')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Paths Comparison')
plt.legend()

plt.figure()
plt.plot(time_values, outx_values, 'r-', label='Out X')
plt.plot(time_values, outy_values, 'b-', label='Out Y')
plt.xlabel('Time')
plt.ylabel('Out Values')
plt.title('Out X and Out Y over Time')
plt.legend()

plt.show()
