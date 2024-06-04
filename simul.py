import numpy as np
import matplotlib.pyplot as plt

# 初始化变量
global x0, y0, x1, y1, xt1, yt1, nx, ny, h, x3, y3, vx, vy, vz, t0, t, anglex, angley, tx, outx, outy, xt, yt, count1, lasterrorx, lasterrory, p, k, id, integralx, integraly

h = 100  # 现实世界坐标下无人机的高度（坐标中心为相机位置）
x3, y3 = 50, 50  # 现实世界坐标下无人机的x，y方向坐标
vx, vy, vz = 10, 10, 0  # 无人机在x，y方向的移动速度
xt1, yt1 = 0, 0  # 无人机的经过时间t后的位置
t0 = 0.05  # 时间，每次循环相当于现实中耗时0.05s
xt, yt = 0, 0  # 相机中心点指向的位置在现实坐标系下的坐标
outx, outy = 0, 0  # 相机指向的位置与无人机位置的差值
t = 0
nx, ny = 0, 0  # 舵机旋转角度=anglex和angley
x0, y0 = 320, 240  # 图像坐标系中原点的坐标
x1, y1 = 320, 240  # 图像坐标系中无人机坐标
anglex, angley = 0, 0
p, k, id = 0.08, 0.01, 0.15
tx = -1

count1 = 0
integralx, integraly = 0, 0
lasterrorx, lasterrory = 0, 0

# 数据存储
outx_values, outy_values, time_values = [], [], []
iterations = 100  # 设置迭代次数

def deg2rad(deg):
    return deg * np.pi / 180

def CameraCoord(xw, yw, zw, theta, alpha):
    global x1, y1
    theta = deg2rad(theta)
    alpha = deg2rad(alpha)
    f = 1.88e-3  # 焦距
    dx, dy = 3.14e-6, 3.75e-6  # 每个像素在 x，y轴方向上的物理尺寸
    u0, v0 = 320, 240  # 像平面原点在像素平面的位置

    # 转换为相机坐标系(xc, yc, zc)
    x_temp = xw * np.cos(theta) - zw * np.sin(theta)
    y_temp = yw
    z_temp = xw * np.sin(theta) + zw * np.cos(theta)

    # alpha
    xc = x_temp
    yc = y_temp * np.cos(alpha) - zw * np.sin(alpha)
    zc = y_temp * np.sin(alpha) + z_temp * np.cos(alpha)

    # 转换为像平面坐标系（x,y）
    x = (f * xc) / zc
    y = (f * yc) / zc

    # 转换为图像像素点坐标系
    x1 = x / dx + u0
    y1 = y / dy + v0
    print('x1= {:.2f}'.format(x1))
    print('y1= {:.2f}'.format(y1))

def transform(angle):
    global h
    radians = deg2rad(angle)
    return h * np.tan(radians)

def Control_servo(axis, V):
    global anglex, angley, nx, ny
    if axis == 'x':
        nx += V
        if nx > 80:
            anglex = 80
        elif nx < -80:
            anglex = -80
        else:
            anglex = nx
        print('nx= {:.2f}'.format(nx))
        print('anglex= {:.2f}'.format(anglex))
    elif axis == 'y':
        ny += V
        if ny > 80:
            angley = 80
        elif ny < -80:
            angley = -80
        else:
            angley = ny
        print('ny= {:.2f}'.format(ny))
        print('angley= {:.2f}'.format(angley))

def control():
    global x0, y0, x1, y1, anglex, angley, xt, yt, lasterrorx, lasterrory, p, id, k, integralx, integraly

    errorx, errory = x1 - x0, y1 - y0

    # 计算误差的变化量（微分项）
    differentx, differenty = errorx - lasterrorx, errory - lasterrory

    # 更新积分项
    integralx += errorx
    integraly += errory

    # 计算控制速度（加入积分项）
    Vx = p * errorx + id * integralx + k * differentx
    Vy = p * errory + id * integraly + k * differenty

    print('integralx= {:.2f}'.format(integralx))
    print('errorx = {:.2f}'.format(errorx))
    print('integraly= {:.2f}'.format(integraly))
    print('errory= {:.2f}'.format(errory))
    print('Vx= {:.2f}'.format(Vx))
    print('Vy= {:.2f}'.format(Vy))

    # 控制伺服电机
    Control_servo('x', Vx)
    Control_servo('y', Vy)

    # 更新上一次的误差值
    lasterrorx, lasterrory = errorx, errory

    # 执行任何需要的变换
    xt = transform(anglex)
    yt = transform(angley)

    print('xt= {:.2f}'.format(xt))
    print('yt= {:.2f}'.format(yt))

# 主循环
for i in range(iterations):
    CameraCoord(xt1, yt1, h, anglex, angley)  # 将真实世界坐标转换为图像坐标
    control()  # 控制舵机转动

    xt1 = x3 + vx * t  # 无人机飞行
    yt1 = y3 + vy * t
    outx = xt1 - xt
    outy = yt1 - yt

    # 存储数据
    outx_values.append(outx)
    outy_values.append(outy)
    time_values.append(count1 * t0)  # 存储当前时间
    count1 += 1
    t = count1 * t0

# 绘图
plt.figure()
plt.plot(time_values, outx_values, 'r', label='outx')  # 使用红色表示 outx
plt.plot(time_values, outy_values, 'b', label='outy')  # 使用蓝色表示 outy
plt.xlabel('Time (count1*t)')
plt.ylabel('Value')
plt.title('outx and outy over Time')
plt.legend()
plt.grid(True)
plt.show()
