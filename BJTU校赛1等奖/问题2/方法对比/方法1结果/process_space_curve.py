import pandas as pd
import math
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from preprocess import *
from scipy.interpolate import CubicSpline
from scipy.integrate import solve_ivp


def process_space_curve(data_path, image_save_path=None, points_save_path=None):
    # 数据预处理
    datas = pd.read_csv(data_path)
    datas["theta"] = datas.apply(lambda row: getTheta(row["曲率ka"], row["曲率kb"], row["曲率kc"]), axis=1)
    datas["k"] = datas.apply(lambda row: getSumk(row["曲率ka"], row["曲率kb"], row["曲率kc"]), axis=1)

    # 三次样条插值
    k_s = CubicSpline(datas["弧长"], datas["k"], bc_type='natural')
    theta_s = CubicSpline(datas["弧长"], datas["theta"], bc_type='natural')
    tue_s = theta_s.derivative()

    # 定义微分方程
    # 定义Frenet - Serret公式对应的微分方程组
    def frenet_serret(s, y):
        T, N, B = y[:3], y[3:6], y[6:]
        k_value = k_s(s)
        tau_value = tue_s(s)
        dT_ds = k_value * N
        dN_ds = -k_value * T + tau_value * B
        dB_ds = -tau_value * N
        return np.concatenate([dT_ds, dN_ds, dB_ds])

    # 初始边界条件
    T0 = np.array([1, 0, 0])  # 初始单位切向量
    N0 = np.array([0, -1, 0])  # 初始单位法向量
    B0 = np.array([0, 0, 1])  # 初始单位副法向量
    y0 = np.concatenate([T0, N0, B0])

    # 定义弧长范围
    s_span = (0, 125)  # 示例范围，可根据实际调整
    s_eval = np.linspace(s_span[0], s_span[1], 10000)  # 用于计算的弧长离散点

    # 求解微分方程组
    sol = solve_ivp(frenet_serret, s_span, y0, t_eval=s_eval)

    # 得到单位切向量T(s)
    T_s = sol.y[:3, :].T

    # 对T(s)进行数值积分求r(s)
    r0 = np.array([0, 0, 0])  # 起始位置
    r_s = np.cumsum(T_s, axis=0) * (s_eval[1] - s_eval[0]) + r0  # 数值积分近似，这里用简单的累积求和近似积分

    # 绘制r(s)的图像
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(r_s[:, 0], r_s[:, 1], r_s[:, 2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Space Curve r(s)')

    # 保存图像
    if image_save_path is not None:
        plt.savefig(image_save_path, dpi=300)

    plt.show()

    if points_save_path is not None:
        points = pd.DataFrame({"x": r_s[:, 0], "y": r_s[:, 1], "z": r_s[:, 2]})
        points.to_csv(points_save_path,index=False)

    return r_s


if __name__ == "__main__":
    data_path = "数据/问题2-曲线2.csv"
    image_save_path = None
    points_save_path = "问题2/方法对比/方法1结果/插值10000/points.csv"
    process_space_curve(data_path, image_save_path, points_save_path)
