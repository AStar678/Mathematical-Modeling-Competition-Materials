import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from numpy import pi
from scipy.interpolate import CubicSpline
from preprocess import *
from tqdm import tqdm


class CurveProcessor:
    def __init__(self, data_path, image_save_path=None, points_save_path=None):
        self.data_path = data_path
        self.image_save_path = image_save_path
        self.points_save_path = points_save_path
        self.ordatas = pd.read_csv(self.data_path)
        self.process_data()

    def calculate_dx(self, k):
        return 0

    def calculate_dy(self, k):
        if k != 0:
            alpha_k = self.datas.iloc[k - 1]["alpha"]
            ka_k = self.datas.iloc[k - 1]["k"]
            return (1 - np.cos(alpha_k)) / ka_k
        return 0

    def calculate_dz(self, k):
        if k != 0:
            alpha_k = self.datas.iloc[k - 1]["alpha"]
            ka_k = self.datas.iloc[k - 1]["k"]
            return np.sin(alpha_k) / ka_k
        return 0

    def getTue(self, k):
        if k > 1:
            theta_k = self.datas.iloc[k - 1]["theta"]
            alpha_k = self.datas.iloc[k - 1]["alpha"]
        else:
            theta_k = 0
            alpha_k = 0
        cos_theta_k = np.cos(theta_k)
        sin_theta_k = np.sin(theta_k)
        cos_alpha_k = np.cos(alpha_k)
        sin_alpha_k = np.sin(alpha_k)

        dx = self.calculate_dx(k - 1)
        dy = self.calculate_dy(k - 1)
        dz = self.calculate_dz(k - 1)

        tue = np.array([
            [cos_theta_k, -sin_theta_k, 0, -dx],
            [cos_alpha_k * sin_theta_k, cos_alpha_k * cos_theta_k, sin_alpha_k, -dy],
            [-sin_alpha_k * sin_theta_k, -sin_alpha_k * cos_theta_k, cos_alpha_k, -dz],
            [0, 0, 0, 1]
        ])
        return tue

    def get_transformation_matrix1(self, theta_i):
        cos_theta = np.cos(theta_i)
        sin_theta = np.sin(theta_i)
        matrix = np.array([
            [cos_theta, 0, -sin_theta],
            [0, 1, 0],
            [sin_theta, 0, cos_theta],
        ])
        return matrix

    def get_rotation_matrix2(self, phi_i, phi_i_plus_1):
        if phi_i_plus_1 >= phi_i:
            delta_phi = phi_i_plus_1 - phi_i
        else:
            delta_phi = 2 * np.pi + phi_i_plus_1 - phi_i
        cos_delta_phi = np.cos(delta_phi)
        sin_delta_phi = np.sin(delta_phi)
        matrix = np.array([
            [cos_delta_phi, -sin_delta_phi, 0],
            [sin_delta_phi, cos_delta_phi, 0],
            [0, 0, 1]
        ])
        return matrix

    def get_p(self, theta_i, k_i):
        p_x = -(1 - np.cos(theta_i)) / k_i
        p_y = 0
        p_z = np.sin(theta_i) / k_i
        p = np.array([p_x, p_y, p_z]).reshape(-1, 1)
        return p

    def get_Ti1(self, i):
        data = self.datas.iloc[i]
        dataminus = self.datas.iloc[i - 1]
        this_theta = data["ds"] * data["k"]
        T = self.get_transformation_matrix1(this_theta).dot(
            self.get_rotation_matrix2(phi_i=dataminus["theta"], phi_i_plus_1=data["theta"]))
        return T

    def get_t(self, i):
        T = self.get_Ti1(i)
        this_theta = self.datas.iloc[i]["ds"] * self.datas.iloc[i]["k"]
        p = self.get_p(this_theta, self.datas.iloc[i]["k"])
        bottom_row = np.array([[0, 0, 0, 1]])
        t = np.vstack((np.hstack((T, p)), bottom_row))
        return t

    def getMulT(self, i):
        bas = self.get_t(1)
        for n in range(2, i):
            bas = bas.dot(self.get_t(n))
        return bas

    def getB(self, k):
        if k == 0:
            B = np.array([[0], [0], [0], [0]])
            return B
        alpha_k = self.datas.iloc[k - 1]["alpha"]
        ka_k = self.datas.iloc[k - 1]["k"]
        B = np.array([[0],
                      [(1 - np.cos(alpha_k)) / ka_k],
                      [np.sin(alpha_k) / ka_k],
                      [1]])
        return B

    def getA(self, k):
        if k == 0:
            A = np.array([[0], [0], [0], [0]])
            return A
        B = self.getB(k)
        rT = self.getTue(k).dot(self.getMulT(k))
        A = (np.linalg.inv(rT)).dot(B)
        return A

    def process_data(self):
        s = self.ordatas["弧长"]
        ka = self.ordatas["曲率ka"]
        kb = self.ordatas["曲率kb"]
        kc = self.ordatas["曲率kc"]

        new_s = np.linspace(s.min(), s.max(), 10000)
        cs = CubicSpline(s, ka)
        new_ka = cs(new_s)

        cs = CubicSpline(s, kb)
        new_kb = cs(new_s)

        cs = CubicSpline(s, kc)
        new_kc = cs(new_s)

        ds = np.diff(new_s)
        ds = np.append(ds, ds[0])

        self.datas = pd.DataFrame({"ds": ds, "ka": new_ka, "kb": new_kb, "kc": new_kc})
        self.datas["theta"] = self.datas.apply(lambda row: getTheta(row["ka"], row["kb"], row["kc"]), axis=1)
        self.datas["k"] = self.datas.apply(lambda row: getSumk(row["ka"], row["kb"], row["kc"]), axis=1)
        self.datas["alpha"] = self.datas["k"] * ds

    def plot_curve(self):
        x_list = []
        y_list = []
        z_list = []
        for i in tqdm(range(1, len(self.datas)), desc="Processing points"):
            A = self.getA(i)
            xyz = A.T.tolist()[0][:3]
            x = xyz[0]
            y = xyz[1]
            z = xyz[2]

            x_list.append(x)
            y_list.append(y)
            z_list.append(z)

        location = pd.DataFrame({"x": x_list, "y": y_list, "z": z_list})


        x = location['x'].values
        y = location['y'].values
        z = location['z'].values

        t = np.linspace(0, len(x) - 1, len(x))
        new_t = np.linspace(0, len(x) - 1, 100)

        cs_x = CubicSpline(t, x)
        cs_y = CubicSpline(t, y)
        cs_z = CubicSpline(t, z)

        new_x = cs_x(new_t)
        new_y = cs_y(new_t)
        new_z = cs_z(new_t)

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(new_x, new_y, new_z, linewidth=0.5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        if self.image_save_path is not None:
            plt.savefig(self.image_save_path, dpi=300)
        
        plt.show()

        if self.points_save_path is not None:
            location.to_csv(self.points_save_path, index=False)


if __name__ == "__main__":
    data_path = "数据/问题2-曲线2.csv"
    image_save_path = None
    points_save_path = "问题2/方法对比/方法2结果/插值10000/points.csv"
    processor = CurveProcessor(data_path, image_save_path, points_save_path)
    processor.plot_curve()
    