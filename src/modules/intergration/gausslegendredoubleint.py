import numpy as np
import sympy as sp
import matplotlib
import math

matplotlib.rcParams['font.sans-serif'] = ['STSong']
matplotlib.rcParams['axes.unicode_minus'] = False


class GuassLegendreDuobleIntergration:

    """
    高斯——勒让德二重积分
    """

    def __init__(self, int_fun, x_span, y_span, zeros_num=10):

        """
        必要的参数初始化
        int_fun: 被积函数
        x_span: x的积分区间
        y_span: y的积分区间
        zeros_num: 高斯零点数量，默认为10
        """

        self.int_fun = int_fun  # 被积函数
        self.ax, self.bx = x_span[0], x_span[1]  # x的积分上下限
        self.ay, self.by = y_span[0], y_span[1]  # y的积分上下限
        self.n = zeros_num
        self.int_value = None  # 最终积分值

    def cal_2d_int(self):

        """
        高斯——勒让德二重积分计算
        1.计算勒让德零点
        2.求插值系数Ak
        3.做积分区间变换，[a,b]——>[-1,1]
        4.生成网格点，计算被积函数的函数值
        5.根据公式构造二重积分值
        """

        A_k, zeros_points = self.__cal_Ak_zeros__()  # 计算高斯零点和Ak系数
        # 积分区间变换
        A_k_x = A_k * (self.bx - self.ax) / 2
        A_k_y = A_k * (self.by - self.ay) / 2
        zeros_points_x = (self.bx - self.ax) / 2 * zeros_points + (self.ax + self.bx) / 2
        zeros_points_y = (self.by - self.ay) / 2 * zeros_points + (self.ay + self.by) / 2
        xy = np.meshgrid(zeros_points_x, zeros_points_y)
        f_val = self.int_fun(xy[0], xy[1])
        self.int_value = 0.0
        for i in range(self.n):
            for j in range(self.n):
                self.int_value += A_k_x[i] * A_k_y[j] * f_val[i, j]
        return self.int_value

    def __cal_Ak_zeros__(self):

        """
        计算勒让德零点和Ak系数
        """

        t = sp.Symbol('t')
        # 勒让德多项式构造
        p_n = (t ** 2 - 1) ** self.n / math.factorial(self.n) / 2 ** self.n
        diff_p_n = sp.diff(p_n, t, self.n)  # 多项式的n阶导数
        # 求解多项式的全部零点，需要更加优秀的算法替代
        zero_points = np.asarray(sp.solve(diff_p_n, t), dtype=np.float64)
        Ak_poly = sp.lambdify(t, 2 / (1 - t ** 2) / (diff_p_n.diff(t, 1) ** 2))
        A_k = Ak_poly(zero_points)  # 求解Ak系数
        return A_k, zero_points
