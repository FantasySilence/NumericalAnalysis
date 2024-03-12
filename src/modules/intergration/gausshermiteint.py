import numpy as np
import sympy as sp
import matplotlib
import math

matplotlib.rcParams['font.sans-serif'] = ['STSong']
matplotlib.rcParams['axes.unicode_minus'] = False


class GuassHermiteIntergration:

    """
    高斯——埃尔米特求积公式
    """

    def __init__(self, int_fun, zeros_num=10):
        """
        必要的参数初始化
        int_fun: 被积函数
        zeros_num: 正交多项式零点个数
        """

        self.int_fun = int_fun  # 被积函数
        self.n = int(zeros_num)  # 正交多项式的零点数
        self.zero_points = None  # 埃尔米特高斯零点
        self.int_value = None  # 积分值结果
        self.A_k = None  # 求积系数

    def cal_int(self):
        """
        高斯——埃尔米特求积公式：求解零点和Ak系数
        """

        t = sp.symbols('t')
        p_n = (-1) ** self.n * sp.exp(t ** 2) * sp.diff(sp.exp(-t ** 2), t, self.n)
        self.zero_points = np.asarray(sp.solve(p_n, t), dtype=np.float64)
        Ak_poly = 2 ** (self.n + 1) * math.factorial(self.n) * np.sqrt(np.pi) / (sp.diff(p_n, t, 1)) ** 2
        self.A_k = sp.lambdify(t, Ak_poly)(self.zero_points)
        f_val = self.int_fun(self.zero_points) * np.exp(self.zero_points ** 2)
        self.int_value = np.dot(f_val, self.A_k)
        return self.int_value
