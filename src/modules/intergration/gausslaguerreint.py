import numpy as np
import sympy as sp
import matplotlib
import math

matplotlib.rcParams['font.sans-serif'] = ['STSong']
matplotlib.rcParams['axes.unicode_minus'] = False



class GuassLaguerreIntergration:
    
    """
    高斯——拉盖尔积分公式：积分区间[a, +∞)
    """

    def __init__(self, int_fun, int_internal, zeros_num=10):
        
        """
        必要的参数初始化
        int_fun: 被积函数
        int_internal: 积分区间
        zeros_num: 正交多项式零点个数
        """
        
        self.int_fun = int_fun      # 被积函数
        if int_internal[1] is not np.infty:
            raise ValueError("积分区间参数设置不规范，应为[a,+∞)！")
        self.a = int_internal[0]   # 积分区间左端点
        self.n = int(zeros_num)+1       # 正交多项式的零点数
        self.zero_points = None         # 拉盖尔高斯零点
        self.int_value = None            # 积分值结果
        self.A_k = None        # 求积系数

    def cal_int(self):

        """
        高斯——拉盖尔求积公式
        """

        t = sp.Symbol('t')
        p_n = sp.exp(t)*sp.diff(t**self.n*sp.exp(-t), t, self.n)
        self.zero_points = np.asarray(sp.solve(p_n, t), dtype=np.float64)
        Ak_poly = sp.lambdify(t, math.factorial(self.n)**2/(t*sp.diff(p_n, t, 1)**2))
        self.A_k = Ak_poly(self.zero_points)
        # 区间转换
        f_val = self.int_fun(self.zero_points+self.a)*np.exp(self.zero_points)
        self.int_value = np.dot(f_val, self.A_k)
        return self.int_value