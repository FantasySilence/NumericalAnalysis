import matplotlib
import numpy as np
import sympy as sp
import pandas as pd

# 绘图设置，可以显示中文
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

from src.common.utilsinterpolation.Interp_utils import cal_interp
from src.common.utilsinterpolation.Interp_utils import plt_interp


class NewtonInterpolation:

    """
    牛顿插值
    """

    def __init__(self, x, y):

        """
        牛顿插值必要的参数初始化
        x:已知数据的x坐标
        y:已知数据的y坐标
        """

        self.x = np.asarray(x, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        if len(self.x) > 1 and len(self.x) == len(self.y):
            self.n = len(self.x)
        else:
            raise ValueError("x,y坐标长度不匹配")
        self.polynomial = None  # 最终的插值多项式的符号表示
        self.poly_coefficient = None  # 最终的插值多项式的系数，幂次从高到低
        self.coefficient_order = None  # 对应多项式系数的阶次
        self.y0 = None  # 所求插值点的值，单个值或者向量
        self.diff_quot = None  # 储存离散数据点的差商

    def __diff_quotient__(self):

        """
        计算牛顿均差（差商）
        """

        diff_quot = np.zeros((self.n, self.n))
        diff_quot[:, 0] = self.y  # 第一列存储y值
        for j in range(1, self.n):  # 按列计算
            for i in range(j, self.n):  # 行，初始值为差商表的对角线值
                diff_quot[i, j] = (diff_quot[i, j - 1] - diff_quot[i - 1, j - 1]) / (self.x[i] - self.x[i - j])
        self.diff_quot = pd.DataFrame(diff_quot)
        return diff_quot

    def fit_interp(self):
        """
        生成牛顿插值多项式
        """
        t = sp.symbols('t')  # 定义符号变量
        diff_quot = self.__diff_quotient__()  # 计算差商表
        d_q = np.diag(diff_quot)  # 构造牛顿插值时只需要对角线元素
        self.polynomial = d_q[0]
        term_poly = t - self.x[0]
        for i in range(1, self.n):
            self.polynomial += d_q[i] * term_poly
            term_poly *= (t - self.x[i])

        # 插值多项式特征
        self.polynomial = sp.expand(self.polynomial)  # 多项式展开
        polynomial = sp.Poly(self.polynomial, t)  # 构造多项式对象
        self.poly_coefficient = polynomial.coeffs()  # 获取多项式的系数
        self.coefficient_order = polynomial.monoms()  # 多项式系数对应的阶次

    def cal_interp(self, x0):

        """
        计算给定插值点的数值
        x0:所求插值的x坐标值
        """

        self.y0 = cal_interp(self.polynomial, x0)
        return self.y0

    def plt_interp(self, x0=None, y0=None):

        """
        绘制牛顿插值曲线
        x0:所求插值的x坐标值
        y0:所求插值的y坐标值
        """

        params = (self.polynomial, self.x, self.y, 'Newton', x0, y0)
        plt_interp(params)
