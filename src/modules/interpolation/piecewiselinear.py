import matplotlib
import numpy as np
import sympy as sp

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

from src.common.utilsinterpolation.piecewise_utils import cal_interp
from src.common.utilsinterpolation.piecewise_utils import plt_interp


class PiecewiseLinearInterpolation:

    """
    分段线性插值
    """

    def __init__(self, x, y):

        """
        分段线性插值必要的参数初始化
        x:已知数据的x坐标
        y:已知数据的y坐标
        """

        self.x = np.asarray(x, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        if len(self.x) > 1 and len(self.x) == len(self.y):
            self.n = len(self.x)
        else:
            raise ValueError("x,y坐标长度不匹配")
        self.linear_poly = None  # 分段线性插值函数
        self.linear_coefficient = None  # 最终的插值多项式的系数，幂次从高到低
        self.y0 = None  # 所求插值点的值，单个值或者向量

    def fit_interp(self):

        """
        生成分段线性插值多项式
        """

        t = sp.Symbol('t')  # 定义符号变量
        self.linear_poly = dict()  # 线性函数
        self.linear_coefficient = np.zeros((self.n - 1, 2))
        for i in range(self.n - 1):
            hi = self.x[i + 1] - self.x[i]  # 每个小区间的长度
            # 分段线性插值的函数公式
            linear_i = (self.y[i + 1] * (t - self.x[i]) - self.y[i] * (t - self.x[i + 1])) / hi
            self.linear_poly[i] = sp.simplify(linear_i)
            linear_obj = sp.Poly(linear_i, t)  # 构造多项式对象
            # 某项系数可能为0，为防止存储错误，分别对应各阶次存储
            mons = linear_obj.monoms()  # 多项式系数对应的阶次
            for j in range(len(mons)):
                self.linear_coefficient[i, mons[j][0]] = linear_obj.coeffs()[j]  # 获取多项式的系数

    def cal_interp(self, x0):

        """
        计算给定插值点的数值
        x0:所求插值的x坐标值
        """

        self.y0 = cal_interp(self.linear_poly, self.x, x0)
        return self.y0

    def plt_interp(self, x0=None, y0=None):

        """
        可视化插值图像和所求的插值点
        """

        params = (self.linear_poly, self.x, self.y, "Piecewise Linear", x0, y0)
        plt_interp(params)
