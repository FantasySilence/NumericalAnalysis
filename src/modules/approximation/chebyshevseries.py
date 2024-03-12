import numpy as np
import sympy as sp
import matplotlib
from scipy import integrate

matplotlib.rcParams['font.sans-serif'] = ['STSong']
matplotlib.rcParams['axes.unicode_minus'] = False

from src.common.utilsapproximation.orthogonal_polynomial_utils import cal_x0
from src.common.utilsapproximation.orthogonal_polynomial_utils import error_analysis
from src.common.utilsapproximation.orthogonal_polynomial_utils import plt_approximation


class ChebyshevSeriesApproximation:

    """
    切比雪夫级数逼近函数:符号运算和数值运算
    """

    def __init__(self, fun, x_span=np.array([-1, 1]), k=6):
        """
        必要的参数初始化
        """

        self.a, self.b = x_span[0], x_span[1]  # 区间左右端点
        self.fun_transform, self.lambda_fun = self.internal_transform(fun)  # 区间转化函数
        self.k = k  # 逼近已知函数所需项数
        self.T_coefficient = None  # 切比雪夫各项和对应系数
        self.approximation_poly = None  # 逼近多项式
        self.poly_coefficient = None  # 逼近多项式系数
        self.polynomial_orders = None  # 逼近多项式各项阶数
        self.max_abs_error = np.infty  # 逼近多项式的最大绝对误差
        self.mae = np.infty  # 随机数模拟的绝对误差均值

    def internal_transform(self, fun):
        """
        区间转化函数
        """

        t = fun.free_symbols.pop()  # 获取函数的符号变量
        fun_transform = fun.subs(t, (self.b - self.a) / 2 * t + (self.b + self.a) / 2)  # 区间变换
        lambda_fun = sp.lambdify(t, fun)  # 构成lambda函数
        return fun_transform, lambda_fun

    def fit_approximation(self):
        """
        切比雪夫级数逼近核心算法:递推Tn(x),求解系数fn，构成逼近多项式
        """

        t = self.fun_transform.free_symbols.pop()  # 获取函数的符号变量
        term = sp.Matrix.zeros(self.k + 1, 1)
        term[0], term[1] = 1, t  # 初始化第一项和第二项
        coefficient = np.zeros(self.k + 1)  # 切比雪夫级数多项式系数
        expr = sp.lambdify(t, self.fun_transform / sp.sqrt(1 - t ** 2))
        coefficient[0] = integrate.quad(expr, -1, 1)[0] * (2 / np.pi)  # f0系数
        expr = sp.lambdify(t, term[1] * self.fun_transform / sp.sqrt(1 - t ** 2))
        coefficient[1] = integrate.quad(expr, -1, 1)[0] * (2 / np.pi)  # f1系数
        self.approximation_poly = coefficient[0] / 2 + coefficient[1] * term[1]
        # 从第三项开始循环求解
        for i in range(2, self.k + 1):
            term[i] = sp.expand(2 * t * term[i - 1] - term[i - 2])  # 递推项Tn(x)
            expr = sp.lambdify(t, term[i] * self.fun_transform / sp.sqrt(1 - t ** 2))
            coefficient[i] = integrate.quad(expr, -1, 1, full_output=1, points=[-1, 1])[0] * (2 / np.pi)  # fn系数
            self.approximation_poly += coefficient[i] * term[i]

        self.T_coefficient = [term, coefficient]  # 存储切比雪夫级数递推项和对应的系数
        self.approximation_poly = sp.expand(self.approximation_poly)
        polynomial = sp.Poly(self.approximation_poly, t)
        self.poly_coefficient = polynomial.coeffs()
        self.polynomial_orders = polynomial.monoms()
        self.error_analysis()  # 误差分析

    def cal_x0(self, x0):
        """
        求解给定点的逼近值
        x0：所求逼近点的x坐标
        """

        return cal_x0(self.approximation_poly, x0, self.a, self.b)

    def error_analysis(self):
        """
        切比雪夫级数零点插值逼近度量
        进行10次模拟，每次模拟指定区间随机生成100个数据点，然后根据度量方法分析
        """

        params = self.approximation_poly, self.lambda_fun, self.a, self.b
        self.max_abs_error, self.mae = error_analysis(params)

    def plt_approximation(self, is_show=True):
        """
        绘制逼近多项式的图像
        """

        params = self.approximation_poly, self.lambda_fun, self.a, self.b, self.k, \
            self.mae, 'Chebyshev Series', is_show
        plt_approximation(params)
