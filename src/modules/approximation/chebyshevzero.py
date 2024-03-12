import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['STSong']
matplotlib.rcParams['axes.unicode_minus'] = False

from src.common.utilsapproximation.Lagrange import LagrangeInterpolation


class ChebyshevZeroPointsInterpolation:

    """
    切比雪夫多项式零点插值
    """

    def __init__(self, fun, x_span=np.array([-1, 1]), order=5):
        """
        必要的参数初始化
        """

        self.fun = fun  # 被逼近函数
        self.order = order  # 插值多项式最高阶数
        self.a, self.b = x_span[0], x_span[1]  # 区间的左右端点
        self.chebyshev_zeros = None  # 切比雪夫多项式的零点
        self.approximation_poly = None  # 最终的逼近多项式
        self.poly_coefficient = None  # 逼近多项式的系数
        self.polynomial_orders = None  # 逼近多项式系数的阶数
        self.max_abs_error = np.infty  # 逼近多项式的最大绝对误差
        self.mae = np.infty  # 随机数模拟的绝对误差均值

    def fit_approximation(self):
        """
        切比雪夫多项式零点插值核心算法
        """

        # 1.求解切比雪夫多项式零点
        self.chebyshev_zeros = np.zeros(self.order + 1)
        for k in range(self.order + 1):
            zero = np.cos((2 * k + 1) / 2 / (self.order + 1) * np.pi)
            # 区间变换
            self.chebyshev_zeros[k] = (self.b - self.a) / 2 * zero + (self.b + self.a) / 2

        # 2.根据零点求解函数值
        fun_values = self.fun(self.chebyshev_zeros)

        # 3.根据零点和函数值构造拉格朗日插值多项式
        lag = LagrangeInterpolation(self.chebyshev_zeros, fun_values)
        lag.fit_interp()
        self.approximation_poly = lag.polynomial
        self.poly_coefficient = lag.poly_coefficient
        self.coefficient_order = lag.coefficient_order

        # 4.误差分析
        self.error_analysis()

    def cal_x0(self, x0):
        """
        求解逼近多项式在给定点x0的逼近值
        x0：所求的逼近点向量
        """

        t = self.approximation_poly.free_symbols.pop()  # 获取多项式的符号变量
        y0 = np.zeros(len(x0))  # 存储逼近点的逼近值
        for i in range(len(x0)):
            y0[i] = self.approximation_poly.evalf(subs={t: x0[i]})
        return y0

    def error_analysis(self):
        """
        切比雪夫多项式零点插值逼近度量
        进行10次模拟，每次模拟指定区间随机生成100个数据点，然后根据度量方法分析
        """

        mae, max_error = np.zeros(10), np.zeros(10)
        for i in range(10):
            xi = self.a + np.random.rand(100) * (self.b - self.a)  # 区间[a, b]内的随机数
            y_ture = self.fun(xi)
            y_appr = self.cal_x0(xi)
            mae[i] = np.mean(np.abs(y_ture - y_appr))  # 每次模拟的100个随机点中绝对误差均值
            max_error[i] = np.max(np.abs(y_ture - y_appr))  # 每次模拟的100个随机点中最大据对误差值
        self.mae = np.mean(mae)  # 10次模拟的均值
        self.max_abs_error = np.max(max_error)  # 10次模拟的绝对误差最大值

    def plt_approximation(self, is_show=True):
        """
        绘制逼近多项式的图像
        """

        if is_show:
            plt.figure(figsize=(8, 6))
        xi = self.a + np.random.rand(100) * (self.b - self.a)  # 区间[a, b]内的随机数
        xi = np.array(sorted(xi))  # 升序排序
        y_ture = self.fun(xi)
        y_appr = self.cal_x0(xi)
        plt.plot(xi, y_ture, 'k+-.', lw=1.5, label='true function')
        plt.plot(xi, y_appr, 'r*--', lw=1.5, label='approximation(k=%d)' % self.order)
        plt.legend(loc='best')
        plt.xlabel('X(Randomly divide 100 points)', fontdict={"fontsize": 12})
        plt.ylabel('Exact vs Appro', fontdict={"fontsize": 12})
        plt.title("Chebyshev Zero Points Interpolation Approximation(mae_10=%.2e)" % self.mae,
                  fontdict={"fontsize": 13})
        plt.grid(ls=":")
        if is_show:
            plt.show()
