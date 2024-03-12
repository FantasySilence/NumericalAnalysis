import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['STSong']
matplotlib.rcParams['axes.unicode_minus'] = False


class DiscreteFourierTransformApproximation:

    """
    离散傅里叶变换逼近
    """

    def __init__(self, y, x_span, fun=None):

        """
        必要的参数初始化
        """

        self.y = y  # 被逼近的离散数据点
        self.a, self.b = x_span[0], x_span[1]
        self.fun = fun  # 被逼近的函数,可以不传入
        self.cos_term = None  # 展开后的余弦项系数
        self.sin_term = None  # 展开后的正弦项系数
        self.approximation_poly = None  # 逼近的最小二乘三角多项式

    def fit_approximation(self):

        """
        离散傅里叶变换逼近求解
        """

        t = sp.symbols('t')
        n = len(self.y)  # 离散数据点的个数
        m = n // 2  # ak的系数个数
        self.cos_term = np.zeros(m + 1)
        self.approximation_poly = 0.0
        idx = np.linspace(0, n, n, endpoint=False, dtype=np.int64)
        if np.mod(n, 2) == 0:  # 偶数个数
            self.sin_term = np.zeros(m - 1)
            for k in range(m + 1):
                self.cos_term[k] = np.dot(self.y, np.cos(np.pi * k * idx / m))
                if k == 0 or k == m:
                    self.cos_term[k] *= 1 / (2 * m) * (-1) ** k
                else:
                    self.cos_term[k] *= 1 / m * (-1) ** k
                self.approximation_poly += self.cos_term[k] * sp.cos(k * t)
            for k in range(1, m):
                self.sin_term[k - 1] = np.dot(self.y, np.sin(np.pi * k * idx / m))
                self.sin_term[k - 1] *= 1 / m * (-1) ** k
                self.approximation_poly += self.sin_term[k - 1] * sp.sin(k * t)
        else:  # 奇数个数
            self.sin_term = np.zeros(m)
            for k in range(m + 1):
                self.cos_term[k] = np.dot(self.y, np.cos(np.pi * k * idx * 2 / (2 * m + 1)))
                if k == 0:
                    self.cos_term[k] *= 1 / (2 * m + 1) * (-1) ** k
                else:
                    self.cos_term[k] *= 2 / (2 * m + 1) * (-1) ** k
                self.approximation_poly += self.cos_term[k] * sp.cos(k * t)
            for k in range(1, m + 1):
                self.sin_term[k - 1] = np.dot(self.y, np.sin(np.pi * k * idx * 2 / (2 * m + 1)))
                self.sin_term[k - 1] *= 2 / (2 * m + 1) * (-1) ** k
                self.approximation_poly += self.sin_term[k - 1] * sp.sin(k * t)

    def cal_x0(self, x0):

        """
        求解在给定点x0的离散傅里叶逼近值
        """

        t = self.approximation_poly.free_symbols.pop()
        approximation_poly = sp.lambdify(t, self.approximation_poly)
        # 区间转换
        x0 = np.asarray(x0, dtype=np.float64)
        xi = (x0 - (self.a + self.b) / 2) * 2 / (self.b - self.a) * np.pi
        y0 = approximation_poly(xi)
        return y0

    def plt_approximation(self, is_show=True):

        """
        绘制逼近多项式的图像
        """

        if is_show:
            plt.figure(figsize=(8, 6))
        xi = self.a + np.random.rand(100) * (self.b - self.a)  # 区间[a, b]内的随机数
        xi = np.array(sorted(xi), dtype=np.float64)  # 升序排序
        yi = self.cal_x0(xi)
        if self.fun is not None:
            y_true = self.fun(xi)
            plt.plot(xi, y_true, 'k*-', lw=1.5, label="true")
            mse = np.sqrt(np.mean((yi - y_true) ** 2))
            plt.title("DFT Approximation Curve(MSE=%.2e)" % mse, fontdict={"fontsize": 14})
            plt.ylabel("Exact VS Approximation", fontdict={"fontsize": 12})
        else:
            plt.ylabel('Y(Approximation)', fontdict={"fontsize": 12})
            plt.title("DFT Approximation Curve", fontdict={"fontsize": 14})
        plt.plot(xi, yi, 'r*--', lw=1.5, label="appproximation")
        plt.xlabel('X(Ramdomly Divide 100 Points)', fontdict={"fontsize": 12})
        plt.legend(loc='best')
        if is_show:
            plt.show()
