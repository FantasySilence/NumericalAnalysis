import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['STSong']
matplotlib.rcParams['axes.unicode_minus'] = False


class LeastSquarePolynomialFitting:

    """
    powered by:@御河DE天街\n
    最小二乘拟合，可以选用不同的基函数\n
    default:以1,x,x**2...等幂函数为基函数\n
    ort:使用正交多项式作为基函数\n
    other:使用自定义的一系列函数作为基函数
    """

    def __init__(self, x, y, k=None, w=None, base_fun='default', fun_list=None):

        """
        一些必要的参数初始化\n
        x, y: 离散数据点，离散数据点的长度需一致\n
        k: 进行多项式拟合时，必须指定的多项式的最高阶次\n
        w: 权系数，长度需与离散数据点的长度一致，默认情况下为1\n
        base_fun: 所用拟合基函数的类型，默认为"default"\n
                       "default":以1,x,x**2...等幂函数为基函数\n
                       "ort":使用正交多项式作为基函数\n
                       "other":使用自定义的一系列函数作为基函数\n
        fun_list: 自定义的基函数，当base_fun = other时，必须指定自定义基函数列表fun_list
        """

        self.x, self.y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        # 如果进行多项式拟合，此变量为多项式最高阶次，需要指定
        self.k = k
        if len(self.x) != len(self.y):
            raise ValueError("离散点数据长度不一致！")
        else:
            self.n = len(self.x)  # 离散点的个数
        if w is None:
            self.w = np.ones(self.n)  # 默认情况下，所有数据权重一致为1
        else:
            if len(w) != self.n:
                raise ValueError("权重长度与离散数据点不一致！")
            else:
                self.w = np.asarray(w, dtype=np.float64)
        if fun_list is not None:
            self.fun_list = list(fun_list)  # 自定义的基函数，不能带值计算
            self.m = len(fun_list)  # 自定义基函数的个数
            self.fun_list1 = None  # 转化后的自定义基函数，可以带值计算
        else:
            self.fun_list = None
            self.m = None
        self.ort_poly = None  # 正交多项式
        self.base_fun = base_fun  # base_fun = default(x的幂函数) or ort(正交多项式) or other(自定义基函数)
        self.fit_poly = None  # 曲线拟合的多项式
        self.poly_coefficient = None  # 多项式的系数向量
        self.polynomial_orders = None  # 系数的阶次
        self.fit_error = None  # 拟合的误差向量
        self.mse = np.infty  # 拟合的均方根误差

    def fit_curve(self):

        """
        最小二乘法拟合
        """

        if self.base_fun == 'default':
            if self.k is None:
                raise ValueError("多项式最高阶次未指定！")
            else:
                self.__poly_fit__()
        elif self.base_fun == 'ort':
            if self.k is None:
                raise ValueError("多项式最高阶次未指定！")
            else:
                self.__ortpoly_fit__()
        elif self.base_fun == 'other':
            if self.fun_list is None:
                raise ValueError("自定义基函数未指定！")
            else:
                self.__other_fit__()
        else:
            raise ValueError("拟合方法有且只有default(x的幂函数) or ort(正交多项式) or other(自定义基函数)")

    def __poly_fit__(self):

        """
        使用x的幂函数作为基函数
        """

        c = np.zeros(2 * self.k + 1)
        b = np.zeros(self.k + 1)  # 右端向量
        for k in range(2 * self.k + 1):
            c[k] = np.dot(self.w, np.power(self.x, k))
        for k in range(self.k + 1):
            b[k] = np.dot(self.w, np.power(self.x, k) * self.y)
        C = np.zeros((self.k + 1, self.k + 1))  # 系数矩阵
        for k in range(self.k + 1):
            C[k, :] = c[k:k + self.k + 1]
        self.poly_coefficient = np.linalg.solve(C, b)

        t = sp.symbols('t')
        self.fit_poly = self.poly_coefficient[0] * 1
        for p in range(1, self.k + 1):
            px = np.power(t, p)  # 幂次
            self.fit_poly += self.poly_coefficient[p] * px
        poly = sp.Poly(self.fit_poly, t)
        self.polynomial_orders = poly.monoms()[::-1]  # 阶次
        self.__cal_fit_error__()  # 误差分析

    def __lambdified__(self, fun_list):

        """
        将传入的自定义基函数转化为可以进行计算的函数
        """

        fun_list1 = []
        for fun in fun_list:
            if isinstance(fun, float):
                raise ValueError("不是，哥们儿，谁教你拿小数做基函数啊(恼)")
            if isinstance(fun, int):
                fun_list1.append(int(fun))
            else:
                t = fun.free_symbols.pop()
                fun = sp.lambdify(t, fun)
                fun_list1.append(fun)
        return fun_list1

    def __ortpoly_fit__(self):

        """
        使用正交多项式作为基函数
        """

        def prod(poly_1, poly_2):
            """
            定义内积运算
            """
            poly = poly_1 * poly_2
            t = poly.free_symbols.pop()
            poly = sp.lambdify(t, poly)
            return np.dot(self.w, poly(self.x))

        t = sp.Symbol('t')
        coefficient = np.zeros(self.k + 1)  # 初始化，用于存储拟合多项式系数
        ort = np.zeros(self.k + 1, dtype=object)  # 初始化，用于存储带权正交多项式
        ort[0], ort[1] = 1, t - np.dot(self.w, self.x) / np.sum(self.w)
        ort[2] = sp.expand((t - (prod(t * ort[1], ort[1]) / prod(ort[1], ort[1]))) * ort[1] - \
                           (prod(ort[1], ort[1]) / np.sum(self.w)) * ort[0])  # 初始化正交多项式递推的前三项
        # 从第四项开始递推求解正交多项式
        for i in range(3, self.k + 1):
            ort[i] = (t - prod(t * ort[i - 1], ort[i - 1]) / prod(ort[i - 1], ort[i - 1])) * ort[i - 1] - \
                     (prod(ort[i - 1], ort[i - 1]) / prod(ort[i - 2], ort[i - 2])) * ort[i - 2]
            ort[i] = sp.expand(ort[i])
        self.ort_poly = ort  # 正交多项式：最终的结果
        coefficient[0] = np.dot(self.y, self.w) / np.sum(self.w)  # 初始化拟合多项式系数的第一项
        lambda_ort = ort.copy()
        self.fit_poly = coefficient[0] * ort[0]  # 初始化拟合多项式的第一项
        # 从第二项开始循环求解拟合多项式系数并输出最终的拟合多项式
        for i in range(1, self.k + 1):
            lambda_ort[i] = sp.lambdify(t, lambda_ort[i])
            coefficient[i] = np.dot(self.w, self.y * lambda_ort[i](self.x)) / \
                             np.dot(lambda_ort[i](self.x), lambda_ort[i](self.x))
            self.fit_poly += coefficient[i] * ort[i]
        self.fit_poly = sp.expand(self.fit_poly)

        # 输出一些拟合多项式的特征
        polynomial = sp.Poly(self.fit_poly, t)
        self.poly_coefficient = polynomial.coeffs()  # 拟合多项式的系数
        self.polynomial_orders = polynomial.monoms()  # 拟合多项式系数的阶次
        self.__cal_fit_error__()  # 误差分析

    def __other_fit__(self):

        """
        使用自定义的基函数
        """

        C = np.zeros((self.m, self.m))  # 初始化法方程系数矩阵
        d = np.zeros(self.m)  # 初始化法方程右端向量
        self.fun_list1 = self.__lambdified__(self.fun_list)  # 转化自定义的基函数
        fun_type = list(map(type, self.fun_list1))  # 提取基函数列表中元素的类型
        function = None  # 元素为函数表达式时元素的类型
        for t in fun_type:
            if t == int:
                continue
            if t != int:
                function = t
                break
        # 构造法方程的系数矩阵
        for i in range(self.m):
            for j in range(self.m):
                if isinstance(self.fun_list1[i], function) and isinstance(self.fun_list1[j], function):
                    C[i, j] = np.dot(self.w, self.fun_list1[i](self.x) * self.fun_list1[j](self.x))
                if isinstance(self.fun_list1[i], int) and isinstance(self.fun_list1[j], function):
                    C[i, j] = np.dot(self.w, self.fun_list1[i] * self.x * self.fun_list1[j](self.x))
                if isinstance(self.fun_list1[i], function) and isinstance(self.fun_list1[j], int):
                    C[i, j] = np.dot(self.w, self.fun_list1[i](self.x) * self.fun_list1[j] * self.x)
                if isinstance(self.fun_list1[i], int) and isinstance(self.fun_list1[j], int):
                    C[i, j] = np.dot(self.w, self.fun_list1[i] * self.fun_list1[j] * self.x)
        # 构造法方程的右端向量
        for i in range(self.m):
            if isinstance(self.fun_list1[i], function):
                d[i] = np.dot(self.w, self.fun_list1[i](self.x) * self.y)
            if isinstance(self.fun_list1[i], int):
                d[i] = np.dot(self.w, self.fun_list1[i] * self.x * self.y)
        # 求解法方程
        self.poly_coefficient = np.linalg.solve(C, d)

        t = sp.symbols('t')
        self.fit_poly = self.poly_coefficient[0] * self.fun_list[0]
        for p in range(1, self.m):
            self.fit_poly += self.poly_coefficient[p] * self.fun_list[p]

        self.__cal_fit_error__()  # 误差分析

    def __cal_fit_error__(self):

        """
        计算拟合的误差和均方根误差
        """

        y_fit = self.cal_x0(self.x)
        self.fit_error = self.y - y_fit  # 误差向量
        self.mse = np.sqrt(np.mean(self.fit_error ** 2))  # 均方根误差

    def cal_x0(self, x0):

        """
        求解给定数值x0的拟合值
        """

        t = self.fit_poly.free_symbols.pop()
        fit_poly = sp.lambdify(t, self.fit_poly)
        return fit_poly(x0)

    def plt_curve_fit(self, is_show=True):

        """
        拟合曲线以及离散数据点的可视化
        """

        xi = np.linspace(self.x.min(), self.x.max(), 100)
        yi = self.cal_x0(xi)  # 拟合值
        if is_show:
            plt.figure(figsize=(8, 6))
        plt.plot(xi, yi, 'k-', lw=1.5, label="拟合曲线")
        plt.plot(self.x, self.y, 'ro', lw=1.5, label="离散数据点")
        if self.k:
            plt.title("最小二乘拟合(MSE=%.2e,Order=%d)" % (self.mse, self.k), fontdict={"fontsize": 14})
        else:
            plt.title("最小二乘拟合(MSE=%.2e)" % self.mse, fontdict={"fontsize": 14})
        plt.xlabel('X', fontdict={"fontsize": 12})
        plt.ylabel("Y", fontdict={"fontsize": 12})
        plt.legend(loc='best')
        if is_show:
            plt.show()
