import numpy as np
import sympy as sp
import matplotlib
from scipy import integrate

matplotlib.rcParams['font.sans-serif'] = ['STSong']
matplotlib.rcParams['axes.unicode_minus'] = False

from src.common.utilsapproximation.best_approximation_utils import cal_x0
from src.common.utilsapproximation.best_approximation_utils import error_analysis
from src.common.utilsapproximation.best_approximation_utils import plt_approximation



class BestSquareApproximation:

    """
    最佳平方逼近
    """

    def __init__(self, fun, x_span, k):

        """
        必要的参数的初始化
        """

        self.fun = fun      # 符号函数定义
        self.k = k  # 逼近多项式的阶次
        if len(x_span) == 2:
            self.a, self.b = x_span[0], x_span[1]
        else:
            raise ValueError("区间的定义有误！")
        self.approximation_poly = None      # 逼近的多项式
        self.poly_coefficient = None    # 逼近多项式的系数
        self.polynomial_orders = None    # 逼近多项式各项阶次
        self.max_abs_error = np.infty    # 10次模拟的逼近多项式的最大绝对误差
        self.mae = np.infty     # 10次模拟的平均绝对误差


    def fit_approximation(self):

        """
        最佳平方逼近：构造希尔伯特矩阵和右端向量，求解方程组获得最佳平方逼近多项式的系数
        """

        t = self.fun.free_symbols.pop()     # 获取函数的自由变量
        H = np.zeros((self.k+1,self.k+1))     # 初始化
        d = np.zeros((self.k+1,1))
        func = self.fun/t
        for i in range(self.k+1):
            # 此循环值计算希尔伯特矩阵的第一行
            H[0,i] = (self.b**(i+1)-self.a**(i+1))/(i+1)
            func = func*t    # f(t),t*f(t),t**2*f(t)...
            func_expr = sp.lambdify(t,func)
            d[i] = integrate.quad(func_expr, self.a, self.b, full_output=1)[0]      # 右端向量内的积分
        for i in range(1,self.k+1):
            # 每次把H矩阵的上一行从第二个元素到最后一个元素的值，赋值给下一行从第一个元素到导数第二个元素
            H[i,:-1] = H[i-1,1:]    
            f1,f2 = self.b**(self.k+i+1), self.a**(self.k+i+1)
            H[i,-1] = (f1-f2)/(self.k+i+1)
        
        print('系数矩阵的条件数:',np.linalg.cond(H))    # 病态矩阵
        self.poly_coefficient = np.linalg.solve(H,d)    # 求解方程组
        px = sp.Matrix.zeros(self.k+1, 1)
        for i in range(self.k+1):
            px[i] = np.power(t, i)
        self.approximation_poly = self.poly_coefficient[0]*px[0]
        for i in range(1,self.k+1):
            self.approximation_poly += self.poly_coefficient[i]*px[i]
        self.approximation_poly = sp.expand(*self.approximation_poly)
        polynomial = sp.Poly(self.approximation_poly, t)
        self.polynomial_orders = polynomial.monoms()[::-1]  # 阶次，从低到高
        self.error_analysis()


    def cal_x0(self, x0):

        """
        求解给定点的逼近值
        x0：所求逼近点的x坐标
        """
        
        return cal_x0(self.approximation_poly, x0)


    def error_analysis(self):

        """
        最佳平方逼近度量
        进行10次模拟，每次模拟指定区间随机生成100个数据点，然后根据度量方法分析
        """
        
        t = self.fun.free_symbols.pop()     # 获取函数的自由变量
        lambda_fun = sp.lambdify(t, self.fun)
        params = self.approximation_poly, lambda_fun, self.a, self.b
        self.max_abs_error, self.mae = error_analysis(params)
    

    def plt_approximation(self, is_show=True):

        """
        绘制逼近多项式的图像
        """
        
        t = self.fun.free_symbols.pop()     # 获取函数的自由变量
        lambda_fun = sp.lambdify(t, self.fun)
        params = self.approximation_poly, lambda_fun, self.a, self.b, self.k, self.mae, 'Best Square', is_show
        plt_approximation(params)