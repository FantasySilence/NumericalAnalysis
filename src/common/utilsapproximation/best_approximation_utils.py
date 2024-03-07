import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

"""
powered by:@御河DE天街
最佳平方逼近使用了本工具包
"""

def cal_x0(approximation_poly,x0):
    
    """
    求解逼近多项式在给定点x0的逼近值
    x0：所求的逼近点向量
    """
    
    t = approximation_poly.free_symbols.pop()  # 获取多项式的符号变量
    y0 = np.zeros(len(x0))      # 存储逼近点的逼近值
    for i in range(len(x0)):
        y0[i] = approximation_poly.evalf(subs={t:x0[i]})
    return y0


def error_analysis(params):

    """
    切比雪夫多项式零点插值逼近度量
    进行10次模拟，每次模拟指定区间随机生成100个数据点，然后根据度量方法分析
    """

    approximation_poly, fun_expr, a, b = params
    mae, max_error = np.zeros(10), np.zeros(10)
    for i in range(10):
        xi = a + np.random.rand(100) * (b - a)   # 区间[a, b]内的随机数
        xi = np.array(sorted(xi))   # 升序排序
        y_ture = fun_expr(xi)
        y_appr = cal_x0(approximation_poly,xi)
        mae[i] = np.mean(np.abs(y_ture - y_appr))   # 每次模拟的100个随机点中绝对误差均值
        max_error[i] = np.max(np.abs(y_ture - y_appr))  # 每次模拟的100个随机点中最大据对误差值
    mae_ = np.mean(mae)     # 10次模拟的均值
    max_abs_error = np.max(max_error)      # 10次模拟的绝对误差最大值
    return max_abs_error, mae_


def plt_approximation(params):

    """
    绘制逼近多项式的图像
    """
    
    approximation_poly, fun_expr, a, b, order, mae, title, is_show = params
    if is_show:    
        plt.figure(figsize=(8, 6))
    xi = a + np.random.rand(100) * (b - a)   # 区间[a, b]内的随机数
    xi = np.array(sorted(xi),dtype=np.float64)   # 升序排序
    y_ture = fun_expr(xi)
    y_appr = cal_x0(approximation_poly,xi)
    plt.plot(xi, y_ture, 'k+-.', lw=1.5, label='true function')
    plt.plot(xi, y_appr, 'r*--', lw=1.5, label='approximation(k=%d)' % order)
    plt.legend(loc='best')
    plt.xlabel('X(Randomly divide 100 points)',fontdict={"fontsize":12})
    plt.ylabel('Exact vs Appro',fontdict={"fontsize":12})
    plt.title("%s Approximation(mae_10=%.2e)"% (title, mae),
             fontdict={"fontsize":13})
    plt.grid(ls=":")
    if is_show:
        plt.show()