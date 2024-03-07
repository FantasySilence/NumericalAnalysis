import os
import sys
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['STSong']
matplotlib.rcParams['axes.unicode_minus'] = False

# 设置全局路径，以便引入模块
test_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root_path = os.path.dirname(test_path)
sys.path.append(root_path)

# 导入对应模块
from src.modules.approximation.chebyshevseries import ChebyshevSeriesApproximation
from src.modules.approximation.chebyshevzero import ChebyshevZeroPointsInterpolation
from src.modules.approximation.legendreseries import LegendreSeriesApproximation

# 三种逼近方法的比较
def runge_fun(x):
    return 1/(x**2+1)

t = sp.symbols('t')
fun = 1/(t**2+1)
orders = [10, 25]
plt.figure(figsize=(14,15))
for i, order in enumerate(orders):
    plt.subplot(321 + i)
    cpzi = ChebyshevZeroPointsInterpolation(runge_fun,x_span=[-5,5],order=order)
    cpzi.fit_approximation()
    cpzi.plt_approximation(is_show=False)
    print("切比雪夫零点插值的最大绝对误差：",cpzi.max_abs_error)
    plt.subplot(323 + i)
    csa = ChebyshevSeriesApproximation(fun, x_span=[-5,5], k=order)
    csa.fit_approximation()
    csa.plt_approximation(is_show=False)
    print("切比雪夫级数逼近的最大绝对误差：",csa.max_abs_error)
    plt.subplot(325 + i)
    lsa = LegendreSeriesApproximation(fun, x_span=[-5,5], k=order)
    lsa.fit_approximation()
    lsa.plt_approximation(is_show=False)
    print("勒让德级数逼近的最大绝对误差：",lsa.max_abs_error)
plt.show()