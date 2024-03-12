import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# 设置全局路径，以便引入模块
test_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root_path = os.path.dirname(test_path)
sys.path.append(root_path)

# 导入对应模块
from src.modules.approximation.chebyshevzero import ChebyshevZeroPointsInterpolation


# 测试用例
def fun(x):
    return np.exp(x)


czpi = ChebyshevZeroPointsInterpolation(fun=fun, order=2, x_span=[-1, 1])
czpi.fit_approximation()
print('切比雪夫多项式零点：')
print(czpi.chebyshev_zeros)
print('切比雪夫多项式插值系数与阶次：')
print(czpi.poly_coefficient)
print(czpi.coefficient_order)
print('切比雪夫多项式零点插值逼近多项式：')
print(czpi.approximation_poly)
print('切比雪夫多项式零点插值逼近的最大绝对误差是：')
print(czpi.max_abs_error)
czpi.plt_approximation()
