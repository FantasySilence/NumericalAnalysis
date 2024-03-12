import os
import sys
import sympy as sp

# 设置全局路径，以便引入模块
test_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root_path = os.path.dirname(test_path)
sys.path.append(root_path)

# 导入对应模块
from src.modules.approximation.chebyshevseries import ChebyshevSeriesApproximation

# 测试用例
t = sp.symbols('t')
fun = sp.exp(t)

csa = ChebyshevSeriesApproximation(fun, x_span=[-1, 1], k=3)
csa.fit_approximation()
print('切比雪夫级数逼近系数及对应阶次：')
print(csa.poly_coefficient)
print(csa.polynomial_orders)
print('切比雪夫级数逼近多项式：')
print(csa.approximation_poly)
print('切比雪夫级数逼近的最大绝对误差是：')
print(csa.max_abs_error)
csa.plt_approximation()
