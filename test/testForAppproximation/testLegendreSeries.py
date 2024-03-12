import os
import sys
import sympy as sp

# 设置全局路径，以便引入模块
test_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root_path = os.path.dirname(test_path)
sys.path.append(root_path)

# 导入对应模块
from src.modules.approximation.legendreseries import LegendreSeriesApproximation

# 测试用例
t = sp.symbols('t')
fun = sp.exp(t)

lsa = LegendreSeriesApproximation(fun, x_span=[-1, 1], k=3)
lsa.fit_approximation()
print('勒让德级数逼近系数及对应阶次：')
print(lsa.poly_coefficient)
print(lsa.polynomial_orders)
print('勒让德级数逼近多项式：')
print(lsa.approximation_poly)
print('勒让德级数逼近的最大绝对误差是：')
print(lsa.max_abs_error)
print(lsa.T_coefficient)
lsa.plt_approximation()
