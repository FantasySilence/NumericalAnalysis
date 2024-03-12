import os
import sys
import sympy as sp

# 设置全局路径，以便引入模块
test_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root_path = os.path.dirname(test_path)
sys.path.append(root_path)

# 导入对应模块
from src.modules.approximation.bestsquare import BestSquareApproximation

# 测试用例
t = sp.symbols('t')
fun = 1 / (1 + t ** 2)

bsa = BestSquareApproximation(fun, x_span=[-5, 5], k=30)
bsa.fit_approximation()
print('最佳平方逼近系数及对应阶次：')
print(bsa.poly_coefficient)
print(bsa.polynomial_orders)
print('最佳平方逼近多项式：')
print(bsa.approximation_poly)
print('最佳平方逼近的最大绝对误差是：')
print(bsa.max_abs_error)
bsa.plt_approximation()
