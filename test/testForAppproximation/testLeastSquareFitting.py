import os
import sys
import numpy as np
import sympy as sp

# 设置全局路径，以便引入模块
test_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root_path = os.path.dirname(test_path)
sys.path.append(root_path)

# 导入对应模块
from src.modules.approximation.leastsquarefitting import LeastSquarePolynomialFitting

# ------ 测试用例 ------ #
# 使用幂函数作为拟合基函数
x = np.linspace(0, 5, 15)
np.random.seed(0)
y = 2 * np.sin(x) * np.exp(-x) + np.random.randn(15) / 100
ls = LeastSquarePolynomialFitting(x, y, k=5)
ls.fit_curve()
print('拟合多项式为：{}'.format(ls.fit_poly))
print('拟合多项式系数：{}'.format(ls.poly_coefficient))
print('拟合多项式系数的阶次：{}'.format(ls.polynomial_orders))
ls.plt_curve_fit()

# 使用自定义的基函数作为拟合基函数
t = sp.Symbol('t')
fun_list = [1, sp.log(t), sp.cos(t), sp.exp(t)]
x = np.array([0.24, 0.65, 0.95, 1.24, 1.73, 2.01, 2.23, 2.52, 2.77, 2.99])
y = np.array([0.23, -0.26, -1.1, -0.45, 0.27, 0.1, -0.29, 0.24, 0.56, 1])
ls1 = LeastSquarePolynomialFitting(x, y, base_fun='other', fun_list=fun_list)
ls1.fit_curve()
print('拟合多项式为：{}'.format(ls1.fit_poly))
print('拟合多项式系数：{}'.format(ls1.poly_coefficient))
ls1.plt_curve_fit()

# 使用正交多项式作为拟合基函数
x = np.array([0, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
y = np.array([1, 1.75, 1.96, 2.19, 2.44, 2.71, 3])
ls2 = LeastSquarePolynomialFitting(x, y, k=2, base_fun='ort')
ls2.fit_curve()
print('拟合多项式：{}'.format(ls2.fit_poly))
print('拟合多项式系数：{}'.format(ls2.poly_coefficient))
print('拟合多项式系数的阶次：{}'.format(ls2.polynomial_orders))
ls2.plt_curve_fit()
