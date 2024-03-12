import os
import sys
import sympy as sp

# 设置全局路径，以便引入模块
test_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root_path = os.path.dirname(test_path)
sys.path.append(root_path)

# 导入对应模块
from src.modules.nonLinearEqSystem.fixedpoint import FixedPointsIteration

# 测试用例
x = sp.Symbol('x')
fun = sp.sqrt(10 - x ** 3) / 2
fpi = FixedPointsIteration(iter_fun=fun, x0=1.5, interval=[1, 2],
                           is_print=True, is_plot=True)
