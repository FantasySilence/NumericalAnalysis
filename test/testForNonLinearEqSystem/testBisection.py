import os
import sys
import sympy as sp

# 设置全局路径，以便引入模块
test_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root_path = os.path.dirname(test_path)
sys.path.append(root_path)

# 导入对应模块
from src.modules.nonLinearEqSystem.bisection import BisectionMethod

# 测试用例
x = sp.Symbol('x')
fun = sp.exp(-x) - sp.sin(sp.pi / 2 * x)
b = BisectionMethod(fun, [0, 1])
