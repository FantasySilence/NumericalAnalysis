import os
import sys
import sympy as sp

# 设置全局路径，以便引入模块
test_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root_path = os.path.dirname(test_path)
sys.path.append(root_path)

# 导入对应模块
from src.modules.nonLinearEqSystem.secant import SecantMethod

# 测试用例
x = sp.Symbol('x')
fun = x**3-3*x+1
secant = SecantMethod(fun, x0=0.5, x1=0.4, eps=1e-8, is_print=True)