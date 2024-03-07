import os
import sys
import sympy as sp

# 设置全局路径，以便引入模块
test_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root_path = os.path.dirname(test_path)
sys.path.append(root_path)

# 导入对应模块
from src.modules.nonLinearEqSystem.eqsystem import NonLinearSystem

# 测试用例
x1 = sp.Symbol('x1')
x2 = sp.Symbol('x2')

f11 = (x1**2+x2**2+8)/10
f12 = (x1*x2**2+x1+8)/10
F1 = sp.Matrix([f11,f12])

f21 = x1**2-10*x1+x2**2+8
f22 = x1*x2**2+x1-10*x2+8
F2 = sp.Matrix([f21,f22])

Eq1 = NonLinearSystem(F1, [0, 0], is_print=True, solve_type="broyden")
Eq2 = NonLinearSystem(F2, [0,0], is_print=True, solve_type="newton")