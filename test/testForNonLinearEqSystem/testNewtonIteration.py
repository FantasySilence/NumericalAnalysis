import os
import sys
import sympy as sp

# 设置全局路径，以便引入模块
test_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root_path = os.path.dirname(test_path)
sys.path.append(root_path)

# 导入对应模块
from src.modules.nonLinearEqSystem.newtoniteration import NewtonIterationMethod

# 测试用例
x = sp.Symbol('x')
fun = x ** 3 + 10 * x - 20
fun1 = x ** 3 / 3 - x
newton1 = NewtonIterationMethod(fun1, x0=1.5, iter_type="offhill")
newton2 = NewtonIterationMethod(fun1, x0=1.5, iter_type="simnewton")
newton3 = NewtonIterationMethod(fun1, x0=1.5, iter_type="default")
