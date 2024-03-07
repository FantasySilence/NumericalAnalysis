import os
import sys
import numpy as np

# 设置全局路径，以便引入模块
test_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root_path = os.path.dirname(test_path)
sys.path.append(root_path)

# 导入对应模块
from src.modules.intergration.gausschebyshevint import GuassChebyshevIntergration

# 测试用例
def fun1(x):
    # np.exp(x)/np.sqrt(1-x**2)
    return np.exp(x)/np.sqrt(1-x**2)
a1 = 3.97746326050642

def fun2(x):
    # x**2*np.sqrt(1-x**2)
    return x**2*np.sqrt(1-x**2)
a2 = np.pi/8

cheb1 = GuassChebyshevIntergration(fun1, zeros_num=10,cheb_type=1)
cheb1.cal_int()
print(cheb1.zero_points)
print(cheb1.A_k)
print(cheb1.int_value, a1 - cheb1.int_value)
print("-"*60)
cheb2 = GuassChebyshevIntergration(fun2, zeros_num=10,cheb_type=2)
cheb2.cal_int()
print(cheb2.zero_points)
print(cheb2.A_k)
print(cheb2.int_value, a2 - cheb2.int_value)