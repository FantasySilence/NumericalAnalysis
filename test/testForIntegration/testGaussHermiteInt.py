import os
import sys
import numpy as np

# 设置全局路径，以便引入模块
test_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root_path = os.path.dirname(test_path)
sys.path.append(root_path)

# 导入对应模块
from src.modules.intergration.gausshermiteint import GuassHermiteIntergration

# 测试用例
def fun(x):
    return x**2*np.exp(-x**2)

herm = GuassHermiteIntergration(fun, zeros_num=15)
herm.cal_int()
print(herm.zero_points)
print(herm.A_k)
print(herm.int_value, np.sqrt(np.pi)/2 - herm.int_value)