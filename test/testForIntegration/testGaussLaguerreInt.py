import os
import sys
import numpy as np

# 设置全局路径，以便引入模块
test_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root_path = os.path.dirname(test_path)
sys.path.append(root_path)

# 导入对应模块
from src.modules.intergration.gausslaguerreint import GuassLaguerreIntergration

# 测试用例
def fun(x):
    return np.sin(x)*np.exp(-x)

lagr = GuassLaguerreIntergration(fun, [-2, np.infty], zeros_num=15)
lagr.cal_int()
print(lagr.zero_points)
print(lagr.A_k)
val = 0.5*np.exp(2)*(np.cos(-2)+np.sin(-2))
print(lagr.int_value, val - lagr.int_value)