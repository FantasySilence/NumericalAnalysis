import os
import sys
import numpy as np

# 设置全局路径，以便引入模块
test_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root_path = os.path.dirname(test_path)
sys.path.append(root_path)

# 导入对应模块
from src.modules.intergration.rombergaccelerationint\
import RombergAccelerationQuadrature

# 测试用例
def fun(x):
    return x**(3/2)

raq1 = RombergAccelerationQuadrature(fun, [0, 1], acceleration_num=15)
raq1.cal_int()
print(raq1.int_value)
print(raq1.romberg_table)

def fun2(x):
    return np.exp(x**2)*((0<=x)&(x<=2))+80/(4-np.sin(16*np.pi*x))*((2<x)&(x<=4))

acceleration_num = np.arange(10,21,1)
a = 57.764450125048512  # 测试函数的一个高精度积分值
for num in acceleration_num:
    raq2 = RombergAccelerationQuadrature(fun2, [0, 4], acceleration_num=num)
    raq2.cal_int()
    print("外推次数%d,积分值%.15f,误差%.15e"%(num,raq2.int_value,a-raq2.int_value))