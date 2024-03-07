import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# 设置全局路径，以便引入模块
test_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root_path = os.path.dirname(test_path)
sys.path.append(root_path)

# 导入对应模块
from src.modules.intergration.cubicbsplineinterpint\
import CubicBsplineInterpolationIntergration

# 测试用例
x = np.linspace(0, 24, 25)
y = np.array([0, 0.45, 1.79, 4.02, 7.15, 11.18, 16.09, 21.90, 29.05,
                29.05, 29.05, 29.05, 29.05, 22.42, 17.9, 17.9, 17.9,
                17.9, 14.34, 11.01, 8.9, 6.54, 2.03, 0.55, 0])
cbsi1 = CubicBsplineInterpolationIntergration(x, y)
int_value1 = cbsi1.cal_int()
print(int_value1) 
plt.figure(figsize=(8, 6))
plt.plot(x, y, "ko-")
plt.show()

def fun(x):
    return np.sin(x)/x
x = np.linspace(0,1,30)
y = np.zeros(30)
y[1:] = fun(x[1:])
y[0] = 1

cbsi2 = CubicBsplineInterpolationIntergration(x, y)
int_value2 = cbsi2.cal_int()
print(int_value2)