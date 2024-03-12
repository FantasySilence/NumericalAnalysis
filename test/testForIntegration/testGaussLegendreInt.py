import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# 设置全局路径，以便引入模块
test_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root_path = os.path.dirname(test_path)
sys.path.append(root_path)

# 导入对应模块
from src.modules.intergration.gausslegendreint \
    import GuassLegendreIntergration


# 测试用例
def fun(x):
    return np.sin(x) * np.exp(-x)


int_res = 0.5 * (1 - np.exp(-8) * (np.sin(8) + np.cos(8)))
precision = []
guass_zeros_num = np.arange(10, 21, 1)
for num in guass_zeros_num:
    leg = GuassLegendreIntergration(fun, [0, 8], zeros_num=num)
    int_value = leg.cal_int()
    precision.append(abs(int_res - int_value))
    print("num:%d, 积分值:%.15f, 误差:%.15e" % (num, int_value, precision[-1]))

plt.figure(figsize=(8, 6))
plt.plot(guass_zeros_num, precision, "ko-")
plt.show()
