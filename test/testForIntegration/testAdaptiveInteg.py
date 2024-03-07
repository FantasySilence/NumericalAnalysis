import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# 设置全局路径，以便引入模块
test_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root_path = os.path.dirname(test_path)
sys.path.append(root_path)

# 导入对应模块
from src.modules.intergration.adaptiveinteg import AdaptiveIntergralAlgorithm

# 测试用例
def fun(x):
    return 1/(np.sin(6*np.pi*x)+np.sqrt(x))

def fun2(x):
    return x**2*np.log(x)

ada = AdaptiveIntergralAlgorithm(fun, [1,2], eps=1e-5)
ada.cal_int()
print("自适应积分值：", ada.int_value)
print("划分区间：", ada.x_node)
print("划分节点数：", len(ada.x_node))

plt.figure(figsize=(8,12))
plt.subplot(211)
xi = np.linspace(1, 2, 100)
yi = fun(xi)
y_val = fun(ada.x_node)
plt.plot(ada.x_node, y_val, "k.")
plt.fill_between(xi, yi, color = "c", alpha=0.4)
plt.subplot(212)
bins = np.linspace(1, 2, 11)
n = plt.hist(ada.x_node, bins=bins, color="r", alpha=0.5)
plt.plot((n[1][:-1]+n[1][1:])/2, n[0], "ko-", lw=2)
plt.show()