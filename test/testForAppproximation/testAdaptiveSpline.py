import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# 设置全局路径，以便引入模块
test_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root_path = os.path.dirname(test_path)
sys.path.append(root_path)

# 导入对应模块
from src.modules.approximation.adaptivespline import AdaptiveSplineApproximation


# 测试用例
def fun(x):
    return 1 / (1 + x ** 2)


ase = AdaptiveSplineApproximation(fun, x_span=[-5, 5], eps=1e-8)
ase.fit_approximation()
print(ase.node)
print(ase.spline_obj.m)
ase.plt_approximation(is_show=False)

plt.figure(figsize=(13, 5))
plt.subplot(121)
bins = np.linspace(-5, 5, 11)
n = plt.hist(ase.node, bins=bins, rwidth=0.8, color="r", alpha=.5)
plt.plot((n[1][:-1] + n[1][1:]) / 2, n[0], "ko-")
plt.subplot(122)
y_val = fun(ase.node)
plt.plot(ase.node, y_val, "ko-")
plt.show()
