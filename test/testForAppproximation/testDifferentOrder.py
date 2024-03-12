import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# 设置全局路径，以便引入模块
test_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root_path = os.path.dirname(test_path)
sys.path.append(root_path)

# 导入对应模块
from src.modules.approximation.leastsquarefitting import LeastSquarePolynomialFitting

# 最小二乘曲线拟合不同阶次对比
x = np.linspace(0, 5, 100)
np.random.seed(0)
y = np.sin(x) * np.exp(-x) + np.random.randn(100) / 100
xi = np.linspace(0, 5, 100)
plt.figure(figsize=(8, 6))
orders = [12, 15, 17, 20, 25]
line_style = ["--", ":", "-", "-.", "-*"]
for k, line in zip(orders, line_style):
    ls = LeastSquarePolynomialFitting(x, y, k=k)
    ls.fit_curve()
    yi = ls.cal_x0(xi)
    plt.plot(xi, yi, line, lw=1.5, label="Order=%d, MSE=%.2e" % (k, ls.mse))
plt.plot(x, y, 'ko', label="离散数据点")
plt.xlabel('X', fontdict={"fontsize": 12})
plt.ylabel("Y", fontdict={"fontsize": 12})
plt.title("Least Square Fitting with Different Orders", fontdict={"fontsize": 14})
plt.legend(loc='best')
plt.show()
