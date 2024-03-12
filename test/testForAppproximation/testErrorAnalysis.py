import os
import sys
import matplotlib
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

matplotlib.rcParams['font.sans-serif'] = ['STSong']
matplotlib.rcParams['axes.unicode_minus'] = False

# 设置全局路径，以便引入模块
test_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root_path = os.path.dirname(test_path)
sys.path.append(root_path)

# 导入对应模块
from src.modules.approximation.chebyshevseries import ChebyshevSeriesApproximation

# 误差分析，以切比雪夫级数逼近为例
x = sp.symbols('x')
fun = sp.exp(x)

orders = np.arange(4, 21, 1)
mae_ = np.zeros(len(orders))
for i, order in enumerate(orders):
    print(order)
    czpi = ChebyshevSeriesApproximation(fun, x_span=[0, 1], k=order)
    czpi.fit_approximation()
    mae_[i] = czpi.mae

plt.figure(figsize=(8, 6))
plt.plot(orders, mae_, 'ro-', lw=1.5)
plt.xlabel('Orders', fontdict={"fontsize": 12})
plt.ylabel('Mean Abs Error', fontdict={"fontsize": 12})
plt.title("Absolute Error Variation Curve with Different Orders", fontdict={"fontsize": 14})
plt.grid(ls=":")
plt.show()
