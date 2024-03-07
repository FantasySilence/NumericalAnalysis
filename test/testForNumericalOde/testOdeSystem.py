import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# 设置全局路径，以便引入模块
test_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root_path = os.path.dirname(test_path)
sys.path.append(root_path)

# 导入对应模块
from src.modules.numeriaclOde.odeSystem import FirstOrderOde

# 测试用例
@np.vectorize
def F(x,y,t):
    return np.array([3*x+5*y+np.exp(-t),
                    -5*x+3*y])

plt.figure(figsize=(16,6), facecolor="white", dpi=80)
plt.suptitle("求解一阶微分方程组", fontsize=20)

# 绘制方程组的数值解
plt.subplot(1,2,1)
model = FirstOrderOde(fun=F, interval=[0,2], Y0=[0,1], solve_type="RK4", is_print=False)
model.plot(is_show=False)

# 绘制方程组的解析解
plt.subplot(1,2,2)
def f1(x):
    return np.exp(3*x)*(4*np.cos(5*x)+46*np.sin(5*x)-4*np.exp(-4*x))/41
def f2(x):
    return np.exp(3*x)*(46*np.cos(5*x)-4*np.sin(5*x)-5*np.exp(-4*x))/41
plt.plot(model.tn, f1(model.tn), label="f1")
plt.plot(model.tn, f2(model.tn), label="f2")
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.title("解析解f(t)", fontsize=14)
plt.xlabel("t", fontsize=12)
plt.ylabel("f", fontsize=12)
plt.legend(frameon=False)
plt.show()