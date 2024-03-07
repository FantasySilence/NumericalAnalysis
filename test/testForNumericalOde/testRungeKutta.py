import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# 设置全局路径，以便引入模块
test_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root_path = os.path.dirname(test_path)
sys.path.append(root_path)

# 导入对应模块
from src.modules.numeriaclOde.rungekutta import RungeKuttaOdeMethod

# 测试用例
def f(x, y):
    if x == 0 and y == 0:
        return 1
    else:
        return y/x - 2*y**2

# 解析解
def g(x):
    return x/(1+x**2)

plt.figure(figsize=(10,6))
x = np.linspace(0,3,100)
plt.plot(x, g(x), 'k-', linewidth=1.5, label="解析解")
test1 = RungeKuttaOdeMethod(fun=f, interval=[0,3], x0=0, y0=0, h=0.2, is_print=False, RK_order=2)
test1.plot(is_show=False)
test2 = RungeKuttaOdeMethod(fun=f, interval=[0,3], x0=0, y0=0, h=0.2, is_print=False, RK_order=3)
test2.plot(is_show=False)
test3 = RungeKuttaOdeMethod(fun=f, interval=[0,3], x0=0, y0=0, h=0.2, is_print=True, RK_order=4)
test3.plot(is_show=False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.title("解析解与RK解的比较")
plt.legend(frameon=False, loc="lower right")
plt.show()