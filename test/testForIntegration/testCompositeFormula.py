import os
import sys
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# 设置全局路径，以便引入模块
test_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root_path = os.path.dirname(test_path)
sys.path.append(root_path)

# 导入对应模块
from src.modules.intergration.compositeformula \
    import CompositQuadratureIntegration

# 测试用例
# 测试，求sin(x)的积分
print("=" * 50)
t = sp.Symbol('t')
int_fun = sp.sin(t)
cqi1 = CompositQuadratureIntegration(int_fun, [0, np.pi / 2], int_type="trapezoid")
int_value1 = cqi1.cal_int()
print("复合梯形公式积分值：", int_value1)
print("复合梯形公式积分的余项：", cqi1.int_remainder)
print("=" * 50)
cqi2 = CompositQuadratureIntegration(int_fun, [0, np.pi / 2], int_type="simpson")
int_value2 = cqi2.cal_int()
print("复合辛普森公式积分值：", int_value2)
print("复合辛普森公式积分的余项：", cqi2.int_remainder)
print("=" * 50)
cqi3 = CompositQuadratureIntegration(int_fun, [0, np.pi / 2], int_type="cotes")
int_value3 = cqi3.cal_int()
print("复合科特斯公式积分值：", int_value3)
print("复合科特斯公式积分的余项：", cqi3.int_remainder)
print("=" * 50)

# 分段高震荡函数的测试
import time

t = sp.Symbol('t')
fun1 = sp.exp(t ** 2)  # 分段函数的第一段，0=<x<=2
fun2 = 80 / (4 - sp.sin(16 * np.pi * t))  # 分段函数的第二段
int_fun = sp.Piecewise((fun1, t <= 2), (fun2, t <= 4))  # 构造分段函数
fun_expr = sp.lambdify(t, int_fun)  # 转化为lambda函数
plt.figure(figsize=(8, 6))
xi = np.linspace(0, 4, 500)
yi = fun_expr(xi)
plt.plot(xi, yi, "k-")
plt.fill_between(xi, yi, color="c", alpha=0.5)
plt.xlabel("X", fontdict={"fontsize": 12})
plt.ylabel("Y", fontdict={"fontsize": 12})
plt.title("分段高震荡函数的积分区间", fontdict={"fontsize": 14})
plt.show()

internal_num = np.arange(1000, 101001, 2000)
a = 57.764450125048512  # 测试函数的一个高精度积分值
for num in internal_num:
    cqi4 = CompositQuadratureIntegration(int_fun, [0, 4], internal_num=num, int_type="cotes")
    start = time.time()
    int_value4 = cqi4.cal_int()
    end = time.time()
    print("划分区间数%d,积分值%.15f,误差%.10e,运行消耗时间%.10fs"
          % (num, int_value4, a - int_value4, end - start))
