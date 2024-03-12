import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# =======第2题=======#
from utils.Ques_2 import SolutionToQuesTwo

print("#=======第2题=======#")
t = sp.Symbol("t")
# 勒让德多项式
fun_list = [1, t, (3 * t ** 2 - 1) / 2, (5 * t ** 3 - 3 * t) / 2]
# 题目中给出的离散数据点
x = np.array([0, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9, 1])
y = np.array([1, 1.25, 1.3, 1.35, 1.6, 1.8, 2.6, 3])
# 由于勒让德多项式多项式的权函数为1，故计算时采用默认值
sol_2 = SolutionToQuesTwo(x, y, k=3, fun_list=fun_list)
sol_2.fit_curve()
print("拟合多项式为：{}".format(sol_2.fit_poly))
sol_2.plt_curve_fit()
print("=" * 20)
print()

# =======第3题=======#
from utils.Ques_3 import SolutionToQuesThree

print("#=======第3题=======#")
mat = np.array([[20, 5, 1, 24], [1, 8, 1, 9], [3, -3, 18, 4]])  # 方程组的增广矩阵
sol_3 = SolutionToQuesThree(
    matrix=mat, x0=None, max_iter=1000, epsilon=5e-6, is_print=True
)
# 计算迭代矩阵的谱半径
eigval, _ = np.linalg.eig(sol_3.G)
spe_radius = np.max(np.abs(eigval))
if spe_radius < 1:
    print("迭代矩阵的谱半径为：", spe_radius)
    print("迭代矩阵的谱半径小于1，迭代矩阵收敛")
print("=" * 20)
print()

# =======第4题=======#
from utils.Ques_4 import SolutionToQuesFour

x = sp.Symbol("x")
fun = x * sp.exp(3 * x) - 2
sol_4 = SolutionToQuesFour(fun=fun, x0=0.5, eps=1e-15, is_print=True)
print()

# =======第5题=======#
from utils.Ques_5 import SolutionToQuesFive
from utilsFiles.filesio import FilesIO

print("#=======第5题=======#")


# 题目中的微分方程
def f(x, y):
    return -(x ** 2) * y ** 2


# 题目中微分方程的精确解
def g(x):
    return 3 / (1 + x ** 3)


sol_5 = SolutionToQuesFive(fun=f, interval=[0, 1.5], x0=0, y0=3, h=0.1, is_print=False)
com_df = sol_5.res_df
com_df["解析解"] = com_df["Xn"].apply(g)
com_df["误差"] = com_df["解析解"] - com_df["Yn"]
print("最终结果")
print("-" * 20)
print(com_df)
plt.figure(figsize=(8, 6), facecolor="white", dpi=80)
sol_5.plot(is_show=False)
x = np.linspace(0, 1.5, 100)
plt.plot(x, g(x), "b-", alpha=0.5, lw=1.5, label="解析解")
plt.xlabel("x", fontsize=12)
plt.ylabel("y", fontsize=12)
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["top"].set_visible(False)
plt.legend(loc="best", frameon=False)
plt.savefig(FilesIO.getSavePath("image_5.png"))
plt.show()
print("-" * 20)
print("=" * 20)
