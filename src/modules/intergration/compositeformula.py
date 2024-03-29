import numpy as np
import sympy as sp
import matplotlib
from scipy import optimize

matplotlib.rcParams['font.sans-serif'] = ['STSong']
matplotlib.rcParams['axes.unicode_minus'] = False


class CompositQuadratureIntegration:
    """
    复合求积公式：复合梯形公式，复合辛普森公式，复合科特斯公式
    """

    def __init__(self, int_fun, int_internal, internal_num=16, int_type="simpson"):

        """
        必要的参数初始化
        int_fun: 被积函数
        int_internal: 积分区间
        internal_num: 积分区间的分割数量
        int_type: 积分公式的类型
                    "simpson": 复合辛普森公式
                    "trapezoid": 复合梯形公式
                    "cotes": 复合科特斯公式
        """

        self.int_fun = int_fun  # 符号定义的被积函数
        if len(int_internal) == 2:
            self.a, self.b = int_internal[0], int_internal[1]  # 积分区间
        else:
            raise ValueError("积分区间参数设置不规范，应为[a,b]！")
        self.n = int(internal_num)  # 默认等分子区间数为16
        self.int_type = int_type  # 积分公式类型，默认采用辛普森复合求积公式
        self.int_value = None  # 积分值结果
        self.int_remainder = None  # 积分余项值

    def cal_int(self):

        """
        根据参数设置选择不同的积分类型
        """

        t = self.int_fun.free_symbols.pop()  # 提取被积函数的自由变量
        fun_expr = sp.lambdify(t, self.int_fun)  # 将被积函数转化为lambda函数
        if self.int_type == "trapezoid":
            self.int_value = self.__cal_trapezoid__(t, fun_expr)
        elif self.int_type == "simpson":
            self.int_value = self.__cal_simpson__(t, fun_expr)
        elif self.int_type == "cotes":
            self.int_value = self.__cal_cotes__(t, fun_expr)
        else:
            raise ValueError("复合积分类型仅支持trapezoid,simpson,cotes")
        return self.int_value

    def __cal_trapezoid__(self, t, fun_expr):

        """
        复合梯形公式
        t: 自由变量
        fun_expr: 内积函数
        """

        h = (self.b - self.a) / self.n  # 子区间步长
        x_k = np.linspace(self.a, self.b, self.n + 1)  # 等分点
        f_val = fun_expr(x_k)  # 对应等分点的函数值
        int_value = h / 2 * (f_val[0] + f_val[-1] + 2 * sum(f_val[1:-1]))  # 复合梯形公式
        # 计算积分余项
        diff_fun = self.int_fun.diff(t, 2)  # 被积函数的二阶导数
        max_val = self.__fun_maximize__(t, diff_fun)  # 求函数的最大值
        self.int_remainder = (self.b - self.a) / 12 * h ** 2 * max_val  # 积分余项
        return int_value

    def __cal_simpson__(self, t, fun_expr):

        """
        复合辛普森公式
        t: 自由变量
        fun_expr: 内积函数
        """

        h = (self.b - self.a) / 2 / self.n  # 子区间步长
        x_k = np.linspace(self.a, self.b, 2 * self.n + 1)  # 等分点
        f_val = fun_expr(x_k)  # 对应等分点的函数值
        idx = np.linspace(0, 2 * self.n, 2 * self.n + 1, dtype=np.int64)  # 节点的索引下标
        f_val_even = f_val[np.mod(idx, 2) == 0]  # 子区间的端点值
        f_val_odd = f_val[np.mod(idx, 2) == 1]  # 子区间的中点值
        int_value = h / 3 * (f_val[0] + f_val[-1] + 2 * sum(f_val_even[1:-1]) + 4 * sum(f_val_odd))  # 复合辛普森公式
        # 计算积分余项
        diff_fun = self.int_fun.diff(t, 4)  # 被积函数的二阶导数
        max_val = self.__fun_maximize__(t, diff_fun)  # 求函数的最大值
        self.int_remainder = (self.b - self.a) / 180 * (h / 2) ** 4 * max_val  # 积分余项
        return int_value

    def __cal_cotes__(self, t, fun_expr):

        """
        复合科特斯公式
        t: 自由变量
        fun_expr: 内积函数
        """

        h = (self.b - self.a) / 4 / self.n  # 子区间步长
        x_k = np.linspace(self.a, self.b, 4 * self.n + 1)  # 等分点
        f_val = fun_expr(x_k)  # 对应等分点的函数值
        idx = np.linspace(0, 4 * self.n, 4 * self.n + 1, dtype=np.int64)  # 节点的索引下标
        f_val_0 = f_val[np.mod(idx, 4) == 0]  # 下标为4k
        f_val_1 = f_val[np.mod(idx, 4) == 1]  # 下标为4k+1
        f_val_2 = f_val[np.mod(idx, 4) == 2]  # 下标为4k+2
        f_val_3 = f_val[np.mod(idx, 4) == 3]  # 下标为4k+3
        # 复合科斯特公式
        int_value = 2 * h / 45 * (7 * (f_val[0] + f_val[-1]) + 14 * sum(f_val_0[1:-1]) +
                                  32 * (sum(f_val_1) + sum(f_val_3)) + 12 * sum(f_val_2))
        # 计算积分余项
        diff_fun = self.int_fun.diff(t, 6)  # 被积函数的二阶导数
        max_val = self.__fun_maximize__(t, diff_fun)  # 求函数的最大值
        self.int_remainder = (self.b - self.a) / 945 * 2 * h ** 6 * max_val  # 积分余项
        return int_value

    def __fun_maximize__(self, t, diff_fun):

        """
        求函数的最大值
        diff_fun: 被积函数的n阶导数
        """

        fun_min = sp.lambdify(t, -diff_fun)
        res = optimize.minimize_scalar(fun_min, bounds=(self.a, self.b), method="Bounded")
        return -res.fun
