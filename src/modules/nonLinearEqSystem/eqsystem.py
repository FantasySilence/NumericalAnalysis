import sympy as sp
import numpy as np
import pandas as pd


class NonLinearSystem:

    """
    求解非线性方程组
    主要分不动点迭代法和牛顿迭代法求解
    """

    def __init__(self, A, x0: list, eps: float = 1e-6,
                 is_print: bool = True, solve_type: str = "newton"):

        """
        参数初始化
        A：方程组矩阵,如果选用不动点迭代传入迭代格式,否则直接传入方程
        x0：迭代初始值(向量), 一般为0
        eps: 求根精度
        is_print：是否打印迭代过程
        solve_type：求解方式，默认为牛顿迭代法
        """

        self.A = A
        if len(x0) != len(A):
            raise ValueError("迭代初始值向量设置不规范,长度与方程组不一致！")
        self.x0 = np.asarray(x0).T  # 迭代初始值(向量)
        self.is_print = is_print
        self.solve_type = solve_type
        self.eps = eps
        self.res = None  # 求解结果
        self.error = None  # 求解误差
        self.xn, self.delta_xn = [x0], [" "]  # 迭代值和误差值
        self.label = None  # 记录迭代结果中x的角标
        self.__cal_res__()
        if is_print:
            self.__print__()

    def __cal_res__(self):

        """
        求解方程组的根
        """

        if self.solve_type.lower() == "newton":
            self.__newton_iter__()
        elif self.solve_type.lower() == "broyden":
            self.__broyden_iter__()
        else:
            raise ValueError("求解方式输入错误,仅支持牛顿迭代法(newton),不动点迭代法(broyden)")

    def __newton_iter__(self):

        """
        非线性方程组的牛顿迭代法
        """

        # TODO 代码实现牛顿迭代法的部分错误(零除错误,陷入死循环)避免
        # 将迭代格式转化为lambda函数
        t = list(self.A.free_symbols)
        # 记录变量顺序，可能不是x1,x2,...
        self.label = t
        # 转换为lambda函数
        F = sp.lambdify(t, self.A, modules="numpy")
        F_j = sp.lambdify(t, self.A.jacobian(t), modules="numpy")
        # 进行迭代，两个迭代值之间的误差小于精度要求，迭代结束
        while True:
            # 带入向量函数后，结果需要降维然后再进行下一步迭代
            self.xn.append(self.xn[-1] - np.linalg.inv(F_j(*self.xn[-1])) @ F(*self.xn[-1]).reshape(-1))
            self.delta_xn.append(np.linalg.norm(self.xn[-1] - self.xn[-2], 2))
            if self.delta_xn[-1] < self.eps:
                self.res = self.xn[-1]
                self.error = self.delta_xn[-1]
                break

    def __broyden_iter__(self):

        """
        非线性方程组的不动点迭代法
        """

        # TODO 代码实现对迭代格式敛散性的判断
        # 将迭代格式转化为lambda函数
        t = list(self.A.free_symbols)
        # 记录变量顺序，可能不是x1,x2,...
        self.label = t
        iter_fun = sp.lambdify(t, self.A, modules="numpy")
        # 进行迭代，两个迭代值之间的误差小于精度要求，迭代结束
        while True:
            # 带入向量函数后，结果需要降维然后再进行下一步迭代
            self.xn.append(iter_fun(*self.xn[-1]).reshape(-1))
            self.delta_xn.append(np.linalg.norm(self.xn[-1] - self.xn[-2], 2))
            if self.delta_xn[-1] < self.eps:
                self.res = self.xn[-1]
                self.error = self.delta_xn[-1]
                break

    def __print__(self):

        """
        打印迭代过程
        """

        pd.set_option('display.float_format', lambda x: '%e' % x)
        pd.set_option('display.unicode.ambiguous_as_wide', True)
        pd.set_option('display.unicode.east_asian_width', True)

        if self.solve_type.lower() == "broyden":
            print("不动点迭代法解非线性方程组实验数据")
        else:
            print("牛顿迭代法解非线性方程组实验数据")
        print("=" * 80)
        iter_df = pd.DataFrame()
        iter_df["迭代次数"] = range(len(self.xn))
        iter_df["迭代过程中x的取值(%s)" % self.label] = self.xn
        iter_df["迭代过程中的误差"] = self.delta_xn
        iter_df.set_index("迭代次数", inplace=True)
        print(iter_df)
        print("=" * 80)
        print("最终根(%s)：" % self.label, self.res)
        print("误差：%.4e" % self.error)
