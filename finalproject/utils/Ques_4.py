import sympy as sp
import pandas as pd


class SolutionToQuesFour:

    """
    第四问，采用牛顿迭代法
    """

    def __init__(self, fun, x0, eps: float = 1e-12, is_print: bool = True):

        """
        参数初始化\n
        fun：带求根的方程\n
        x0：迭代初始值\n
        interval：求根区间\n
        eps：求根精度要求\n
        """

        self.fun = fun  # 符号定义
        self.eps = eps
        self.error = None  # 求根误差  
        self.res = None  # 求根结果
        self.x0 = x0
        self.xn, self.delta_xn = [self.x0], [" "]  # 记录迭代结果
        self.__cal_res__()
        if is_print:
            self.__print__()

    def __cal_res__(self):

        """
        计算方程的根
        """

        t = self.fun.free_symbols.pop()
        # 牛顿迭代法的迭代格式
        iter_fun = t - self.fun / sp.diff(self.fun, t)
        # 将方程转化为lambda函数
        iter_fun = sp.lambdify(iter_fun.free_symbols.pop(), iter_fun)
        # 进行迭代，两个迭代值之间的误差小于精度要求，迭代结束
        while True:
            self.xn.append(iter_fun(self.xn[-1]))
            self.delta_xn.append(abs(self.xn[-1] - self.xn[-2]))
            if self.delta_xn[-1] < self.eps:
                self.res = self.xn[-1]
                self.error = self.delta_xn[-1]
                break

    def __print__(self):

        """
        打印输出结果
        """

        pd.set_option('display.float_format', lambda x: '%e' % x)
        pd.set_option('display.unicode.ambiguous_as_wide', True)
        pd.set_option('display.unicode.east_asian_width', True)

        print("牛顿迭代实验数据")
        print("=" * 80)
        iter_df = pd.DataFrame()
        iter_df["迭代次数"] = range(len(self.xn))
        iter_df["迭代过程中x的取值(xn)"] = self.xn
        iter_df["迭代过程中的误差"] = self.delta_xn
        iter_df.set_index("迭代次数", inplace=True)
        print(iter_df)
        print("=" * 80)
        print("最终根：%e" % self.res)
        print("误差：%.4e" % self.error)
