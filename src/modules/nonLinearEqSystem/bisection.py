import sympy as sp
import pandas as pd


class BisectionMethod:

    """
    二分法求解非线性方程的根
    """

    def __init__(self, fun, interval: list, eps: float = 1e-6, is_print: bool = True):

        """
        参数初始化
        fun: 非线性方程,符号定义
        interval: 求根区间
        eps: 求根精度要求
        """

        self.fun = fun  # 符号定义
        self.eps = eps
        self.error = None  # 求根误差
        if len(interval) == 2:
            self.a, self.b = interval[0], interval[1]
            self.an, self.bn, self.xn = [self.a], [self.b], [(self.a + self.b) / 2]  # 区间记录
            self.res = None  # 求根结果
        else:
            raise ValueError("区间参数设置不规范，应为[a,b]！")
        self.__cal_res__()
        if is_print:
            self.__print__()

    def __cal_res__(self):

        """
        计算方程的根
        """

        t = self.fun.free_symbols.pop()  # 提取被积函数的自由变量
        fun_expr = sp.lambdify(t, self.fun)  # 将被积函数转化为lambda函数
        fa, fb = fun_expr(self.a), fun_expr(self.b)
        if fa * fb > 0:
            raise ValueError("该区间不含根！")
        i = 0  # 记录迭代次数
        while self.b - self.a > self.eps:
            i += 1
            # 计算区间中点的值
            c = (self.a + self.b) / 2
            fc = fun_expr(c)
            # 判断区间中点是否为根
            if fc == 0:
                self.res = c
                break
            # 判断下次二分时的区间所在位置
            elif fa * fc < 0:
                self.b = c
                self.an.append(self.a)
                self.bn.append(self.b)
                self.xn.append((self.a + self.b) / 2)
            else:
                self.a = c
                self.an.append(self.a)
                self.bn.append(self.b)
                self.xn.append((self.a + self.b) / 2)
        self.iter_num = i
        self.error = self.xn[-1] - self.xn[-2]  # 最终的误差
        self.res = self.xn[-1]  # 最终的解

    def __print__(self):

        """
        输出迭代的过程以及最终的解
        """

        pd.set_option('display.float_format', lambda x: '%e' % x)
        pd.set_option('display.unicode.ambiguous_as_wide', True)
        pd.set_option('display.unicode.east_asian_width', True)

        print("二分法迭代实验数据")
        print("=" * 80)
        iter_df = pd.DataFrame()
        iter_df["迭代次数"] = range(self.iter_num + 1)
        iter_df["迭代过程中a的取值(an)"] = self.an
        iter_df["迭代过程中x的取值(xn)"] = self.xn
        iter_df["迭代过程中b的取值(bn)"] = self.bn
        iter_df.set_index("迭代次数", inplace=True)
        print(iter_df)
        print("=" * 80)
        print("最终根：%e" % self.res)
        print("迭代次数：", self.iter_num)
        print("误差：%.4e" % self.error)
