import sympy as sp
import pandas as pd



class SecantMethod:

    """
    弦截法求解非线性方程的解
    """

    def __init__(self, fun, x0, x1, eps:float=1e-8, is_print:bool=True):

        """
        参数初始化
        fun：方程
        x0, x1：迭代初始值(两个)
        eps：求根精度要求
        is_print：是否打印迭代过程
        """

        self.fun = fun    # 符号定义
        self.eps = eps
        self.error = None  # 求根误差  
        self.x0, self.x1 = x0, x1
        self.xn, self.delta_xn = [self.x0, self.x1], ["————", "————"]  # 记录迭代结果
        self.res = None  # 最终根
        self.__cal_res__()
        if is_print:
            self.__print__()
    
    
    def __cal_res__(self):

        """
        计算方程的根
        """

        t = self.fun.free_symbols.pop()
        # 将方程转化为lambda函数
        fun_expr = sp.lambdify(t, self.fun)
        # 进行迭代，两个迭代值之间的误差小于精度要求，迭代结束
        while True:
            # 弦截法的迭代格式
            x = self.xn[-1] -\
                  ((self.xn[-1]-self.xn[-2])/(fun_expr(self.xn[-1])-fun_expr(self.xn[-2])))*fun_expr(self.xn[-1])
            self.xn.append(x)
            self.delta_xn.append(abs(self.xn[-1]-self.xn[-2]))
            if self.delta_xn[-1] < self.eps:
                self.res = self.xn[-1]
                self.error = self.delta_xn[-1]
                break
        

    def __print__(self):

        """
        输出迭代的过程以及最终的解
        """

        pd.set_option('display.float_format', lambda x: '%e' % x)
        pd.set_option('display.unicode.ambiguous_as_wide', True)
        pd.set_option('display.unicode.east_asian_width', True)

        print("弦截法实验数据")
        print("="*80)
        iter_df = pd.DataFrame()
        iter_df["迭代次数"] = range(len(self.xn))
        iter_df["迭代过程中x的取值(xn)"] = self.xn
        iter_df["迭代过程中的误差"] = self.delta_xn
        iter_df.set_index("迭代次数", inplace=True)
        print(iter_df)
        print("="*80)
        print("最终根：%e"%self.res)
        print("误差：%.4e"%self.error)