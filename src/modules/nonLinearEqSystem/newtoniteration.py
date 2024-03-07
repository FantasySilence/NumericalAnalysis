import sympy as sp
import pandas as pd



class NewtonIterationMethod:

    """
    牛顿迭代法求解非线性方程的解
    包含一般的牛顿迭代法，简化的牛顿迭代法，牛顿下山法
    """

    def __init__(self, fun, x0, eps:float=1e-12, is_print:bool=True, iter_type:str="default"):

        """
        参数初始化\n
        fun：带求根的方程\n
        x0：迭代初始值\n
        interval：求根区间\n
        eps：求根精度要求\n
        iter_type： 迭代类型，默认为default，可选简化的牛顿迭代法(simnewton)，牛顿下山法(offhill)
        """

        self.fun = fun    # 符号定义
        self.eps = eps
        self.error = None  # 求根误差  
        self.res = None    # 求根结果
        self.x0 = x0
        self.iter_type = iter_type
        self.xn, self.delta_xn = [self.x0], [" "]  # 记录迭代结果
        self.__cal_res__()
        if is_print:
            self.__print__()
        
    
    def __cal_res__(self):

        """
        计算方程的根
        """

        if self.iter_type.lower() == "default":
            self.__default_iter__()
        elif self.iter_type.lower() == "simnewton":
            self.__simnewton_iter__()
        elif self.iter_type.lower() == "offhill":
            self.__offhill_iter__()
        else:
            raise ValueError("迭代类型输入错误,仅支持一般的牛顿迭代法(default),简化的牛顿迭代法(simnewton),牛顿下山法(offhill)")
            

    def __default_iter__(self):

        """
        一般的牛顿迭代法
        """

        # TODO 代码实现牛顿迭代法的部分错误(零除错误,陷入死循环)避免
        t = self.fun.free_symbols.pop()
        # 牛顿迭代法的迭代格式
        iter_fun = t - self.fun/sp.diff(self.fun, t)
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


    def __simnewton_iter__(self):
        
        """
        简化的牛顿迭代法
        """

        t = self.fun.free_symbols.pop()
        dfun = sp.diff(self.fun, t)
        # 简化的牛顿迭代法的迭代格式
        iter_fun = t - self.fun/sp.lambdify(dfun.free_symbols.pop(), dfun)(self.x0)
        # 将方程转化为lambda函数
        iter_fun = sp.lambdify(iter_fun.free_symbols.pop(), iter_fun)
        # 进行迭代,两个迭代值之间的误差小于精度要求，迭代结束
        while True:
            self.xn.append(iter_fun(self.xn[-1]))
            self.delta_xn.append(abs(self.xn[-1] - self.xn[-2]))
            if self.delta_xn[-1] < self.eps:
                self.res = self.xn[-1]
                self.error = self.delta_xn[-1]
                break


    def __offhill_iter__(self):
        
        """
        牛顿下山法
        """

        # TODO 牛顿下山法的具体过程仍存疑
        a = 1   # 下山因子
        self.offhill_number = [a]   # 记录下山因子的变化情况
        t = self.fun.free_symbols.pop()
        # 进行迭代,两个迭代值之间的误差小于精度要求，迭代结束
        while True:
            # 牛顿下山法的迭代格式
            iter_fun = t - a*(self.fun/sp.diff(self.fun, t))
            # 将方程转化为lambda函数
            iter_fun = sp.lambdify(iter_fun.free_symbols.pop(), iter_fun)
            self.xn.append(iter_fun(self.xn[-1]))
            self.delta_xn.append(abs(self.xn[-1] - self.xn[-2]))
            a *= 0.5   # 下山因子缩小为原来的1/2
            self.offhill_number.append(a)
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
        print("="*80)
        iter_df = pd.DataFrame()
        iter_df["迭代次数"] = range(len(self.xn))
        iter_df["迭代过程中x的取值(xn)"] = self.xn
        iter_df["迭代过程中的误差"] = self.delta_xn
        if self.iter_type.lower() == "offhill":
            iter_df["下山因子"] = self.offhill_number
        iter_df.set_index("迭代次数", inplace=True)
        print(iter_df)
        print("="*80)
        print("最终根：%e"%self.res)
        print("误差：%.4e"%self.error)