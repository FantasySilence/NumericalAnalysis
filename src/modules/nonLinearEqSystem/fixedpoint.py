import sympy as sp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as axisartist
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['STSong']
matplotlib.rcParams['axes.unicode_minus'] = False



class FixedPointsIteration:

    """
    不动点迭代求解非线性方程的根
    """

    def __init__(self, iter_fun, x0, interval:list=None, eps:float=1e-5, 
                 is_print:bool=True, is_plot:bool=False):

        """
        参数初始化
        iter_fun：迭代格式
        x0：迭代初始值
        interval：求根区间
        eps：求根精度要求
        """

        self.iter_fun = iter_fun    # 符号定义
        self.eps = eps
        if x0 <= interval[1] and x0 >= interval[0]:
            self.x0 = x0
        else:
            raise ValueError("初始值不在区间[a,b]内！")
        self.error = None  # 求根误差
        if len(interval) == 2:
            self.a, self.b = interval[0], interval[1]
            self.xn, self.delta_xn = [self.x0], ["--"]  # 记录迭代结果
            self.res = None  # 求根结果
        else:
            raise ValueError("区间参数设置不规范，应为[a,b]！")
        self.__cal_res__()
        if is_print:
            self.__print__()
        if is_plot:
            self.__plot__()
        
        
    def __cal_res__(self):

        """
        计算方程的根
        """

        # TODO 代码实现对迭代格式敛散性的判断
        """
        给出迭代格式的导函数，求解导函数的绝对值小于1时x的取值
        在此基础上给出根的范围，如果可以找到一个区间，使得根存在于找到的区间
        并且找到的区间满足导函数的绝对值小于1时x的取值，那么在找到的区间中选取
        初始值，迭代格式收敛
        """
        # 将迭代格式转化为lambda函数
        fun_expr = sp.lambdify(self.iter_fun.free_symbols.pop(), self.iter_fun) 
        # 进行迭代，两个迭代值之间的误差小于精度要求，迭代结束
        while True:
            self.xn.append(fun_expr(self.xn[-1]))
            self.delta_xn.append(abs(self.xn[-1] - self.xn[-2]))
            if self.delta_xn[-1] < self.eps:
                self.res = self.xn[-1]
                self.error = self.delta_xn[-1]
                break
    

    def __plot__(self):

        """
        绘制迭代过程的蛛网图
        """

        fun_expr = sp.lambdify(self.iter_fun.free_symbols.pop(), self.iter_fun) 
        x = np.linspace(self.a, self.b, 1000)
        # 全部的P点和Q点，用于绘制迭代格式和y=x的连线
        P = np.array([self.xn[i] for i in range(len(self.xn))])
        Q = np.array([fun_expr(self.xn[i]) for i in range(len(self.xn))])

        # 创建空坐标系
        board = plt.figure(figsize=(8, 6), facecolor="white", dpi=80)
        axis = axisartist.Subplot(board, 111)
        board.add_axes(axis)
        axis.set_aspect("equal")
        axis.axis[:].set_visible(False)
        
        # x轴绘制
        # new_floating_axis(a,b)中a是纵向(0)或横向(1),b是坐标轴起始位置
        axis.axis["x"] = axis.new_floating_axis(0, self.a)    
        axis.axis["x"].set_axisline_style("->", size=1.0)
        axis.axis["x"].set_axis_direction("top")
        axis.set_xlim(self.a, self.b)

        # y轴绘制
        # new_floating_axis(a,b)中a是纵向(0)或横向(1),b是坐标轴起始位置
        axis.axis["y"] = axis.new_floating_axis(1, self.a)
        axis.axis["y"].set_axisline_style("->", size=1.0)
        axis.axis["y"].set_axis_direction("right")
        axis.set_ylim(self.a, self.b)

        plt.plot(x, fun_expr(x), lw=1, label="迭代格式")
        plt.plot(x, x, lw=1, label="y=x")
        
        # 绘制迭代格式和y=x的连线
        for i in range(1, len(self.xn)):
            plt.plot([P[i-1], Q[i-1]], [fun_expr(P[i-1]), Q[i-1]], "k--", lw=0.5, alpha=0.5)
            plt.plot([P[i], Q[i-1]], [fun_expr(P[i]), Q[i-1]], "k--", lw=0.5, alpha=0.5)
        
        # 标注部分点
        for i in range(3):
            plt.text(P[i], fun_expr(P[i]), "$P_%d$=%.2f"% (i, P[i]), fontsize=4)
            plt.text(Q[i], Q[i], "$Q_%d$=%.2f"% (i, Q[i]), fontsize=4)
        plt.title("不动点迭代实验蛛网图")    
        plt.legend(frameon=False, loc="upper left", fontsize=5)
        plt.tight_layout()
        plt.show()


    def __print__(self):

        """
        输出迭代的过程以及最终的解
        """

        pd.set_option('display.float_format', lambda x: '%e' % x)
        pd.set_option('display.unicode.ambiguous_as_wide', True)
        pd.set_option('display.unicode.east_asian_width', True)

        print("不动点迭代实验数据")
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
