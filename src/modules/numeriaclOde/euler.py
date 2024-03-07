import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['STSong']
matplotlib.rcParams['axes.unicode_minus'] = False



class EulerOdeMethod:

    """
    欧拉法(改进的欧拉法)求解常微分方程的数值解
    """

    def __init__(self, fun, interval:list, x0:float, y0:float, h:float=0.001,
                  is_print:bool=True, solve_type:str="euler"):

        """
        参数初始化
        fun：待求解的方程(df/dx = fun(x,y))，使用数值定义
        interval：方程的求解区间([a,b])
        x0,y0：初值条件
        h：步长
        is_print：是否打印求解结果
        solve_type：求解方式,默认为欧拉法(euler）, 此外还有改进的欧拉法(improved_euler)
        """

        self.fun = fun
        if type(interval) is not list:
            raise TypeError("求解区间请以列表的形式传入")
        else:
            if len(interval) != 2:
                raise ValueError("求解区间设置不对，请以[a,b]的形式传入")
        self.interval = np.asarray(interval)
        a, b, self.h = interval[0], interval[1], h
        self.solve_type = solve_type
        self.xn, self.yn = np.arange(a, b+h, h), np.zeros(len(np.arange(a, b+h, h)))       # 储存结果
        self.x0, self.yn[0] = x0, y0
        self.__cal_res__()
        if is_print:
            self.__print__()

    
    def __cal_res__(self):

        """
        求解方程
        """

        # 显式欧拉法
        if self.solve_type.lower() == "euler":
            for i in range(1, len(self.xn)):
                self.yn[i] = self.yn[i-1] + self.h*self.fun(self.xn[i-1], self.yn[i-1])

        # 改进的欧拉法
        elif self.solve_type.lower() == "improved_euler":
            for i in range(1, len(self.xn)):
                # 预测
                y_hat = self.yn[i-1] + self.h*self.fun(self.xn[i-1], self.yn[i-1])
                # 校正
                self.yn[i] = self.yn[i-1] + \
                    self.h*(self.fun(self.xn[i-1], self.yn[i-1]) + self.fun(self.xn[i], y_hat))/2
                
        else:
            raise ValueError("求解方式输入错误,仅支持欧拉法(euler), 改进欧拉法(improved_euler)")


    def __print__(self):

        """
        打印输出结果
        """

        pd.set_option('display.max_rows', None)
        pd.set_option('display.unicode.ambiguous_as_wide', True)
        pd.set_option('display.unicode.east_asian_width', True)

        df = pd.DataFrame()
        df["Xn"] = self.xn
        df["Yn"] = self.yn
        if self.solve_type.lower() == "euler":
            print("欧拉法求解常微分方程的数值解")
        else:
            print("改进的欧拉法求解常微分方程的数值解")
        print("="*20)
        print(df)
        print("="*20)
        

    def plot(self, is_show:bool=True):

        """
        绘制结果
        is_show: 是否显示图像
        """

        if is_show:
            plt.figure(figsize=(16,12), dpi=160)
        if self.solve_type.lower() == "euler":
            plt.title("显式欧拉法\n求解常微分方程的数值解", fontsize=14)
            lab = "显式欧拉法"
        else:
            self.solve_type.lower() == "improved_euler"
            plt.title("改进的欧拉法\n求解常微分方程的数值解", fontsize=14)
            lab = "改进的欧拉法"
        x = np.asarray(self.xn)
        y = np.asarray(self.yn)
        plt.plot(x, y, 'o-', linewidth=1.5, markersize=4, label=lab)
        plt.xlabel("x", fontsize=12)
        plt.ylabel("y", fontsize=12)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.legend(frameon=False)
        if is_show:
            plt.show()