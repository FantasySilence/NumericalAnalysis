import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['font.sans-serif'] = ['STSong']
matplotlib.rcParams['axes.unicode_minus'] = False



class RungeKuttaOdeMethod:

    """
    龙格-库塔方法求解常微分方程的数值解
    """

    def __init__(self, fun, interval:list, x0:float, y0:float, h:float=0.001,
                  is_print:bool=True, RK_order:int=4):

        """
        参数初始化
        fun：待求解的方程(df/dx = fun(x,y))，使用数值定义
        interval：方程的求解区间([a,b])
        x0,y0：初值条件
        h：步长
        is_print：是否打印求解结果
        RK_order：求解方式,默认为2阶龙格-库塔方法，此外还有3阶和4阶龙格-库塔方法
        """

        self.fun = fun
        if type(interval) is not list:
            raise TypeError("求解区间请以列表的形式传入")
        else:
            if len(interval) != 2:
                raise ValueError("求解区间设置不对，请以[a,b]的形式传入")
        self.interval = np.asarray(interval)
        a, b, self.h = interval[0], interval[1], h
        self.RK_order = RK_order
        self.xn, self.yn = np.arange(a, b+h, h), np.zeros(len(np.arange(a, b+h, h)))       # 储存结果
        self.x0, self.yn[0] = x0, y0
        self.__cal_res__()
        if is_print:
            self.__print__()
    

    def __cal_res__(self):

        """
        求解方程
        """

        # 2阶龙格-库塔方法(RK2)
        if self.RK_order == 2:
            for i in range(1, len(self.xn)):
                K1 = self.fun(self.xn[i-1], self.yn[i-1])
                K2 = self.fun(self.xn[i-1] + self.h/2, self.yn[i-1] + self.h*K1/2)
                self.yn[i] = self.yn[i-1] + self.h*K2

        # 3阶龙格-库塔方法(RK3)
        elif self.RK_order == 3:
            for i in range(1, len(self.xn)):
                K1 = self.fun(self.xn[i-1], self.yn[i-1])
                K2 = self.fun(self.xn[i-1] + self.h/2, self.yn[i-1] + self.h*K1/2)
                K3 = self.fun(self.xn[i-1] + self.h, self.yn[i-1] - self.h*K1 + 2*self.h*K2)
                self.yn[i] = self.yn[i-1] + self.h*(K1 + 4*K2 + K3)/6
        
        # 4阶龙格-库塔方法(RK4)
        elif self.RK_order == 4:
            for i in range(1, len(self.xn)):
                K1 = self.fun(self.xn[i-1], self.yn[i-1])
                K2 = self.fun(self.xn[i-1] + self.h/2, self.yn[i-1] + self.h*K1/2)
                K3 = self.fun(self.xn[i-1] + self.h/2, self.yn[i-1] + self.h*K2/2)
                K4 = self.fun(self.xn[i-1] + self.h, self.yn[i-1] + self.h*K3)
                self.yn[i] = self.yn[i-1] + self.h*(K1 + 2*K2 + 2*K3 + K4)/6

        else:
            raise ValueError("求解方式输入错误,仅支持2阶龙格-库塔方法(RK2), 3阶龙格-库塔方法(RK3),4阶龙格-库塔方法(RK4)")

    
    def __print__(self):

        """
        打印求解结果
        """

        pd.set_option('display.max_rows', None)
        pd.set_option('display.unicode.ambiguous_as_wide', True)
        pd.set_option('display.unicode.east_asian_width', True)

        df = pd.DataFrame()
        df["Xn"] = self.xn
        df["Yn"] = self.yn
        if self.RK_order == 2:
            print("2阶龙格-库塔方法\n求解常微分方程的数值解")
        elif self.RK_order == 3:
            print("3阶龙格-库塔方法\n求解常微分方程的数值解")
        else:
            print("4阶龙格-库塔方法\n求解常微分方程的数值解")
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
        if self.RK_order == 2:
            plt.title("2阶龙格-库塔方法\n求解常微分方程的数值解", fontsize=14)
            lab = "RK2"
        elif self.RK_order == 3:
            plt.title("3阶龙格-库塔方法\n求解常微分方程的数值解", fontsize=14)
            lab = "RK3"
        else:
            plt.title("4阶龙格-库塔方法\n求解常微分方程的数值解", fontsize=14)
            lab = "RK4"
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