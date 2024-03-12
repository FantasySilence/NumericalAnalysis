import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['STSong']
matplotlib.rcParams['axes.unicode_minus'] = False

from utilsFiles.filesio import FilesIO


class SolutionToQuesFive:

    """
    第五问
    """

    def __init__(self, fun, interval: list, x0: float, y0: float, h: float = 0.001, is_print: bool = True):

        """
        参数初始化\n
        fun：待求解的方程(df/dx = fun(x,y))，使用数值定义
        interval：方程的求解区间([a,b])
        x0,y0：初值条件
        h：步长
        is_print：是否打印求解结果
        """

        self.fun = fun
        if type(interval) is not list:
            raise TypeError("求解区间请以列表的形式传入")
        else:
            if len(interval) != 2:
                raise ValueError("求解区间设置不对，请以[a,b]的形式传入")
        self.interval = np.asarray(interval)
        a, b, self.h = interval[0], interval[1], h
        self.xn, self.yn = np.arange(a, b + h, h), np.zeros(len(np.arange(a, b + h, h)))  # 储存结果
        self.x0, self.yn[0] = x0, y0
        self.res_df = None  # 最终结果，以DataFrame的形式呈现
        self.__cal_res__()
        if is_print:
            self.__print__()

    def __cal_res__(self):

        """
        计算结果
        """

        for i in range(1, len(self.xn)):
            K1 = self.fun(self.xn[i - 1], self.yn[i - 1])
            K2 = self.fun(self.xn[i - 1] + self.h / 2, self.yn[i - 1] + self.h * K1 / 2)
            K3 = self.fun(self.xn[i - 1] + 3 * self.h / 4, self.yn[i - 1] + 3 * self.h * K2 / 4)
            self.yn[i] = self.yn[i - 1] + self.h * (2 * K1 + 3 * K2 + 4 * K3) / 9

        # 最终结果，以DataFrame的形式呈现
        df = pd.DataFrame()
        df["Xn"] = self.xn
        df["Yn"] = self.yn
        self.res_df = df

    def __print__(self):

        """
        打印求解结果
        """

        pd.set_option('display.max_rows', None)
        pd.set_option('display.unicode.ambiguous_as_wide', True)
        pd.set_option('display.unicode.east_asian_width', True)

        print("该方程的数值解为")
        print("-" * 20)
        print(self.res_df)
        print("-" * 20)

    def plot(self, is_show: bool = True):

        """
        绘制结果
        is_show: 是否显示图像
        """

        if is_show:
            plt.figure(figsize=(8, 6), facecolor="white", dpi=80)
            plt.title("该方程的数值解", fontsize=14)
        x = np.asarray(self.xn)
        y = np.asarray(self.yn)
        plt.plot(x, y, 'ro-', linewidth=1.5, markersize=2, label="数值解")
        plt.xlabel("x", fontsize=12)
        plt.ylabel("y", fontsize=12)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        if is_show:
            plt.savefig(FilesIO.getSavePath("image_5.png"))
            plt.show()
