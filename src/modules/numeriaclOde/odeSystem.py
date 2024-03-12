import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['STSong']
matplotlib.rcParams['axes.unicode_minus'] = False


class FirstOrderOde:

    """
    求解一阶微分方程组
    采用欧拉法以及2-4阶龙格-库塔法
    """

    def __init__(self, fun, interval: list, Y0: list, h: float = 0.001,
                 is_print: bool = True, solve_type: str = "RK4"):

        """
        参数初始化\n
        fun：待求解的方程组，为数值定义的向量函数\n
        Y0：初值条件，向量，维度以及数字位置与向量函数中元素一致\n
        interval：方程的求解区间([a,b])\n
        h：步长\n
        is_print：是否打印求解结果\n
        solve_type：求解方式,默认为欧拉法("Euler")，此外还有2-4阶龙格-库塔法("RK2", "RK3", "RK4")
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
        self.tn = np.arange(a, b + h, h)  # 方程组的自变量
        self.Yn = np.zeros((len(Y0), len(self.tn)))  # 储存结果
        self.Yn[:, 0] = Y0
        self.__cal_res__()
        if is_print:
            self.__print__()

    def __cal_res__(self):

        """
        计算方程组的解
        """

        # 改进的欧拉法
        if self.solve_type == "Euler":
            for i in range(1, len(self.tn)):
                K1 = self.fun(*self.Yn[:, i - 1], self.tn[i - 1])
                K2 = self.fun(*(self.Yn[:, i - 1] + self.h * K1 / 2), self.tn[i - 1] + self.h / 2)
                self.Yn[:, i] = self.Yn[:, i - 1] + self.h * (K1 + K2) / 2

        # 2阶龙格-库塔法(RK2)
        elif self.solve_type == "RK2":
            for i in range(1, len(self.tn)):
                K1 = self.fun(*self.Yn[:, i - 1], self.tn[i - 1])
                K2 = self.fun(*(self.Yn[:, i - 1] + self.h * K1 / 2), self.tn[i - 1] + self.h / 2)
                self.Yn[:, i] = self.Yn[:, i - 1] + self.h * K2

        # 3阶龙格-库塔法(RK3)
        elif self.solve_type == "RK3":
            for i in range(1, len(self.tn)):
                K1 = self.fun(*self.Yn[:, i - 1], self.tn[i - 1])
                K2 = self.fun(*(self.Yn[:, i - 1] + self.h * K1 / 2), self.tn[i - 1] + self.h / 2)
                K3 = self.fun(*(self.Yn[:, i - 1] - self.h * K1 + 2 * self.h * K2), self.tn[i - 1] + self.h)
                self.Yn[:, i] = self.Yn[:, i - 1] + self.h * (K1 + 4 * K2 + K3) / 6

        # 4阶龙格-库塔法(RK4)
        elif self.solve_type == "RK4":
            for i in range(1, len(self.tn)):
                K1 = self.fun(*self.Yn[:, i - 1], self.tn[i - 1])
                K2 = self.fun(*(self.Yn[:, i - 1] + self.h * K1 / 2), self.tn[i - 1] + self.h / 2)
                K3 = self.fun(*(self.Yn[:, i - 1] + self.h * K2 / 2), self.tn[i - 1] + self.h / 2)
                K4 = self.fun(*(self.Yn[:, i - 1] + self.h * K3), self.tn[i - 1] + self.h)
                self.Yn[:, i] = self.Yn[:, i - 1] + self.h * (K1 + 2 * K2 + 2 * K3 + K4) / 6

        else:
            raise ValueError("求解方式输入错误,仅支持欧拉法(Euler)，2-4阶龙格-库塔法(RK2, RK3, RK4)")

    def __print__(self):

        """
        打印输出结果
        """

        pd.set_option('display.max_rows', None)
        pd.set_option('display.unicode.ambiguous_as_wide', True)
        pd.set_option('display.unicode.east_asian_width', True)

        df = pd.DataFrame()
        df["t"] = self.tn
        for i in range(len(self.Yn)):
            df["y" + str(i + 1)] = self.Yn[i, :]
        if self.solve_type == "Euler":
            print("显式欧拉法求解\n一阶方程组的数值解")
        elif self.solve_type == "RK2":
            print("2阶龙格-库塔法\n一阶方程组的数值解")
        elif self.solve_type == "RK3":
            print("3阶龙格-库塔法\n一阶方程组的数值解")
        else:
            print("4阶龙格-库塔法\n一阶方程组的数值解")
        print("=" * 20)
        print(df)
        print("=" * 20)

    def plot(self, is_show: bool = True):

        """
        绘制结果
        is_show：是否显示图像
        """

        if is_show:
            plt.figure(figsize=(16, 12), dpi=300)
        if self.solve_type == "Euler":
            plt.title("显式欧拉法\n求解一阶方程组的数值解", fontsize=14)
        elif self.solve_type == "RK2":
            plt.title("2阶龙格-库塔法\n求解一阶方程组的数值解", fontsize=14)
        elif self.solve_type == "RK3":
            plt.title("3阶龙格-库塔法\n求解一阶方程组的数值解", fontsize=14)
        else:
            plt.title("4阶龙格-库塔法\n求解一阶方程组的数值解", fontsize=14)
        t = np.asarray(self.tn)
        for i in range(self.Yn.shape[0]):
            y = np.asarray(self.Yn[i, :])
            plt.plot(t, y, linewidth=1.5, label="y" + str(i + 1))
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.xlabel("t", fontsize=12)
        plt.ylabel("F", fontsize=12)
        plt.legend(frameon=False)
        if is_show:
            plt.show()
