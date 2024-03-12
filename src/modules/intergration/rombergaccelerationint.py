import numpy as np
import matplotlib
import mpmath

matplotlib.rcParams['font.sans-serif'] = ['STSong']
matplotlib.rcParams['axes.unicode_minus'] = False


class RombergAccelerationQuadrature:

    """
    龙贝格加速法求数值积分
    """

    def __init__(self, int_fun, int_internal, acceleration_num=10):

        """
        必要的参数初始化
        int_fun: 被积函数
        int_internal: 积分区间
        acceleration_num: 外推次数
        """

        self.int_fun = int_fun  # 符号定义的被积函数
        if len(int_internal) == 2:
            self.a, self.b = int_internal[0], int_internal[1]  # 积分区间
        else:
            raise ValueError("积分区间参数设置不规范，应为[a,b]！")
        self.n = int(acceleration_num)  # 默认外推次数为10
        self.int_value = None  # 积分值结果
        self.romberg_table = None  # 龙贝格加速计算的表格

    def cal_int(self):

        """
        龙贝格求积公式的核心算法
        """

        self.romberg_table = np.zeros((self.n + 1, self.n + 1))
        # 第一列储存逐次分半梯形公式积分值
        n, h = 1, self.b - self.a  # 初始划分区间数和子区间步长
        T_before, T_next = 0, h / 2 * (self.int_fun(self.a) + self.int_fun(self.b))
        self.romberg_table[0, 0] = T_next
        for i in range(1, self.n + 1):
            n, h = 2 * n, h / 2  # 逐次分半
            T_before = T_next  # 积分值的更新
            xi = np.linspace(self.a, self.b, n + 1)
            idx = np.linspace(0, n, n + 1, dtype=np.int64)
            xi_odd = xi[np.mod(idx, 2) == 1]
            yi_odd = self.int_fun(xi_odd)
            T_next = T_before / 2 + h * sum(yi_odd)
            self.romberg_table[i, 0] = T_next

        # 龙贝格外推
        for i in range(self.n):
            pw = mpmath.power(4, i + 1)
            self.romberg_table[:self.n - i, i + 1] = (pw * self.romberg_table[1:self.n - i + 1, i] -
                                                      self.romberg_table[:self.n - i, i]) / (pw - 1)

        self.int_value = self.romberg_table[0, -1]
