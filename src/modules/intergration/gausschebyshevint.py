import numpy as np
import matplotlib

matplotlib.rcParams["font.sans-serif"] = ["STSong"]
matplotlib.rcParams["axes.unicode_minus"] = False



class GuassChebyshevIntergration:

    """
    高斯——切比雪夫求积公式
    """

    def __init__(self, int_fun, zeros_num=10, cheb_type=1):
        """
        必要的参数初始化
        int_fun: 被积函数
        int_internal: 积分区间
        zeros_num: 正交多项式零点个数
        """

        self.int_fun = int_fun  # 被积函数
        self.n = int(zeros_num)  # 正交多项式的零点数
        if cheb_type in [1, 2]:
            self.cheb_type = cheb_type  # 切比雪夫类型，第一类或第二类切比雪夫多项式
        else:
            raise ValueError(
                "切比雪夫类型设置不规范，应为1或2,对应第一类或第二类切比雪夫多项式！"
            )
        self.zero_points = None  # 勒让德高斯零点
        self.int_value = None  # 积分值结果
        self.A_k = None  # 求积系数


    def cal_int(self):
        """
        高斯——切比雪夫求积公式
        """

        if self.cheb_type == 1:
            k_i = np.linspace(0, self.n, self.n + 1, dtype=np.int64)
            self.zero_points = np.cos((2 * k_i + 1) * np.pi / (2 * self.n + 2))
            self.A_k = np.pi / (self.n + 1)
            f_val = self.int_fun(self.zero_points) * np.sqrt(1 - self.zero_points**2)
            self.int_value = self.A_k * np.sum(f_val)
        elif self.cheb_type == 2:
            k_i = np.linspace(1, self.n, self.n, dtype=np.int64)
            self.zero_points = np.cos(k_i * np.pi / (self.n + 1))
            self.A_k = np.pi / (self.n + 1) * np.sin(k_i * np.pi / (self.n + 1)) ** 2
            f_val = self.int_fun(self.zero_points) / np.sqrt(1 - self.zero_points**2)
            self.int_value = np.dot(self.A_k, f_val)
        return self.int_value
