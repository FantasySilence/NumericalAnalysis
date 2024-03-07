import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['STSong']
matplotlib.rcParams['axes.unicode_minus'] = False



class AdaptiveIntergralAlgorithm:
    
    """
    自适应积分算分，每个小区间采用辛普森公式
    """

    def __init__(self, int_fun, int_internal, eps=1e-8):

        """
        必要的参数初始化
        int_fun: 被积函数
        int_internal: 积分区间
        eps: 精度
        """

        self.int_fun = int_fun      # 符号定义的被积函数
        if len(int_internal) == 2:
            self.a, self.b = int_internal[0], int_internal[1]   # 积分区间
        else:
            raise ValueError("积分区间参数设置不规范，应为[a,b]的形式")
        self.eps = eps                # 精度
        self.int_value = None         # 积分值结果
        self.x_node = [self.a, self.b]  # 最终划分的节点分布情况

    def cal_int(self):

        """
        自适应积分算法，采用递归求解
        """

        self.int_value = self.__sub_cal_int__(self.a, self.b)
        self.x_node = np.asarray(sorted(self.x_node))
        return self.int_value
    
    def __sub_cal_int__(self, a, b):

        """
        递归计算每个子区间的积分值，并根据精度要求是否再次划分区间
        """

        complete_int_value = self.__simpson_int__(a, b)     # 子区间采用辛普森公式
        mid = (a+b)/2
        left_half = self.__simpson_int__(a, mid)
        right_half = self.__simpson_int__(mid, b)
        # 精度判断
        if abs(complete_int_value - (left_half + right_half)) < 5*self.eps:
            int_value = left_half + right_half
        else:
            self.x_node.append(mid)
            int_value = self.__sub_cal_int__(a, mid) + self.__sub_cal_int__(mid, b)
        return int_value
    
    def __simpson_int__(self, a, b):

        """
        子区间采用辛普森公式
        a: 子区间的左端点
        b: 子区间的右端点
        """

        mid = (a+b)/2
        return (b-a)/6*(self.int_fun(a)+4*self.int_fun(mid)+self.int_fun(b))