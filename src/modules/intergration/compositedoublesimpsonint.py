import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['STSong']
matplotlib.rcParams['axes.unicode_minus'] = False



class CompositeDoubleSimpsonIntergration:

    """
    复合辛普森二重积分,自适应划分
    """

    def __init__(self, int_fun, x_span, y_span, eps=1e-6, max_split=100, increment=10):

        """
        必要的参数初始化
        int_fun: 被积函数
        x_span: x的积分区间
        y_span: y的积分区间
        eps: 自适应积分精度，前后两次积分值的差的绝对值最为判断依据
        max_split: 最大划分次数,默认为100
        increment: 每次递增区间数,默认从10开始
        """

        self.int_fun = int_fun      # 被积函数
        self.x_span = np.asarray(x_span, dtype=np.float64)  # x的积分区间
        self.y_span = np.asarray(y_span, dtype=np.float64)  # y的积分区间
        self.eps = eps      # 自适应积分精度，前后两次积分值的差的绝对值最为判断依据
        self.max_split = max_split  # 最大划分次数
        self.increment = increment  # 每次递增区间数为10
        self._intergral_values_ = []    # 存储自适应过程中的每次积分值
        self._n_splits_ = []    # 存储自适应过程中的每次划分的区间数
        self.int_value = None       # 积分值
        self.internal_num = 0        # 划分区间数

    def cal_2d_int(self):

        """
        复合辛普森二重积分计算
        """

        for i in range(self.max_split):
            n = self.increment*(i+1)    # 每次递增10个
            hx, hy = np.diff(self.x_span)/n, np.diff(self.y_span)/n     # 子区间长度
            # x,y等分节点
            xi = np.linspace(self.x_span[0], self.x_span[1], n+1, endpoint=True)
            yi = np.linspace(self.y_span[0], self.y_span[1], n+1, endpoint=True)
            xy = np.meshgrid(xi, yi)
            int1 = np.sum(self.int_fun(xy[0][:-1, :-1], xy[1][:-1, :-1]))
            int2 = np.sum(self.int_fun(xy[0][1:, :-1], xy[1][1:, :-1]))
            int3 = np.sum(self.int_fun(xy[0][:-1, 1:], xy[1][:-1, 1:]))
            int4 = np.sum(self.int_fun(xy[0][1:, 1:], xy[1][1:, 1:]))
            xci = np.divide(xy[0][:-1, :-1]+xy[0][1:, 1:], 2)   # x的各个中点值
            yci = np.divide(xy[1][:-1, :-1]+xy[1][1:, 1:], 2)   # y的各个中点值
            int5 = np.sum(self.int_fun(xci, xy[1][:-1,:-1])) + np.sum(self.int_fun(xy[0][:-1,:-1], yci))+\
                   np.sum(self.int_fun(xci, xy[1][1:, 1:])) + np.sum(self.int_fun(xy[0][1:, 1:], yci))
            int6 = np.sum(self.int_fun(xci, yci))
            int_value = hx*hy/36*(int1 + int2 + int3 + int4 + 4*int5 + 16*int6)
            self._intergral_values_.append(int_value)
            self._n_splits_.append(n)
            if len(self._intergral_values_) > 1 and np.abs(int_value - self._intergral_values_[-2]) < self.eps:
                break
        self.int_value = self._intergral_values_[-1]
        self.internal_num = self._n_splits_[-1]

    def plt_precision(self, is_show=True):

        """
        绘制自适应积分过程精度收敛曲线
        """

        if is_show:
            plt.figure(figsize=(8,6))
        plt.plot(self._n_splits_, self._intergral_values_, "ko-")
        plt.xlabel("划分区间数", fontdict={"fontsize":12})
        plt.ylabel("积分近似值", fontdict={"fontsize":12})
        plt.title("Composite Double Simpson Intergration", fontdict={"fontsize":14})
        if is_show:
            plt.show()