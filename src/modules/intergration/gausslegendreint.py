import numpy as np
import sympy as sp
import matplotlib
import math 

matplotlib.rcParams['font.sans-serif'] = ['STSong']
matplotlib.rcParams['axes.unicode_minus'] = False



class GuassLegendreIntergration:

    """
    高斯——勒让德求积公式
    """

    def __init__(self, int_fun, int_internal, zeros_num=10):
        
        """
        必要的参数初始化
        int_fun: 被积函数
        int_internal: 积分区间
        zeros_num: 正交多项式零点个数
        """
        
        self.int_fun = int_fun      # 被积函数
        if len(int_internal) == 2:
            self.a, self.b = int_internal[0], int_internal[1]   # 积分区间
        else:
            raise ValueError("积分区间参数设置不规范，应为[a,b]！")
        self.n = int(zeros_num)       # 正交多项式的零点数
        self.zero_points = None        # 勒让德高斯零点
        self.int_value = None            # 积分值结果
        self.A_k = None        # 求积系数
    
    def cal_int(self):

        """
        高斯——勒让德求积公式：核心部分求解零点与系数
        """

        self.__cal_Ak_coef__()      # 求解系数,同时获得了零点
        f_val = self.int_fun(self.zero_points)    # 零点函数值
        self.int_value = np.dot(self.A_k, f_val)    # 插值型求积公式
        return self.int_value

    def __cal_guass_zeros_points__(self):

        """
        高斯零点计算
        """

        t = sp.Symbol('t')
        # 勒让德多项式构造
        p_n = (t**2-1)**self.n/math.factorial(self.n)/2**self.n
        diff_p_n = sp.diff(p_n, t, self.n)  # 多项式的n阶导数
        # 求解多项式的全部零点，需要更加优秀的算法替代
        self.zero_points = np.asarray(sp.solve(diff_p_n, t), dtype=np.float64)
        return diff_p_n, t
    
    def __cal_Ak_coef__(self):

        """
        计算Ak系数
        """

        diff_p_n, t = self.__cal_guass_zeros_points__()     # 计算高斯零点
        Ak_poly = sp.lambdify(t, 2/(1-t**2)/(diff_p_n.diff(t, 1)**2))
        self.A_k = Ak_poly(self.zero_points)    # 求解Ak系数
        # 区间转换, [a,b]——>[-1,1]
        self.A_k = self.A_k*(self.b - self.a)/2
        self.zero_points = self.zero_points*(self.b - self.a)/2 + (self.a + self.b)/2