import numpy as np
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['STSong']
matplotlib.rcParams['axes.unicode_minus'] = False



class CubicBsplineInterpolationIntergration:

    """
    离散数据积分：B样条插值法
    """

    def __init__(self, x, y):

        """
        必要的参数初始化
        x, y: 离散数据点，离散数据点的长度需一致
        """

        self.x, self.y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        if len(self.x) >= 3:
            if len(self.x) != len(self.y):
                raise ValueError("离散数据点维度不匹配！")
            else:
                self.n = len(self.x)
                self.h = self.__check_equidistant__()
        else:
            raise ValueError("离散数据点长度需大于等于3！")
        self.int_value = None
    
    def __check_equidistant__(self):

        """
        检查数据点是否是等距的
        """

        xx = np.linspace(min(self.x), max(self.x), len(self.x), endpoint=True)
        if (self.x == xx).all() or (self.x == xx[::-1]).all():
            return self.x[1] - self.x[0]
        else:
            raise ValueError("离散数据点不是等距的！")
        
    def cal_int(self):

        """
        三次B样条插值计算离散数据积分
        """

        c_spline = self.__natural_bspline__()
        self.int_value = self.h/24*(c_spline[0]+c_spline[-1]) +\
                         self.h/2*(c_spline[1] + c_spline[-2]) +\
                         23*self.h/24*(c_spline[2] + c_spline[-3]) +\
                         self.h*sum(c_spline[3:-3])
        return self.int_value

    def __natural_bspline__(self):

        """
        计算B样条函数系数
        """

        A = 4*np.eye(self.n - 2)
        I = np.eye(self.n - 2)
        AL = np.r_[I[1:, :], np.zeros((1, self.n-2))]
        AR = np.r_[np.zeros((1, self.n-2)), I[:-1, :]]
        A = A + AL + AR
        # 构造右端向量
        b = np.zeros(self.n - 2)
        b[1:-1] = 6*self.y[2:-2]
        b[0] = 6*self.y[1] - self.y[0]
        b[-1] = 6*self.y[-2] - self.y[-1]
        # 求解控制节点，共n+2个
        c = np.zeros(self.n+2)
        d = np.linalg.solve(A, b)
        c[2:-2] = d
        c[1] = self.y[0]
        c[0] = 2*c[1] - c[2]
        c[-2] = self.y[-1]
        c[-1] = 2*c[-2] - c[-3]
        return c