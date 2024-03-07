import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['STSong']
matplotlib.rcParams['axes.unicode_minus'] = False

from src.common.utilsapproximation.matrix_utils\
import CubicSplineNaturalInterpolation



class AdaptiveSplineApproximation:

    """
    自适应三次样条逼近：节点序列未必等距划分，每个小区间也并非等长的。
    """

    def __init__(self, fun, x_span, eps=1e-5):

        """
        必要的参数初始化
        """

        self.fun = fun    # 被逼近的函数
        self.a, self.b = x_span[0], x_span[1]
        self.eps = eps    # 每个区间段的逼近精度
        self.node = np.array([self.a, (self.a+self.b)/2, self.b])   # 初始化节点序列
        self.max_error = None    # 最终满足要求精度下的整个区间的最大误差
        self.node_num = 0   # 最终满足要求精度的，最终划分的节点序列个数
        self.spline_obj = None    # 三次样条插值对象


    def fit_approximation(self):

        """
        自适应三次样条逼近：采用自然边界条件
        """

        flag = True     # 整个区间不再进行划分，即不再增加节点序列，如果划分则为True
        self.max_error, n, self.node_num = 0, 10, len(self.node)
        while flag and len(self.node)<=1000:
            flag = False    # 默认不再划分，满足了精度要求
            # 在当前节点序列下，采用分段三次样条插值生成pi(x)
            y_node = self.fun(self.node)    # 节点序列下的函数值
            k_node = np.copy(self.node)  
            self.spline_obj = CubicSplineNaturalInterpolation(k_node, y_node) 
            self.spline_obj.fit_interp()    # 生成三次样条插值逼近函数p(x)   
            insert_num = 0      # 当前区间段前已插入的节点数                             
            for i in range(len(k_node)-1):
                # 查找每个区间段的最大误差以及对应的坐标点
                nodes_merge = []    # 用于合并节点
                mx, me = self.__find_max_error__(k_node[i], k_node[i+1], n)
                if me>self.eps:
                    nodes_merge.extend(self.node[:i+insert_num+1])
                    nodes_merge.extend([mx])    # 插入
                    self.node_num += 1      # 节点数加1
                    nodes_merge.extend(self.node[i+insert_num+1:])
                    insert_num += 1      # 前面已插入的节点数加1
                    self.node = np.copy(nodes_merge)
                    flag = True
                elif me>self.max_error:
                    self.max_error = me


    def __find_max_error__(self, a, b, n):

        """
        求解区间[a,b]上的最大逼近误差相对应的坐标点
        """

        esp0 = 1e-2     # 区间是否再次划分的精度，不宜过小
        max_error, max_x = 0, a     # 记录区间最大误差和所对应的坐标点
        tol, max_error_before = 1, 0
        # tol = np.abs(max_error - max_error_before)
        while tol > esp0:
            if b-a<self.eps:
                break
            t_n = np.linspace(a, b, n)
            f_val = self.fun(t_n)
            p_val = self.spline_obj.cal_interp(t_n)
            error = np.abs(f_val - p_val)
            max_idx = np.argmax(error)   # 最大误差对应的索引
            max_error_before = max_error
            if error[max_idx]>max_error:
                max_x, max_error = t_n[max_idx], error[max_idx]
            tol = np.abs(max_error - max_error_before)
            n *= 2      # 每次等分点是上一次的2倍
        return max_x, max_error


    def cal_x0(self,x0):

        return self.spline_obj.cal_interp(x0)


    def plt_approximation(self, is_show=True):

        """
        绘制逼近多项式图像
        """

        if is_show:
            plt.figure(figsize=(8, 6))
        xi = self.a + np.random.rand(100) * (self.b - self.a)   # 区间[a, b]内的随机数
        xi = np.array(sorted(xi),dtype=np.float64)   # 升序排序
        yi = self.cal_x0(xi)
        y_true = self.fun(xi)
        plt.plot(xi, y_true, 'k*-', lw=1.5, label="true")
        plt.plot(xi, yi, 'r*--', lw=1.5, label="appproximation")
        mse = np.sqrt(np.mean((yi-y_true)**2))
        plt.title("Adaptive Spline Approximation Curve(MSE=%.2e)"%mse,fontdict={"fontsize":14})
        plt.xlabel('X(Ramdomly Divide 100 Points)',fontdict={"fontsize":12})
        plt.ylabel("Exact VS Approximation",fontdict={"fontsize":12})
        plt.legend(loc='best')
        if is_show:
            plt.show()