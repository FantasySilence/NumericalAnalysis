import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['STSong']
matplotlib.rcParams['axes.unicode_minus'] = False

from utilsFiles.filesio import FilesIO

class SolutionToQuesTwo:

    """
    第二问
    """

    def __init__(self, x, y, k=None, w=None, fun_list = None):

        """
        参数初始化\n
        x, y: 离散数据点，离散数据点的长度需一致\n
        k: 进行多项式拟合时，必须指定的多项式的最高阶次\n
        w: 权系数，长度需与离散数据点的长度一致，默认情况下为1\n
        fun_list: 自定义的基函数,本题中为勒让德多项式
        """

        self.x, self.y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        # 如果进行多项式拟合，此变量为多项式最高阶次，需要指定
        self.k = k
        if len(self.x) != len(self.y):
            raise ValueError("离散点数据长度不一致！")
        else:
            self.n = len(self.x)    # 离散点的个数
        if w is None:
            self.w = np.ones(self.n)    # 默认情况下，所有数据权重一致为1
        else:
            if len(w) != self.n:
                raise ValueError("权重长度与离散数据点不一致！")
            else:
                self.w = np.asarray(w, dtype=np.float64)
        self.fit_poly = None                # 曲线拟合的多项式
        self.poly_coefficient = None        # 基函数前的系数组成的向量
        self.polynomial_orders = None       # 系数的阶次
        self.fit_error = None               # 拟合的误差向量
        self.mse = np.infty                 # 拟合的均方根误差
        self.fun_list = list(fun_list)      # 自定义的基函数，本题中为勒让德多项式
        self.m = len(fun_list)              # 自定义基函数的个数
    

    def __lambdified__(self,fun_list):

        """
        将传入的自定义基函数转化为可以进行计算的函数
        """

        fun_list1 = []
        for fun in fun_list:
            if isinstance(fun, float):
                raise ValueError("请避免使用小数！")
            if isinstance(fun, int):
                fun_list1.append(int(fun))
            else:
                t = fun.free_symbols.pop()
                fun = sp.lambdify(t, fun)
                fun_list1.append(fun)
        return fun_list1
    
    def fit_curve(self):

        """
        最小二乘法拟合
        """

        C = np.zeros((self.m,self.m))   # 初始化法方程系数矩阵
        d = np.zeros(self.m)            # 初始化法方程右端向量
        self.fun_list1 = self.__lambdified__(self.fun_list)           # 转化自定义的基函数 
        fun_type = list(map(type, self.fun_list1))               # 提取基函数列表中元素的类型
        function = None                                          # 元素为函数表达式时元素的类型
        for t in fun_type:
            if t == int:
                continue
            if t != int:
                function = t
                break
        # 构造法方程的系数矩阵
        for i in range(self.m):
            for j in range(self.m):
                if isinstance(self.fun_list1[i],function) and isinstance(self.fun_list1[j],function):
                    C[i,j] = np.dot(self.w, self.fun_list1[i](self.x)*self.fun_list1[j](self.x))
                if isinstance(self.fun_list1[i],int) and isinstance(self.fun_list1[j],function):
                    C[i,j] = np.dot(self.w, self.fun_list1[i]*self.x*self.fun_list1[j](self.x))
                if isinstance(self.fun_list1[i],function) and isinstance(self.fun_list1[j],int):
                    C[i,j] = np.dot(self.w, self.fun_list1[i](self.x)*self.fun_list1[j]*self.x)
                if isinstance(self.fun_list1[i],int) and isinstance(self.fun_list1[j],int):
                    C[i,j] = np.dot(self.w, self.fun_list1[i]*self.fun_list1[j]*self.x)
        # 构造法方程的右端向量
        for i in range(self.m):
            if isinstance(self.fun_list1[i],function):
                d[i] = np.dot(self.w, self.fun_list1[i](self.x)*self.y)
            if isinstance(self.fun_list1[i],int):
                d[i] = np.dot(self.w, self.fun_list1[i]*self.x*self.y)
        # 求解法方程
        self.poly_coefficient = np.linalg.solve(C,d)

        t = sp.symbols('t')
        self.fit_poly = self.poly_coefficient[0]*self.fun_list[0]
        for p in range(1,self.m):
            self.fit_poly += self.poly_coefficient[p]*self.fun_list[p]
        
        self.__cal_fit_error__()    # 误差分析
    

    def __cal_fit_error__(self):

        """
        计算拟合的误差和均方根误差
        """

        y_fit = self.__cal_x0__(self.x)
        self.fit_error = self.y - y_fit     # 误差向量
        self.mse = np.sqrt(np.mean(self.fit_error**2))    # 均方根误差
    
    def __cal_x0__(self,x0):

        """
        求解给定数值x0的拟合值
        """

        t = self.fit_poly.free_symbols.pop()
        fit_poly = sp.lambdify(t, self.fit_poly)
        return fit_poly(x0)
    
    def plt_curve_fit(self,is_show=True):

        """
        拟合曲线以及离散数据点的可视化
        """
        
        xi = np.linspace(self.x.min(), self.x.max(), 100)
        yi = self.__cal_x0__(xi)    # 拟合值
        if is_show:
            plt.figure(figsize=(8, 6), facecolor="white", dpi=80)
        plt.plot(xi, yi, 'k-', lw=1.5, label="拟合曲线")
        plt.plot(self.x, self.y, 'ro', lw=1.5, label="离散数据点")
        if self.k:
            plt.title("最小二乘拟合(MSE=%.2e,Order=%d)"%(self.mse,self.k),fontdict={"fontsize":14})
        else:
            plt.title("最小二乘拟合(MSE=%.2e)"%(self.mse),fontdict={"fontsize":14})
        plt.xlabel('X',fontdict={"fontsize":12})
        plt.ylabel("Y",fontdict={"fontsize":12})
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.legend(loc='best', frameon=False)
        if is_show:
            plt.savefig(FilesIO.getSavePath("image_2.png"))
            plt.show()