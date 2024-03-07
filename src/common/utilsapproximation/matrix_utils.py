import numpy as np

"""
powered by:@御河DE天街
自适应样条逼近时会使用本工具包
"""

class CubicSplineNaturalInterpolation:

    """
    三次样条插值
    complete：第一种边界条件，已知两端的一阶导数值
    second：第二种边界条件，已知两端的二阶导数值
    natural：第二种边界条件，自然边界条件
    periodic：第三种边界条件，周期边界条件
    """
    
    def __init__(self, x, y):

        """
        三次样条插值必要参数的初始化
        """
        
        self.x=np.asarray(x,dtype=np.float64)
        self.y=np.asarray(y,dtype=np.float64)
        if len(self.x)>1 and len(self.x)==len(self.y):
            self.n=len(self.x)      #已知离散数据点的个数
        else:
            raise ValueError("x,y坐标长度不匹配")
        self.y0=None   #所求插值点的值，单个值或者向量
        self.m = None   
    

    def fit_interp(self):

        """
        构造三次样条插值多项式
        """

        self.__natural_spline__(self.x,self.y)
        

    def __natural_spline__(self,x,y):

        """
        求解自然边界条件
        """    

        A=np.diag(2*np.ones(self.n))
        A[0,1],A[-1,-2]=1,1     #边界特殊情况
        c=np.zeros(self.n)
        for i in range(1,self.n-1):
            u=(x[i]-x[i-1])/(x[i+1]-x[i-1])
            lambda_=(x[i+1]-x[i])/(x[i+1]-x[i-1])
            c[i]=3*lambda_*(y[i]-y[i-1])/(x[i]-x[i-1])+\
                 3*u*(y[i+1]-y[i])/(x[i+1]-x[i])
            A[i,i+1],A[i,i-1]=u,lambda_
        c[0]=3*(y[1]-y[0])/(x[1]-x[0])
        c[-1]=3*(y[-1]-y[-2])/(x[-1]-x[-2])    #边界条件
        # self.m=np.linalg.solve(A,c)   # 普通方法
        diag_b, diag_a, diag_c = np.diag(A), np.diag(A, -1), np.diag(A, 1)      # 追赶法求解
        cmtm = ChasingMethodTridiagonalMatrix(diag_a, diag_b, diag_c,c)
        self.m = cmtm.fit_solve()


    def cal_interp(self,x0):

        """
        计算给定插值点的数值
        x0:所求插值的x坐标值
        """

        x0=np.asarray(x0,dtype=np.float64)
        n0=len(x0)      #所求插值点的个数
        self.y0=np.zeros(n0)    #储存插值点y0所对应的插值
        for i in range(n0):
            idx=0   #子区间索引值初始化
            for j in range(len(self.x)-1):
                #查找x0所在的子区间，获取子区间的索引值idx
                if self.x[j]<=x0[i]<=self.x[j+1] or self.x[j+1]<=x0[i]<=self.x[j]:
                    idx=j
                    break
            hi=self.x[idx+1]-self.x[idx]      #子区间长度
            t = x0[i]
            self.y0[i]=self.y[idx]/hi**3*(2*(t-self.x[idx])+hi)*(self.x[idx+1]-t)**2+\
                       self.y[idx+1]/hi**3*(2*(self.x[idx+1]-t)+hi)*(t-self.x[idx])**2+\
                       self.m[idx]/hi**2*(t-self.x[idx])*(self.x[idx+1]-t)**2-\
                       self.m[idx+1]/hi**2*(self.x[idx+1]-t)*(t-self.x[idx])**2
        return self.y0



class ChasingMethodTridiagonalMatrix:

    """
    追赶法求解三对角矩阵
    """

    def __init__(self,diag_a,diag_b,diag_c,d_vector,sol_method="guass"):
        """
        必要的参数初始化
        """
        self.a = np.asarray(diag_a,dtype=np.float64)
        self.b = np.asarray(diag_b,dtype=np.float64)
        self.c = np.asarray(diag_c,dtype=np.float64)
        self.n = len(self.b)
        if len(self.a) != self.n-1 or len(self.c) != self.n-1:
            raise ValueError("系数矩阵对角线元素维度不匹配！")
        self.d_vector = np.asarray(d_vector,dtype=np.float64)
        if len(d_vector) != self.n:
            raise ValueError("右端向量维度与系数矩阵维度不匹配！")
        self.sol_method = sol_method
        self.x, self.y = None, None
        self.eps = None
    

    def fit_solve(self):

        """
        追赶法求解三对角矩阵
        """

        self.y, self.x = np.zeros(self.n), np.zeros(self.n)
        if self.sol_method == "guass":
            self.x = self.__guass_solve__()
        else:
            raise ValueError("未知求解方法！")
        return self.x
    

    def __guass_solve__(self):

        """
        高斯法求解三对角矩阵
        """

        b, d = np.copy(self.b), np.copy(self.d_vector)
        for k in range(self.n-1):
            multiplier = -self.a[k]/b[k]
            b[k+1] += multiplier*self.c[k]
            d[k+1] += multiplier*d[k]
        self.x[-1] = d[-1]/b[-1]
        for i in range(self.n-2,-1,-1):
            self.x[i] = (d[i]-self.c[i]*self.x[i+1])/b[i]
        return self.x