import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

from src.common.utilsinterpolation.Interp_utils import cal_interp
from src.common.utilsinterpolation.Interp_utils import plt_interp



class LagrangeInterpolation:

    """
    拉格朗日插值
    """

    def __init__(self, x, y):

        """
        拉格朗日插值必要的参数初始化
        x:已知数据的x坐标
        y:已知数据的y坐标
        """

        self.x = np.asarray(x,dtype=np.float64)
        self.y = np.asarray(y,dtype=np.float64)
        if len(self.x)>1 and len(self.x)==len(self.y):
            self.n=len(self.x)
        else:
            raise ValueError("x,y坐标长度不匹配")
        self.polynomial=None   #最终的插值多项式的符号表示
        self.poly_coefficient=None   #最终的插值多项式的系数，幂次从高到低
        self.coefficient_order=None   #对应多项式系数的阶次
        self.y0=None   #所求插值点的值，单个值或者向量
    

    def fit_interp(self):

        """
        生成拉格朗日插值多项式
        """

        t=sp.symbols('t')       #定义符号变量
        self.polynomial=0.0
        for i in range(self.n):
            # 针对每个数值点构造插值基函数
            basis_fun=self.y[i]  #插值基函数
            for j in range(i):
                basis_fun*=(t-self.x[j])/(self.x[i]-self.x[j])
            for j in range(i+1,self.n):
                basis_fun*=(t-self.x[j])/(self.x[i]-self.x[j])
            self.polynomial+=basis_fun
        
        # 插值多项式特征
        self.polynomial=sp.expand(self.polynomial)  #多项式展开
        polynomial=sp.Poly(self.polynomial,t)       #构造多项式对象
        self.poly_coefficient=polynomial.coeffs()   #获取多项式的系数
        self.coefficient_order=polynomial.monoms()  #多项式系数对应的阶次
    

    def cal_interp(self,x0):

        """
        计算给定插值点的数值
        x0:所求插值的x坐标值
        """

        self.y0 = cal_interp(self.polynomial,x0)
        return self.y0
    

    def plt_interp(self,x0=None,y0=None):

        """
        可视化插值图像和插值点
        """
        
        params=(self.polynomial,self.x,self.y,'Lagrange',x0,y0)
        plt_interp(params)