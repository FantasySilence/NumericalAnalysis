import numpy as np
from fractions import Fraction

from src.common.utilslinearEqsystem.checker import check_matrix


class GuassElimation:

    """
    高斯消去法求解线性方程组
    """

    @check_matrix
    def __init__(self, matrix:np.ndarray, solve_type:str="order", is_print:bool=True):

        """
        参数初始化\n
        matrix: 线性方程组的增广矩阵\n
        solve_type: 求解线性方程组的方法(order: 高斯顺序消去法,PCA: 列主元消去法)\n
        is_print: 是否输出结果，默认为True\n
        如果想获取消去后的矩阵以及方程组的解以便调用,\n
        请查询类属性self.solve_matrix, self.solution\n
        如果想要查询交换次序，请查询类属性self.In或self.In_dict
        """

        self.matrix = matrix
        self.solve_type = solve_type
        self.is_print = is_print
        self.A = matrix[:,:-1]      # 线性方程组的系数矩阵
        self.b = matrix[:,-1]       # 线性方程组的右端向量
        self.solve_matrix = np.zeros(matrix.shape, dtype=np.float64)   # 存储消去后的矩阵
        self.solution = np.zeros(matrix.shape[1]-1, dtype=np.float64)   # 存储方程的解
        self.In = None      # 列主元消去法的行交换次序
        self.In_dict = None      # 列主元消去法的行交换次序的字典形式
        self.__solve__()
        if self.is_print:
            self.__print__()
            
        
    def __solve__(self):

        """
        求解线性方程组
        """
        
        if self.solve_type.lower() == "order":
            self.__order_solve__()
        elif self.solve_type.lower() == "pca":
            self.In = []    # 存储初等矩阵，用于记录行交换次序
            self.__PCA_solve__()
            self.In_dict = {"第"+str(i+1)+"次行交换对应的初等矩阵":self.In[i] for i in range(len(self.In))}
        else:
            raise ValueError("未知的求解方法,仅支持高斯顺序消元法(order)或列主元消去法(PCA)")
    

    def __print__(self):

        """
        打印输出结果
        """

        np.set_printoptions(formatter={'all':lambda x:str(Fraction(x).limit_denominator())})
        print("消去后的矩阵：\n", self.solve_matrix)
        print("方程组的解：")
        print({"x"+str(i+1):round(self.solution[i], 3) for i in range(len(self.solution))})
        if self.solve_type.lower() == "pca":
            print("初等矩阵的行交换次序：")
            for key, value in self.In_dict.items():
                print(key, ":\n", value)
        np.set_printoptions()


    def __order_solve__(self):

        """
        高斯顺序消去法解线性方程组
        """

        # 进行高斯消去法的第一步,从而初始化solve_matrix矩阵,便于消去法的进行
        self.solve_matrix[0,:] = self.matrix[0,:]  
        for i in range(1, self.matrix.shape[0]):
            self.solve_matrix[i,:] = self.matrix[i,:] - self.matrix[0,:]*(self.matrix[i,0]/self.matrix[0,0])
        # 循环求解消元后的上三角矩阵
        # 遍历列
        for j in range(2, self.matrix.shape[1]-1):
            # 遍历行,进行消去
            for i in range(j, self.matrix.shape[0]):
                self.solve_matrix[i,:] = self.solve_matrix[i,:] -\
                                self.solve_matrix[j-1,:]*(self.solve_matrix[i,j-1]/self.solve_matrix[j-1,j-1])
        # 根据消元后的上三角矩阵得出方程组的解
        self.solution[0] = self.solve_matrix[-1,-1]/self.solve_matrix[-1,-2]
        for i in range(1, self.matrix.shape[1]-1):
            self.solution[i] = (self.solve_matrix[-1-i,-1] - \
                                np.sum(self.solve_matrix[-1-i,-1-i:-1]*self.solution[:i][::-1]))\
                                    /self.solve_matrix[-1-i,-2-i]
        self.solution = self.solution[::-1]


    def __PCA_solve__(self):

        """
        列主元消去法求解线性方程组
        """

        # 列主元消去法的第一步，从而初始化solve_matrix矩阵,便于消去法的进行
        # 寻找第一列的最大值的索引,并进行交换
        idx = np.argmax(np.abs(self.matrix[0:,0]), axis=0)
        self.matrix[[0, idx],:] = self.matrix[[idx, 0],:]
        # 根据换行操作生成相应的初等矩阵，并储存于In中
        I_1 = np.identity(self.matrix.shape[0], dtype=np.float64)
        I_1[[0, idx],:] = I_1[[idx, 0],:]
        self.In.append(I_1)
        # 初始化solve_matrix矩阵
        self.solve_matrix[0,:] = self.matrix[0,:]  
        for i in range(1, self.matrix.shape[0]):
            self.solve_matrix[i,:] = self.matrix[i,:] - self.matrix[0,:]*(self.matrix[i,0]/self.matrix[0,0])
        # 遍历列
        for j in range(2, self.matrix.shape[1]-1):
            # 寻找最大值的索引,并进行交换
            idx = np.argmax(np.abs(self.solve_matrix[j-1:,j]), axis=0)
            self.solve_matrix[[j-1, idx+j-1],:] = self.solve_matrix[[idx+j-1, j-1],:]
            # 根据换行操作生成相应的初等矩阵，并储存于In中
            I_j = np.identity(self.matrix.shape[0], dtype=np.float64)
            I_j[[j-1, idx+j-1],:] = I_j[[idx+j-1, j-1],:]
            self.In.append(I_j)
            # 遍历行,进行消去
            for i in range(j, self.matrix.shape[0]):
                self.solve_matrix[i,:] = self.solve_matrix[i,:] -\
                             self.solve_matrix[j-1,:]*(self.solve_matrix[i,j-1]/self.solve_matrix[j-1,j-1])
        # 根据消元后的上三角矩阵得出方程组的解
        self.solution[0] = self.solve_matrix[-1,-1]/self.solve_matrix[-1,-2]
        for i in range(1, self.matrix.shape[1]-1):
            self.solution[i] = (self.solve_matrix[-1-i,-1] - \
                                np.sum(self.solve_matrix[-1-i,-1-i:-1]*self.solution[:i][::-1]))\
                                    /self.solve_matrix[-1-i,-2-i]
        self.solution = self.solution[::-1]