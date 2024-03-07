import numpy as np
from fractions import Fraction



class TriangleDecomposition:

    """
    直接三角分解法求解线性方程组
    """

    def __init__(self, matrix:np.ndarray, decom_type:str="LU", is_print:bool=True):

        """
        参数初始化\n
        matrix: 线性方程组的增广矩阵\n
        decom_type: 分解方式, 默认为LU分解(LU:直接三角分解法,PLU:列主元的三角分解法)\n
        is_print: 是否输出结果，默认为True\n
        如果想获取L矩阵，U矩阵以及方程组的解以便调用,\n
        请查询类属性self.L, self.U, self.solution\n
        如果想要查询交换次序，请查询类属性self.In或self.In_dict
        """

        check_matrix = matrix.copy()
        if check_matrix[:,:-1].shape[0] != check_matrix[:,:-1].shape[1]:
            raise ValueError('线性方程组的系数矩阵不是方阵')
        if np.linalg.matrix_rank(check_matrix[:,:-1]) != np.linalg.matrix_rank(matrix):
            raise ValueError('该线性方程组无解')
        if np.linalg.matrix_rank(check_matrix[:,:-1]) == np.linalg.matrix_rank(matrix) \
           and np.linalg.matrix_rank(check_matrix[:,:-1]) < check_matrix[:,:-1].shape[0]:
            raise ValueError('该线性方程组有多个解,暂时支持有唯一解的方程组')
        self.A = matrix[:,:-1]      # 线性方程组的系数矩阵
        self.b = matrix[:,-1]       # 线性方程组的右端向量
        self.L = np.identity(self.A.shape[0], dtype=np.float64)     # LU分解的L,为单位下三角矩阵
        self.U = np.zeros(self.A.shape, dtype=np.float64)           # LU分解的U,为上三角矩阵
        self.y = np.zeros(self.A.shape[0], dtype=np.float64)        # Ly=b中的中间变量
        self.x = np.zeros(self.A.shape[0], dtype=np.float64)        # Ux=y中的中间变量
        self.solution = None     # 方程的解
        self.In = None      # 列主元的三角分解法的行交换次序
        self.decom_type = decom_type 
        self.is_print = is_print
        self.__solve__()
        if self.is_print:
            self.__print__()


    def __solve__(self):

        """
        求解线性方程组
        """

        if self.decom_type.lower() == "lu":
            self.__LU_decompose__()
        elif self.decom_type.lower() == "plu":
            self.In = []    # 存储初等矩阵，用于记录行交换次序
            self.__PLU_decompose__()
            self.In_dict = {"第"+str(i+1)+"次行交换对应的初等矩阵":self.In[i] for i in range(len(self.In))}
        else:
            raise ValueError("未知的分解方法,仅支持LU分解(LU)或PLU分解(PLU)")
    

    def __print__(self):

        """
        打印输出结果
        """

        np.set_printoptions(formatter={'all':lambda x:str(Fraction(x).limit_denominator())})
        print("分解后的L矩阵：\n", self.L)
        print("分解后的U矩阵：\n", self.U)
        print("方程组的解：")
        print({"x"+str(i+1):round(self.solution[i], 3) for i in range(len(self.solution))})
        if self.decom_type.lower() == "plu":
            print("初等矩阵的行交换次序：")
            for key, value in self.In_dict.items():
                print(key, ":\n", value)
        np.set_printoptions()
    

    def __LU_decompose__(self):

        """
        直接三角分解法求解线性方程组
        利用三角分解的公式进行L矩阵和U矩阵的获得
        """

        # 计算U矩阵的第一行和L矩阵的第一列
        self.U[0,0:] = self.A[0,:]
        self.L[1:,0] = self.A[1:,0]/self.U[0,0]
        # 循环求解U矩阵和L矩阵
        for i in range(1, self.A.shape[0]):
            # 循环求解U矩阵
            self.U[i,i:] = self.A[i,i:]-sum(self.L[i,j]*self.U[j,i:] for j in range(i)) 
            # 由于循环过程中L矩阵的最后一列无需求解，故判断并提前退出循环                                                 
            if i == self.A.shape[0]-1:break 
            # 循环求解L矩阵    
            self.L[i+1:,i] = (self.A[i+1:,i]-sum(self.L[i+1:,j]*self.U[j,i] for j in range(i)))/self.U[i,i]
        # 求解Ly=b
        self.y[0] = self.b[0]
        for i in range(1, self.A.shape[0]):
            self.y[i] = self.b[i]-np.sum(self.L[i,:i]*self.y[:i])
        # 求解Ux=y
        U_y = np.c_[self.U, self.y]
        self.x[0] = U_y[-1,-1]/U_y[-1,-2]
        for i in range(1, U_y.shape[1]-1):
            self.x[i] = (U_y[-1-i,-1] - np.sum(U_y[-1-i,-1-i:-1]*self.x[:i][::-1]))/U_y[-1-i,-2-i]
        self.solution = self.x[::-1]


    def __PLU_decompose__(self):

        """
        列主元三角分解法求解线性方程组
        利用列主元消去法，在过程中得到L矩阵和U矩阵
        """

        # 初始化系数矩阵，将第一列绝对值最大元素所在行与第一行交换
        A = self.A.copy()   # 复制一份系数矩阵
        b = self.b.copy()   # 复制一份右端向量
        # 寻找第一列的最大值的索引,并进行交换
        idx = np.argmax(np.abs(A[0:,0]), axis=0)
        A[[0,idx],:] = A[[idx,0],:]
        # 记录交换次序
        I_1 = np.identity(self.A.shape[0], dtype=np.float64)
        I_1[[0,idx],:] = I_1[[idx,0],:]
        b = I_1.dot(b)    # 根据交换次序对方程组右端向量进行行交换
        self.In.append(I_1)
        # 初始化U矩阵以及L矩阵的第一列，U矩阵通过消去获得，L矩阵通过消去过程的中间量获得
        self.U[0,:] = A[0,:]
        for i in range(1, self.A.shape[0]):
            self.L[i,0] = A[i,0]/A[0,0]
            self.U[i,:] = A[i,:] - A[0,:]*(A[i,0]/A[0,0])
        # 循环进行列主元消去法，并将过程中产生的元素存入对应的矩阵
        for j in range(2, self.A.shape[0]):
            # 寻找最大值的索引,并进行交换
            idx = np.argmax(np.abs(self.U[j-1:,j-1]), axis=0) 
            self.U[[j-1,idx+j-1],:] = self.U[[idx+j-1,j-1],:]
            # L矩阵的前一列也要同上进行交换
            self.L[[idx+j-1,j-1],j-2] = self.L[[j-1,idx+j-1],j-2]
            # 生成单位矩阵，进行与之前相同的变换得到用于记录变换的初等矩阵
            I_j = np.identity(self.A.shape[0], dtype=np.float64)
            I_j[[j-1,idx+j-1],:] = I_j[[idx+j-1,j-1],:]
            # 对方程组右端向量进行相同的行交换，并记录
            b = I_j.dot(b)
            self.In.append(I_j)
            # 遍历行,进行消去
            for i in range(j, self.A.shape[0]):
                # 消去时，其余行与"首行"的乘数m存入L矩阵
                self.L[i,j-1] = self.U[i,j-1]/self.U[j-1,j-1]
                # 直接对系数矩阵进行消去得到U矩阵
                self.U[i,:] = self.U[i,:] - self.U[j-1,:]*(self.U[i,j-1]/self.U[j-1,j-1])
        # 求解Ly=Pb
        self.y[0] = b[0]
        for i in range(1, self.A.shape[0]):
            self.y[i] = b[i]-np.sum(self.L[i,:i]*self.y[:i])
        # 求解Ux=y
        U_y = np.c_[self.U, self.y]
        self.x[0] = U_y[-1,-1]/U_y[-1,-2]
        for i in range(1, U_y.shape[1]-1):
            self.x[i] = (U_y[-1-i,-1] - np.sum(U_y[-1-i,-1-i:-1]*self.x[:i][::-1]))/U_y[-1-i,-2-i]
        self.solution = self.x[::-1]