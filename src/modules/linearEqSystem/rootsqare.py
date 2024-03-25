import numpy as np
from fractions import Fraction

from src.common.utilslinearEqsystem.checker import IsMatrixReasonable


class RootSquareDecomposition:

    """
    平方根分解法求解线性方程组
    要求系数矩阵为对称(正定)矩阵
    """

    @IsMatrixReasonable("RootSquare")
    def __init__(self, matrix: np.ndarray, decom_type: str = "ichol", is_print: bool = True):

        """
        参数初始化\n
        matrix: 线性方程组的增广矩阵\n
        decom_type: 求解方式, 默认为改进的平方根法(chol:平方根法,ichol:改进的平方根法)\n
        is_print: 是否输出结果，默认为True\n
        如果想获取L矩阵，D矩阵以及方程组的解以便调用,\n
        请查询类属性self.L, self.D, self.solution
        """

        self.A = matrix[:, :-1]  # 线性方程组的系数矩阵
        self.b = matrix[:, -1]  # 线性方程组的右端向量
        self.L = np.identity(self.A.shape[0], dtype=np.float64)  # 平方根分解法，L矩阵，下三角矩阵
        self.D = np.zeros(self.A.shape, dtype=np.float64)  # 改进的平方根分解法，D矩阵，对角矩阵
        self.y = np.zeros(self.A.shape[0], dtype=np.float64)  # Ly=b中的中间变量
        # 平方根分解法L.T*x=y中的中间变量(改进的平方根法DL.T*x=y中的中间变量)
        self.x = np.zeros(self.A.shape[0], dtype=np.float64)
        self.solution = None  # 方程的解
        self.decom_type = decom_type
        self.is_print = is_print
        self.__solve__()
        if self.is_print:
            self.__print__()

    def __solve__(self):

        """
        求解线性方程组
        """

        if self.decom_type.lower() == "chol":
            eigval, _ = np.linalg.eig(self.A)
            if np.any(eigval < 0):
                raise ValueError('线性方程组的系数矩阵不是正定矩阵')
            self.__chol_decompose__()
        elif self.decom_type.lower() == "ichol":
            self.__ichol_decompose__()
        else:
            raise ValueError("未知的分解方法,仅支持平方根法(chol)或改进的平方根法(ichol)")

    def __print__(self):

        """
        打印输出结果
        """

        np.set_printoptions(formatter={'all': lambda x: str(Fraction(x).limit_denominator())})
        print("分解后的L矩阵：\n", self.L)
        if self.decom_type.lower() == "ichol":
            print("分解后的D矩阵：\n", self.D)
        print("方程组的解：")
        print({"x" + str(i + 1): round(self.solution[i], 3) for i in range(len(self.solution))})
        np.set_printoptions()

    def __chol_decompose__(self):

        """
        平方根法求解线性方程组
        要求系数矩阵为对称正定矩阵
        """

        # 计算L矩阵的第一列
        self.L[0, 0] = np.sqrt(self.A[0, 0])
        self.L[1:, 0] = self.A[1:, 0] / self.L[0, 0]
        # 循环计算L矩阵的其余列
        for i in range(1, self.A.shape[0]):
            self.L[i, i] = np.sqrt(self.A[i, i] - np.sum(self.L[i, :i] ** 2))
            # 由于循环过程中L矩阵的最后一列无需求解，故判断并提前退出循环 
            if i == self.A.shape[0] - 1:
                break
            self.L[i + 1:, i] = (self.A[i + 1:, i] - np.sum(self.L[i + 1:, :i] * self.L[i, :i])) / self.L[i, i]
        # 循环计算L矩阵的其余列
        for i in range(1, self.A.shape[0]):
            self.L[i, i] = np.sqrt(self.A[i, i] - np.sum(self.L[i, :i] ** 2))
            self.L[i + 1:, i] = (self.A[i + 1:, i] - np.sum(self.L[i + 1:, :i] * self.L[i, :i])) / self.L[i, i]
        # 求解Ly=b
        self.y[0] = self.b[0]
        for i in range(1, self.A.shape[0]):
            self.y[i] = self.b[i] - np.sum(self.L[i, :i] * self.y[:i])
        # 求解L.T*x=y
        L_T_x = np.c_[self.L.T, self.y]
        self.x[0] = L_T_x[-1, -1] / L_T_x[-1, -2]
        for i in range(1, L_T_x.shape[1] - 1):
            self.x[i] = (L_T_x[-1 - i, -1] - np.sum(L_T_x[-1 - i, -1 - i:-1] * self.x[:i][::-1])) / L_T_x[
                -1 - i, -2 - i]
        self.solution = self.x[::-1]

    def __ichol_decompose__(self):

        """
        改进的平方根法求解线性方程组
        要求系数矩阵为对称矩阵
        """

        # 计算D矩阵的第一个元素以及L矩阵的第一列
        self.D[0, 0] = self.A[0, 0]
        self.L[0:, 0] = self.A[:, 0] / self.D[0, 0]
        # 循环计算D矩阵的其余元素以及L矩阵的其余列
        for i in range(1, self.A.shape[0]):
            self.D[i, i] = self.A[i, i] - sum(self.L[i, j] ** 2 * self.D[j, j] for j in range(i))
            # 由于循环过程中L矩阵的最后一列无需求解，故判断并提前退出循环 
            if i == self.A.shape[0] - 1:
                break
            self.L[i:, i] = (self.A[i:, i] - sum(self.L[i:, j] * self.L[i, j] * self.D[j, j] for j in range(i))) / \
                            self.D[i, i]
        # 求解Ly=b
        self.y[0] = self.b[0]
        for i in range(1, self.A.shape[0]):
            self.y[i] = self.b[i] - np.sum(self.L[i, :i] * self.y[:i])
        DL_T = self.D.dot(self.L.T)
        # 求解DL.T*x=y
        DL_T_x = np.c_[DL_T, self.y]
        self.x[0] = DL_T_x[-1, -1] / DL_T_x[-1, -2]
        for i in range(1, DL_T_x.shape[1] - 1):
            self.x[i] = (DL_T_x[-1 - i, -1] - np.sum(DL_T_x[-1 - i, -1 - i:-1] * self.x[:i][::-1])) / DL_T_x[
                -1 - i, -2 - i]
        self.solution = self.x[::-1]
