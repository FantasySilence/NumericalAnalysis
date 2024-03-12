import numpy as np
from fractions import Fraction

from src.common.utilslinearEqsystem.checker import isMatrixReasonable


class JGSIterationMethod:

    """
    雅可比迭代法，高斯赛德尔迭代法以及SOR超松弛迭代法
    """

    @isMatrixReasonable("Iteration")
    def __init__(self, matrix: np.ndarray, x0=None, Iter_type: str = "jacobi", w: float = 1.0,
                 max_iter: int = 1000, epsilon: float = 1e-15, is_print: bool = True):

        """
        参数初始化\n
        matrix: 方程组的增广矩阵\n
        x0: 初始向量\n
        w: 松弛因子,默认为1，取值范围(0,2)\n
        epsilon: 解的精度\n
        is_print: 是否输出结果，默认为True\n
        Iter_type: 迭代方法(jacobi: 雅可比迭代法, gauss_seidel: 高斯赛德尔迭代法, SOR: 超松弛迭代法)\n
        如果想获取迭代后的方程组的解以便调用,请查询类属性self.solution
        """

        self.epsilon = epsilon
        self.Iter_type = Iter_type
        self.max_iter = max_iter
        self.is_print = is_print
        if w < 0 or w > 2:
            raise ValueError('松弛因子取值范围为(0,2)')
        else:
            self.w = w
        if x0 is not None:
            if x0.shape[0] != matrix.shape[1] - 1:
                raise ValueError('初始向量的维度与方程组的维度不匹配')
            else:
                self.x0 = np.asarray(x0, dtype=np.float64)
        else:
            self.x0 = np.zeros(matrix.shape[1] - 1, dtype=np.float64)
        self.A = matrix[:, :-1]  # 线性方程组的系数矩阵
        self.b = matrix[:, -1]  # 线性方程组的右端向量
        self.D = np.diag(np.diag(self.A))  # D矩阵
        self.L = np.diag(np.diag(self.A)) - np.tril(self.A)  # L矩阵
        self.U = np.diag(np.diag(self.A)) - np.triu(self.A)  # U矩阵
        self.J = None  # 雅可比迭代法的迭代矩阵
        self.G = None  # 高斯赛德尔迭代法的迭代矩阵
        self.L_w = None  # 超松弛迭代法的迭代矩阵
        self.f = None  # 迭代矩阵
        self.solution_list = np.zeros(self.max_iter, dtype=object)  # 存储迭代过程中方程的解
        self.solution = None  # 方程组的解
        self.__solve__()
        if self.is_print:
            self.__print__()

    def __solve__(self):

        """
        求解方程组
        """

        if self.Iter_type.lower() == "jacobi":
            self.__jacobi_iteration__()
        elif self.Iter_type.lower() == "gauss_seidel":
            self.__gauss_seidel_iteration__()
        elif self.Iter_type.lower() == "sor":
            self.__sor_iteration__()
        else:
            raise ValueError(
                "未知的求解方法,仅支持雅可比迭代法(jacobi),高斯赛德尔迭代法(gauss_seidel)或超松弛迭代法(SOR)"
            )

    def __print__(self):

        """
        打印输出结果
        """

        np.set_printoptions(formatter={'all': lambda x: str(Fraction(x).limit_denominator())})
        print("D矩阵：\n", self.D)
        print("L矩阵：\n", self.L)
        print("U矩阵：\n", self.U)
        if self.Iter_type.lower() == "jacobi":
            print("雅可比迭代法的迭代矩阵J：\n", self.J)
            print("雅可比迭代法的迭代矩阵f：\n", self.f)
            print("雅可比迭代法解方程组的解：")
        elif self.Iter_type.lower() == "gauss_seidel":
            print("高斯赛德尔迭代法的迭代矩阵G：\n", self.G)
            print("高斯赛德尔迭代法的迭代矩阵f：\n", self.f)
            print("高斯赛德尔迭代法解方程组的解：")
        elif self.Iter_type.lower() == "sor":
            print("超松弛迭代法的迭代矩阵L_w：\n", self.L_w)
            print("超松弛迭代法的迭代矩阵f：\n", self.f)
            print("超松弛迭代法解方程组的解：")
        print({"x" + str(i + 1): round(self.solution[i], 3) for i in range(len(self.solution))})
        print("最终的误差：%e" % self.error)
        print("迭代次数：%d" % self.iter_num)
        np.set_printoptions()

    def __jacobi_iteration__(self):

        """
        雅可比迭代法
        """

        self.J = np.linalg.inv(self.D).dot(self.L + self.U)
        self.f = np.linalg.inv(self.D).dot(self.b)
        self.solution_list[0] = self.x0.copy()  # 初始向量
        eigval, _ = np.linalg.eig(self.J)
        if np.max(np.abs(eigval)) > 1:
            raise ValueError("该方程组存在不稳定的迭代方法")
        i = 1
        while i < self.max_iter:
            self.solution_list[i] = self.J.dot(self.solution_list[i - 1]) + self.f
            # 退出循环，满足精度要求
            if np.linalg.norm((self.solution_list[i] - self.solution_list[i - 1]), np.inf) < self.epsilon:
                break
            i += 1
        self.iter_num = i  # 记录最终的迭代次数
        # 最终的误差
        self.error = np.linalg.norm((self.solution_list[i - 1] - self.solution_list[i - 2]), np.inf)
        # 方程组的解
        self.solution = self.solution_list[i - 1]

    def __gauss_seidel_iteration__(self):

        """
        高斯-赛德尔迭代法
        """

        self.G = np.linalg.inv(self.D - self.L).dot(self.U)
        self.f = np.linalg.inv(self.D - self.L).dot(self.b)
        self.solution_list[0] = self.x0.copy()  # 初始向量
        eigval, _ = np.linalg.eig(self.G)
        if np.max(np.abs(eigval)) > 1:
            raise ValueError("该方程组存在不稳定的迭代方法")
        i = 1
        while i < self.max_iter:
            self.solution_list[i] = self.G.dot(self.solution_list[i - 1]) + self.f
            # 退出循环，满足精度要求时退出循环
            if np.linalg.norm((self.solution_list[i] - self.solution_list[i - 1]), np.inf) < self.epsilon:
                break
            i += 1
        self.iter_num = i  # 记录最终的迭代次数
        # 最终的误差
        self.error = np.linalg.norm((self.solution_list[i - 1] - self.solution_list[i - 2]), np.inf)
        # 方程组的解
        self.solution = self.solution_list[i - 1]

    def __sor_iteration__(self):

        """
        超松弛迭代法
        """

        self.L_w = np.linalg.inv(self.D - self.w * self.L).dot((1 - self.w) * self.D + self.w * self.U)
        self.f = self.w * np.linalg.inv(self.D - self.w * self.L).dot(self.b)
        self.solution_list[0] = self.x0.copy()  # 初始向量
        eigval, _ = np.linalg.eig(self.L_w)
        if np.max(np.abs(eigval)) > 1:
            raise ValueError("该方程组存在不稳定的迭代方法")
        i = 1
        while i < self.max_iter:
            self.solution_list[i] = self.L_w.dot(self.solution_list[i - 1]) + self.f
            # 退出循环，满足精度要求时退出循环
            if np.linalg.norm((self.solution_list[i] - self.solution_list[i - 1]), np.inf) < self.epsilon:
                break
            i += 1
        self.iter_num = i  # 记录最终的迭代次数
        # 最终的误差
        self.error = np.linalg.norm((self.solution_list[i - 1] - self.solution_list[i - 2]), np.inf)
        # 方程组的解
        self.solution = self.solution_list[i - 1]
