import numpy as np
import pandas as pd
from fractions import Fraction


class SolutionToQuesThree:

    """
    第三问
    """

    def __init__(self, matrix: np.ndarray, x0=None, max_iter: int = 1000, epsilon: float = 1e-15,
                 is_print: bool = True):

        """
        参数初始化\n
        matrix: 方程组的增广矩阵\n
        x0: 初始向量\n
        epsilon: 解的精度\n
        如果想获取迭代后的方程组的解以便调用,请查询类属性self.solution
        """

        self.epsilon = epsilon
        self.max_iter = max_iter
        self.is_print = is_print
        self.iter_num = 0  # 初始化，用于记录迭代次数
        check_matrix = matrix.copy()
        if check_matrix[:, :-1].shape[0] != check_matrix[:, :-1].shape[1]:
            raise ValueError('线性方程组的系数矩阵不是方阵')
        if np.diag(check_matrix[:, :-1]).any() == 0:
            raise ValueError('线性方程组的系数矩阵的对角线元素存在0')
        if np.linalg.matrix_rank(check_matrix[:, :-1]) != np.linalg.matrix_rank(matrix):
            raise ValueError('该线性方程组无解')
        if np.linalg.matrix_rank(check_matrix[:, :-1]) == np.linalg.matrix_rank(matrix) \
                and np.linalg.matrix_rank(check_matrix[:, :-1]) < check_matrix[:, :-1].shape[0]:
            raise ValueError('该线性方程组有多个解,暂时支持有唯一解的方程组')
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
        self.G = None  # 高斯赛德尔迭代法的迭代矩阵
        self.f = None  # 迭代矩阵
        self.solution_list = np.zeros(self.max_iter, dtype=object)  # 存储迭代过程中方程的解
        self.solution = None  # 方程组的解
        self.res_df = None  # 存储迭代过程中方程组的解，尝试以DataFrame存储
        self.__solve__()
        if is_print:
            self.__print__()

    def __solve__(self):

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
        # 储存迭代过程
        self.res_df = pd.DataFrame()
        self.res_df["迭代次数"] = list(range(self.iter_num))
        self.res_df["Xn"] = self.solution_list[:self.iter_num]
        self.res_df.set_index("迭代次数", inplace=True)

    def __print__(self):

        """
        打印输出结果
        """

        np.set_printoptions(formatter={'all': lambda x: str(Fraction(x).limit_denominator())})
        print("D矩阵：\n", self.D)
        print("L矩阵：\n", self.L)
        print("U矩阵：\n", self.U)
        print("迭代过程：\n", self.res_df)
        print("高斯赛德尔迭代法的迭代矩阵G：\n", self.G)
        print("高斯赛德尔迭代法的迭代矩阵f：\n", self.f)
        print("高斯赛德尔迭代法解方程组的解：")
        print({"x" + str(i + 1): round(self.solution[i], 3) for i in range(len(self.solution))})
        print("最终的误差：%e" % self.error)
        print("迭代次数：%d" % self.iter_num)
        np.set_printoptions()
