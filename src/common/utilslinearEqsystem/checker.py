import numpy as np


class isMatrixReasonable:

    """
    定义一个装饰器类，用于检查传入的矩阵是否满足条件
    """

    def __init__(self, model_name: str):
        
        self.model_name = model_name

    
    def __call__(self, func):

        def wrapper(cls, matrix: np.ndarray, *args, **kwargs):
            
            if self.model_name == "GaussElimate":
                self._checkForGauss(matrix)
            if self.model_name == "Iteration":
                self._checkForIteration(matrix)
            if self.model_name == "RootSquare":
                self._checkForSquare(matrix)
            if self.model_name == "TriangleDecomposite":
                self._checkForTriangle(matrix)
            res = func(cls, matrix, *args, **kwargs)
            return res
        return wrapper
    

    def _checkForGauss(self, matrix: np.ndarray):

        """
        适用于高斯消去法
        """

        check_matrix = matrix.copy()
        if check_matrix[:, :-1].shape[0] != check_matrix[:, :-1].shape[1]:
            raise ValueError('线性方程组的系数矩阵不是方阵')
        if np.diag(check_matrix[:, :-1]).any() == 0:
            raise ValueError('线性方程组的系数矩阵的对角线元素存在0')
        if np.linalg.matrix_rank(check_matrix[:, :-1]) != np.linalg.matrix_rank(matrix):
            raise ValueError('该线性方程组无解')
        if np.linalg.matrix_rank(check_matrix[:, :-1]) == np.linalg.matrix_rank(matrix) \
           and np.linalg.matrix_rank(check_matrix[:, :-1]) != check_matrix[:, :-1].shape[0]:
            raise ValueError('该线性方程组有多个解,暂时支持有唯一解的方程组')


    def _checkForIteration(self, matrix: np.ndarray):

        """
        适用于迭代法
        """

        self._checkForGauss(matrix)

    
    def _checkForSquare(self, matrix: np.ndarray):

        """
        适用于平方根分解法
        """

        check_matrix = matrix.copy()
        if check_matrix[:,:-1].shape[0] != check_matrix[:,:-1].shape[1]:
            raise ValueError('线性方程组的系数矩阵不是方阵')
        if not np.allclose(check_matrix[:,:-1], check_matrix[:,:-1].T):
            raise ValueError('线性方程组的系数矩阵不是对称矩阵')
        if np.linalg.matrix_rank(check_matrix[:,:-1]) != np.linalg.matrix_rank(matrix):
            raise ValueError('该线性方程组无解')
        if np.linalg.matrix_rank(check_matrix[:,:-1]) == np.linalg.matrix_rank(matrix) \
           and np.linalg.matrix_rank(check_matrix[:,:-1]) < check_matrix[:,:-1].shape[0]:
            raise ValueError('该线性方程组有多个解,暂时支持有唯一解的方程组')
    

    def _checkForTriangle(self, matrix: np.ndarray):

        """
        适用于三角分解法
        """

        check_matrix = matrix.copy()
        if check_matrix[:,:-1].shape[0] != check_matrix[:,:-1].shape[1]:
            raise ValueError('线性方程组的系数矩阵不是方阵')
        if np.linalg.matrix_rank(check_matrix[:,:-1]) != np.linalg.matrix_rank(matrix):
            raise ValueError('该线性方程组无解')
        if np.linalg.matrix_rank(check_matrix[:,:-1]) == np.linalg.matrix_rank(matrix) \
           and np.linalg.matrix_rank(check_matrix[:,:-1]) < check_matrix[:,:-1].shape[0]:
            raise ValueError('该线性方程组有多个解,暂时支持有唯一解的方程组')
        