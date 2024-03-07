import numpy as np

def check_matrix(func):

    def wrapper(cls, matrix:np.ndarray, *args, **kwargs):

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
        result = func(cls, matrix, *args, **kwargs)
        return result
    
    return wrapper