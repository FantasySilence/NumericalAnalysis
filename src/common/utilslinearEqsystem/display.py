import numpy as np
from sympy import Rational
from sympy import Matrix


def sci_display(arr: np.ndarray):
    """
    让矩阵显示出来(以数学的形式)
    """

    sci_arr = np.zeros(arr.shape, dtype=object)
    # 如果是一维数组(向量)
    if len(arr.shape) == 1:
        for i in range(arr.shape[0]):
            sci_arr[i] = Rational(arr[i]).limit_denominator()
        return Matrix(sci_arr)
    # 如果是二维数组(矩阵)
    elif len(arr.shape) == 2:
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                sci_arr[i, j] = Rational(arr[i, j]).limit_denominator()
        return Matrix(sci_arr)
