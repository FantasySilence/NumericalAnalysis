import os
import sys
import numpy as np

# 设置全局路径，以便引入模块
test_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root_path = os.path.dirname(test_path)
sys.path.append(root_path)

# 导入对应模块
from src.modules.linearEqSystem.rootsqare import RootSquareDecomposition
from src.common.utilslinearEqsystem.MatrixMetrics import ParaMatrixMetrics

# 测试用例
# 对称非正定矩阵,精确解为[10/9,7/9,23/9]
matrix1 = np.array([[2, -1, 1, 4], [-1, -2, 3, 5], [1, 3, 1, 6]])
model1 = RootSquareDecomposition(matrix=matrix1, decom_type="ichol")
para1 = ParaMatrixMetrics(matrix=model1.A)
# 对称正定矩阵,精确解为[10,24,-9]
matrix2 = np.array([[1, 2, 6, 4], [2, 5, 15, 5], [6, 15, 46, 6]])
model2 = RootSquareDecomposition(matrix=matrix2, decom_type="chol")
para2 = ParaMatrixMetrics(matrix=model2.A)
