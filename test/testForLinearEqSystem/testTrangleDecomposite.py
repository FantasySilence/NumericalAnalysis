import os
import sys
import numpy as np

# 设置全局路径，以便引入模块
test_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root_path = os.path.dirname(test_path)
sys.path.append(root_path)

# 导入对应模块
from src.modules.linearEqSystem.triangledecomposite import TriangleDecomposition
from src.common.utilslinearEqsystem.MatrixMetrics import ParaMatrixMetrics

# 测试用例
matrix1 = np.array([[1,2,3,14],[2,5,2,18],[3,1,5,20]])
matrix2 = np.array([[10,7,8,7,32],[7,5,6,5,23],[8,6,10,9,33],[7,5,9,10,31]])
matrix3 = np.array([[1,1,1,4],[2,1,3,7],[3,1,6,2]])
model1 = TriangleDecomposition(matrix=matrix1, decom_type="LU")
para1 = ParaMatrixMetrics(matrix=model1.A)
model2 = TriangleDecomposition(matrix=matrix2, decom_type="PLU")
para2 = ParaMatrixMetrics(matrix=model2.A)