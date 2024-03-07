import os
import sys
import numpy as np

# 设置全局路径，以便引入模块
test_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root_path = os.path.dirname(test_path)
sys.path.append(root_path)

# 导入对应模块
from src.modules.linearEqSystem.gausselimate import GuassElimation
from src.common.utilslinearEqsystem.MatrixMetrics import ParaMatrixMetrics

# 测试用例
matrix = np.array([[12,-3,3,15],[-18,3,-1,-15],[1,1,1,6]])      # 精确解为[1,2,3]
guass = GuassElimation(matrix, solve_type="order")
para1 = ParaMatrixMetrics(matrix=guass.A)

matrix2 = np.array([[4,3,2,2],[6,-3,-1,7],[2,6,7,-5]])      # 精确解为[1,0,-1]
guass2 = GuassElimation(matrix2, solve_type="pca")
para2 = ParaMatrixMetrics(matrix=guass2.A)