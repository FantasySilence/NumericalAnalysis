import os
import sys
import numpy as np

# 设置全局路径，以便引入模块
test_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root_path = os.path.dirname(test_path)
sys.path.append(root_path)

# 导入对应模块
from src.modules.linearEqSystem.iterativemethod import JGSIterationMethod

# 测试用例
matrix1 = np.array([[8, -3, 2, 20], [4, 11, -1, 33], [6, 3, 12, 36]])  # 精确解为[3,2,1]
matrix2 = np.array([[4, 3, 2, 2], [6, -3, -1, 7], [2, 6, 7, -5]])  # 精确解为[1,0,-1]
matrix3 = np.array([[-4, 1, 1, 1, 1], [1, -4, 1, 1, 1], [1, 1, -4, 1, 1], [1, 1, 1, -4, 1]])  # 精确解为[-1,-1,-1,-1]
model1 = JGSIterationMethod(matrix1, Iter_type="jacobi", max_iter=10)
model2 = JGSIterationMethod(matrix2, Iter_type="gauss_seidel", max_iter=10)
model3 = JGSIterationMethod(matrix3, Iter_type="sor", w=1.3, max_iter=10)
