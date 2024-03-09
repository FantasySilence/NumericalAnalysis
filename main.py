"""
-----------------------------------------
Author: @Fantasy_Silence
Time: 2024-03-09
IDE: Visual Studio Code  Python: 3.9.7
-----------------------------------------
这是主程序入口，在这里同一调用各个模块解决问题
"""

# 一个简单的例子
import numpy as np
from src.modules.linearEqSystem.gausselimate import GuassElimation

matrix = np.array([[12, -3, 3, 15], [-18, 3, -1, -15], [1, 1, 1, 6]])  # 精确解为[1,2,3]
guass = GuassElimation(matrix, solve_type="order")
