import os
import sys
import numpy as np

# 设置全局路径，以便引入模块
test_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root_path = os.path.dirname(test_path)
sys.path.append(root_path)

# 导入对应模块
from src.modules.intergration.gausslegendredoubleint \
    import GuassLegendreDuobleIntergration


# 测试用例
def fun(x, y):
    return np.exp(-x ** 2 - y ** 2)


gldi = GuassLegendreDuobleIntergration(fun, [0, 1], [0, 1], zeros_num=15)
res = gldi.cal_2d_int()
print("二重积分结果：", res)
