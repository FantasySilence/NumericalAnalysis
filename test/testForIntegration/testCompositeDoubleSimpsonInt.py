import os
import sys
import numpy as np

# 设置全局路径，以便引入模块
test_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root_path = os.path.dirname(test_path)
sys.path.append(root_path)

# 导入对应模块
from src.modules.intergration.compositedoublesimpsonint\
import CompositeDoubleSimpsonIntergration

# 测试用例
def fun(x,y):
    return np.exp(-x**2-y**2)

cdsi = CompositeDoubleSimpsonIntergration(fun, [0,1], [0,1], eps=1e-15)
cdsi.cal_2d_int()
print("划分区间数:%d, 积分近似值:%.15f"%(cdsi.internal_num, cdsi.int_value))
cdsi.plt_precision()