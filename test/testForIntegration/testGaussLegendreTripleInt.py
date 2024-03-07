import os
import sys
import numpy as np

# 设置全局路径，以便引入模块
test_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root_path = os.path.dirname(test_path)
sys.path.append(root_path)

# 导入对应模块
from src.modules.intergration.gausslegendretripleint\
import GuassLegendreTripleIntergration

# 测试用例
def fun(x,y,z):
    return 4*x*z*np.exp(-x**2*y-z**2)

int_precision = 1.7327622230312205
glti = GuassLegendreTripleIntergration(fun, [0,1], [0,np.pi], [0,np.pi], zeros_num=[11, 12, 15])
res = glti.cal_3d_int()
print("积分值：%.15f, 精度：%.15e"%(res, int_precision-res))