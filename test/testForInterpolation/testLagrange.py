import os
import sys
import numpy as np

# 设置全局路径，以便引入模块
test_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root_path = os.path.dirname(test_path)
sys.path.append(root_path)

# 导入对应模块
from src.modules.interpolation.lagrange import LagrangeInterpolation

# 测试数据
x = np.linspace(0, 24, 13, endpoint=True)
y = np.array([12, 9, 9, 10, 18, 24, 28, 27, 25, 20, 18, 15, 13])
x0 = np.array([1, 10.5, 13, 18.7, 22.3])

# 调用示例
lag_Interp = LagrangeInterpolation(x=x, y=y)
lag_Interp.fit_interp()
print('拉格朗日插值多项式如下:')
print(lag_Interp.polynomial)
print('拉格朗日插值多项式系数向量和对应阶次:')
print(lag_Interp.poly_coefficient)
print(lag_Interp.coefficient_order)
y1 = lag_Interp.cal_interp(x0)
print('所求插值点的值为：', y1)
lag_Interp.plt_interp(x0)
