import os
import sys
import numpy as np

# 设置全局路径，以便引入模块
test_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root_path = os.path.dirname(test_path)
sys.path.append(root_path)

# 导入对应模块
from src.modules.interpolation.newtonDiff import NewtonDiffInterpolation

# 测试数据
x = np.linspace(0, 24, 13, endpoint=True)
y = np.array([12, 9, 9, 10, 18, 24, 28, 27, 25, 20, 18, 15, 13])
x0 = np.array([1, 10.5, 13, 18.7, 22.3])

# 牛顿差分插值调用示例
Ndi_Interp = NewtonDiffInterpolation(x=x, y=y, diff_type='backward')
Ndi_Interp.__diff_matrix__()
Ndi_Interp.fit_interp()
print('牛顿差分插值多项式如下:')
print(Ndi_Interp.polynomial)
print('牛顿差分插值多项式系数向量和对应阶次:')
print(Ndi_Interp.poly_coefficient)
print(Ndi_Interp.coefficient_order)
y3 = Ndi_Interp.cal_interp(x0)
print('所求插值点的值为：', y3)
Ndi_Interp.plt_interp(x0, y3)
