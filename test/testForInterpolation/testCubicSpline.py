import os
import sys
import numpy as np

# 设置全局路径，以便引入模块
test_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root_path = os.path.dirname(test_path)
sys.path.append(root_path)

# 导入对应模块
from src.modules.interpolation.cubicspline import CubicSplineInterpolation

# 测试数据
x = np.linspace(0, 2 * np.pi, 10, endpoint=True)
y = np.sin(x) * np.cos(x)
dy = np.cos(x)
d2y = -np.sin(x)
x0 = np.array([np.pi / 2, 2.158, 3.58, 4.784])

# 三次样条插值调用示例
# boundary_type = ['complete','second','natural','periodic']
csi_Interp = CubicSplineInterpolation(x, y, dy=None, d2y=None, boundary_type="periodic")
csi_Interp.fit_interp()
print(csi_Interp.poly_coefficient)
print(csi_Interp.polynomial)
y0 = csi_Interp.cal_interp(x0)
print("所求插值点的值为：", y0)
csi_Interp.plt_interp(x0, y0)
