import os
import sys
import numpy as np

# 设置全局路径，以便引入模块
test_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root_path = os.path.dirname(test_path)
sys.path.append(root_path)

# 导入对应模块
from src.modules.interpolation.hermite import HermiteInterpolation
from src.modules.interpolation.hermite import Piecewise2CubicHermiteInterpolation

# 测试数据
x=np.linspace(0,2*np.pi,10,endpoint=True)
y=np.sin(x)*np.cos(x)
dy=np.cos(x)
d2y=-np.sin(x)
x0=np.array([np.pi/2,2.158,3.58,4.784])

# 埃尔米特插值(带1阶导数值)调用示例
her_Interp=HermiteInterpolation(x=x,y=y,dy=dy)
her_Interp.fit_interp()
print('埃尔米特插值多项式如下:')
print(her_Interp.polynomial)
print('埃尔米特插值多项式系数向量和对应阶次:')
print(her_Interp.poly_coefficient)
print(her_Interp.coefficient_order)
y0=her_Interp.cal_interp(x0)
print('所求插值点的值为：',y0)
her_Interp.plt_interp(x0,y0)

# 两点三次埃尔米特插值调用示例
her2_Interp=Piecewise2CubicHermiteInterpolation(x,y,dy)
her2_Interp.fit_interp()
print(her2_Interp.poly_coefficient)
y0=her2_Interp.cal_interp(x0)
print('所求插值点的值为：',y0)
her2_Interp.plt_interp(x0,y0)