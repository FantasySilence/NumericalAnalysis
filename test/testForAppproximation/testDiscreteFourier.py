import os
import sys
import numpy as np

# 设置全局路径，以便引入模块
test_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root_path = os.path.dirname(test_path)
sys.path.append(root_path)

# 导入对应模块
from src.modules.approximation.discretefourier import DiscreteFourierTransformApproximation


# 测试用例
def fun(x):
    return x ** 4 - 3 * x ** 3 + 2 * x ** 2 - np.tan(x * (x - 2))


x = np.linspace(0, 2, 20, endpoint=False)
y = fun(x)
dft = DiscreteFourierTransformApproximation(y, x_span=[0, 2], fun=fun)
dft.fit_approximation()
print('正弦项系数为：', dft.sin_term)
print('余弦项系数为：', dft.cos_term)
print('离散傅里叶变换逼近多项式：')
print(dft.approximation_poly)
dft.plt_approximation()
