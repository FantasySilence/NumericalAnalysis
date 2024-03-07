# NumericalAnalysis

2023-2024-1学期学习《数值分析》所学习的内容

<center>(更新中......)</center>


## 一、插值算法

### 1.[拉格朗日插值](src/modules/interpolation/lagrange.py)

​		对某个多项式函数，已知有给定的$k+1$个取值点：
$$
(x_0, y_0), (x_1, y_1), ...,(x_k,y_k)
$$
其中，$x_j$对应着自变量的位置，而$y_j$对应着函数在这个位置的取值。

假设任意两个不同的$x_j$都互不相同，那么应用拉格朗日插值公式所得到的拉格朗日插值多项式为：
$$
L(x)=\sum_{j}^{k}y_jl_j(x)
$$
其中，每个$l_j(x)$为拉格朗日基本多项式（或称“插值基函数”），其表达式为：
$$
l_j(x)=\prod_{i=0,i\neq{j}}^{k}\frac{x-x_i}{x_j-x_i}=\frac{x-x_0}{x_j-x_0}\cdot\cdot\cdot\frac{x-x_{j-1}}{x_j-x_{j-1}}\frac{x-x_{j+1}}{x_j-x_{j+1}}\cdot\cdot\cdot\frac{x-x_k}{x_j-x_k}
$$
拉格朗日插值基函数的$l_j(x)$的特点是在$x_j$上的取值为1，在其他的点$x_i, i\neq{j}$上取值为0.

​		若在$[a,b]$上用$L_n(x)$近似$f(x)$，则其截断误差为$R_n(x)=f(x)-L_n(x)$，也成为插值多项式的余项，关于插值余项的估计有如下的定理：

​		设$f^{(n)}(x)$在$[a,b]$上连续，$f^{(n+1)}(x)$在$(a,b)$内存在，节点$a \le x_0<x_1<\cdots <x_n \le b$，$L_n(x)$时满足要求的插值多项式，则对任何$x \in [a,b]$插值余项为：
$$
R_n(x)=f(x)-L_n(x)=\frac{f^{(n+1)}(\xi)}{(n+1)!}\omega_{n+1}(x)
$$
其中，$\omega_{n+1}=(x-x_0)(x-x_1)\cdots(x-x_n)$，$\xi \in (a,b)$且依赖于$x$。

### 2.[牛顿均差插值](src/modules/interpolation/newton.py)

​		

### 3.[牛顿差分插值](src/modules/interpolation/newtonDiff.py)

### 4.[分段线性插值](src/modules/interpolation/piecewiselinear.py)

### 5.[埃尔米特插值](src/modules/interpolation/hermite.py)

### 6.[三次样条插值](src/modules/interpolation/cubicspline.py)

## 二、函数逼近

## 三、数值积分

## 四、线性方程组求解的直接法与迭代法

## 五、非线性方程(组)求解的迭代法

## 六、常微分方程的数值解法

### 代码(临时目录，便于索引)

- 项目中包含了插值，逼近，数值积分，(非)线性方程组求解的直接法与迭代法，ODE的数值解法
- 同时给出了示例与测试代码予以参考
  - 插值方法[src/modules/interpolation](src/modules/interpolation)
  - 插值方法的示例[src/test/testForInterpolation](src/test/testForInterpolation)
  - 函数逼近[src/modules/approximation](src/modules/approximation)
  - 函数逼近的示例[test/testForAppproximation](test/testForAppproximation)
  - 数值积分[src/modules/intergration](src/modules/intergration)
  - 数值积分的示例[test/testForIntegration](test/testForIntegration)
  - 线性方程组求解的直接法和迭代法[src/modules/linearEqSystem](src/modules/linearEqSystem)
  - 线性方程组求解的示例[test/testForLinearEqSystem](test/testForLinearEqSystem)
  - 非线性方程(组)求解的迭代法[src/modules/nonLinearEqSystem](src/modules/nonLinearEqSystem)
  - 非线性方程(组)求解的示例[test/testForNonLinearEqSystem](test/testForNonLinearEqSystem)
  - ODE的数值解法[src/modules/numeriaclOde](src/modules/numeriaclOde)
  - ODE的数值解法示例[test/testForNumericalOde](test/testForNumericalOde)


### 技术栈

Python

### License

GPL-3.0 license
