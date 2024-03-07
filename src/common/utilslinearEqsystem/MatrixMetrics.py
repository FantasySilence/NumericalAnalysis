import numpy as np
import pandas as pd

class ParaMatrixMetrics:

    """
    线性方程组的系数矩阵的一些算数特征计算
    例如矩阵条件数，算子范数等
    """

    def __init__(self, matrix:np.ndarray, matrix_name:str=None):

        """
        参数初始化\n
        matrix: 待计算的矩阵\n
        matrix_name: 矩阵的名字\n
        如果想要查询算数特征以便于调用,\n
        请查询类属性self.number_dict, self.number_list\n
        self.number_list中数值的顺序与self.number_dict一致
        """

        self.F_norm = np.linalg.norm(matrix, 'fro')         # F范数
        self.one_norm = np.linalg.norm(matrix, 1)           # 1范数
        self.two_norm = np.linalg.norm(matrix, 2)           # 2范数
        self.inf_norm = np.linalg.norm(matrix, np.inf)      # 无穷范数
        self.inf_cond = np.linalg.cond(matrix, np.inf)      # 无穷条件数
        self.two_cond = np.linalg.cond(matrix, 2)           # 2条件数
        eigval, _ = np.linalg.eig(matrix)
        self.spe_radius = np.max(np.abs(eigval))            # 谱半径
        self.matrix_name = matrix_name      # 矩阵的名字
        # 将相应的范数等数字存在字典中便于之后可能的调用
        self.number_dict = {"F范数": self.F_norm, "无穷条件数": self.inf_cond, "2-条件数":self.two_cond,
                            "1范数": self.one_norm, "2范数": self.two_norm, "无穷范数": self.inf_norm,
                            "谱半径": self.spe_radius}
        self.number_list = [self.F_norm, self.inf_cond, self.two_cond, self.one_norm, 
                            self.two_norm, self.inf_norm, self.spe_radius]
        self.__output__()
    

    def __output__(self):

        """
        输出算数特征
        """

        pd.set_option('display.float_format', lambda x: '%.3f' % x)
        pd.set_option('display.unicode.ambiguous_as_wide', True)
        pd.set_option('display.unicode.east_asian_width', True)

        print("="*70)
        df = pd.DataFrame(columns=['F范数', '无穷条件数', '2-条件数', '1-范数', '2-范数', '无穷范数', '谱半径'])
        df.loc[0] = [self.F_norm, self.inf_cond, self.two_cond, self.one_norm, 
                     self.two_norm, self.inf_norm, self.spe_radius]
        if self.matrix_name:
            df.index = [self.matrix_name]
        else:
            df.index = ['矩阵']
        print(df)
        print("="*70)


if __name__ == '__main__':

    a = np.array([[1, 2], [3, 4]])
    b = np.array([[0,-3/4,-1/2],[2,0,-1/3],[-2/7,-6/7,0]])
    model = ParaMatrixMetrics(b)


class IterationMetrics:

    """
    迭代法相关的算数特征计算
    例如收敛速度等
    """

    def __init__(self, matrix:np.ndarray, matrix_name:str=None):

        """
        参数初始化\n
        matrix: 待计算的矩阵\n
        matrix_name: 矩阵的名字\n
        """

        self.matrix_name = matrix_name      # 矩阵的名字
        