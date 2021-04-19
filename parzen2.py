import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb, perm

sample_num = 100000
# 获取要拟合的分布抽样并排序 Y = 5-10*(1-X)**0.5
ran = np.random.rand(sample_num)
ran = 5 - 10 * (1 - ran) ** 0.5
ran = np.sort(ran)


# 高斯核
def ker_gass(x0):
    return (1 / (2 * np.pi) ** 0.5) * np.e ** -(x0 ** 2 / 2)


# Epanechnikov核
def ker_Epanechnikov(x0):
    return 3 / 4 * (1 - x0 ** 2)


# 拟合概率密度函数
def fitting_proba_density(X, h, way):
    if way == 1:  # 使用均匀核
        i_X = 0
        begin_ran = 0
        end_ran = 0
        sum0 = np.zeros(len(X) + 1)
        while i_X < len(X):
            while begin_ran < sample_num:
                if X[i_X] - h > ran[begin_ran]:
                    sum0[i_X] -= 0.5
                    begin_ran += 1
                else:
                    break
            while end_ran < sample_num:
                if X[i_X] + h >= ran[end_ran]:
                    sum0[i_X] += 0.5
                    end_ran += 1
                else:
                    break
            sum0[i_X + 1] = sum0[i_X]
            i_X += 1
        return sum0[0:-1] / h / sample_num
    elif way == 2:  # 使用高斯核
        sum0 = np.zeros(len(X))
        for i in range(sample_num):
            sum0 += ker_gass((ran[i] - X) / h)
        return sum0 / h / sample_num
    else:  # 使用Epanechnikov核
        i_X = 0
        begin_ran = 0
        end_ran = 0
        sum0 = np.zeros(len(X))
        while i_X < len(X):
            while begin_ran < sample_num:
                if X[i_X] - h > ran[begin_ran]:
                    begin_ran += 1
                else:
                    break
            while end_ran < sample_num:
                if X[i_X] + h >= ran[end_ran]:
                    end_ran += 1
                else:
                    break
            i = begin_ran
            while i < end_ran:
                sum0[i_X] += ker_Epanechnikov((ran[i] - X[i_X]) / h)
                i += 1
            i_X += 1
        return sum0 / h / sample_num


# 画出拟合概率密度
def paint_(a):
    X = np.linspace(-10, 10, 500)
    j = 0
    for h in a:
        j += 1
        ax = plt.subplot(2, 2, j)
        ax.set_title('h=' + str(h))  # 设置子图

        X0 = np.linspace(-5, 5, 10)
        Y0 = (-X0 + 5) / 50
        plt.plot(X0, Y0, label='Probability density')  # 分布密度函数

        Y = fitting_proba_density(X, h, 1)  # 均匀核
        ax.plot(X, Y, label='Uniform kernel')
        Y = fitting_proba_density(X, h, 2)  # 高斯核
        ax.plot(X, Y, label='Gassian kernel')
        Y = fitting_proba_density(X, h, 3)  # Epanechnikov核
        ax.plot(X, Y, label='Epanechnikov kernel')
        ax.legend()


paint_([0.1, 0.3, 0.7, 1.4])

# 图像参数
plt.xlim(-10, 10)
plt.show()
