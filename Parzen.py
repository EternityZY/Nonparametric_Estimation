# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

np.random.seed(1)
N = 20
X = np.concatenate((np.random.normal(0, 1, int(0.3 * N)), np.random.normal(5, 1, int(0.7 * N))))[:, np.newaxis]
# np.newaxis是None的意思
# 前半部分的平均值是0，方差是1
# int(0.3 * N)指的是输出多少数量符合要求的数据
# ---------------------------以上是创建数据集-----------------------------------------------------------------


X_plot = np.linspace(-5, 10, 1000)[:, np.newaxis]  # 创建等差数列，在-5和10之间取1000个数
bins = np.linspace(-5, 10, 10)  # 这个的作用是，在相邻两个边界时间的数据对应的y值都一样大
print("bins=", bins)
fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
fig.subplots_adjust(hspace=0.05, wspace=0.05)
# 直方图 1 'Histogram'
print("---------------------------------")
ax[0, 0].hist(X[:, 0], bins=bins, fc='#AAAAFF', normed=True)  # fc指的应该是颜色的编码
# 这里的ax[0,0]的意思是画在第几副图上


ax[0, 0].text(-3.5, 0.31, 'Histogram')  # -3.5, 0.31的意思是每张图的logo要画在什么地方
# 直方图 2 'Histogram, bins shifted'
ax[0, 1].hist(X[:, 0], bins=bins + 0.75, fc='#AAAAFF', normed=True)  # histogram的缩写
ax[0, 1].text(-3.5, 0.31, 'Histogram, bins shifted')  # 每个子图内画标签

# -----------------------------------------------------------------------------------


# 核密度估计 1 'tophat KDE'
kde = KernelDensity(kernel='tophat', bandwidth=0.75).fit(X)  # 什么是带宽
log_dens = kde.score_samples(X_plot)
# 所以这里有两组数据，X和X_plot，其实是在利用X_plot对X进行采样。
# 所以想要复用这段代码的时候，改X即可，X_plot不用修改


ax[1, 0].fill(X_plot[:, 0], np.exp(log_dens), fc='#AAAAFF')  # fill就是用来画概率密度的
ax[1, 0].text(-3.5, 0.31, 'Tophat Kernel Density')  # 设置标题的位置
# 核密度估计 2 'Gaussian KDE'
kde = KernelDensity(kernel='gaussian', bandwidth=0.75).fit(X)
log_dens = kde.score_samples(X_plot)  # 返回的是点x对应概率的log值，要使用exp求指数还原。
ax[1, 1].fill(X_plot[:, 0], np.exp(log_dens), fc='#AAAAFF')  # fill就是用来画概率密度的
# 所以上面一句代码就非常清晰了，X_plot[:, 0]是具体数据，np.exp(log_dens)指的是该数据对应的概率
ax[1, 1].text(-3.5, 0.31, 'Gaussian Kernel Density')  # 设置标题的位置

print("ax.ravel()=", ax.ravel())
print("X.shape[0]=", X.shape[0])
print("X=", X)

# 这个是为了在每个子图的下面画一些没用的标记，不看也罢
for axi in ax.ravel():
    axi.plot(X[:, 0], np.zeros(X.shape[0]) - 0.01, '+k')
    axi.set_xlim(-4, 9)  # 设定上下限
    axi.set_ylim(-0.02, 0.34)

##画图过程是两行两列，这里是遍历第1列，每个位置的左侧画一个“xNormalized Density”
for axi in ax[:, 0]:
    print("axi=", axi)
    axi.set_ylabel('Normalized Density')

##画图过程是两行两列，这里是遍历第2行，每个位置画一个“x”
for axi in ax[1, :]:
    axi.set_xlabel('x')
plt.show()




# ravel函数的作用如下：
# >>> x = np.array([[1, 2, 3], [4, 5, 6]])
# >>> print(np.ravel(x))
# [1 2 3 4 5 6]