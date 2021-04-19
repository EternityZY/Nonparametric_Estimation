import random

import numpy as np
import math
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from TwoNormal import TwoNomal


def cube(u):
    T = abs(u)
    if all(t <= 0.5 for t in T):
        return 1
    else:
        return 0


def gaussian_1D(u):
    t = 1 / ((math.pi * 2) ** 0.5)
    return t * math.exp(- (u ** 2) / 2)


def guaasin_ND(u, d=2):
    t = 1/((math.pi*2)**(0.5*d))
    return t * math.exp(- (np.dot(u.T,u)) / 2)


def exp(u):
    return math.exp(-abs(u))

# Epanechnikov核
def Epanechnikov(u):
    T = abs(u)
    if all(t <= 1 for t in T):
        return 3 / 4 * (1-u**2)
    else:
        return 0


def Parzen(Data, X, h, d, f):
    Prob = []
    n = len(Data)
    for x in X:
        p = 0.0
        for s in Data:
            p += f((s - x) / h)
        Prob.append(p / (n * (h ** d)))
    return np.array(Prob)


def eudistance(x, y):
    d = 0.0
    T = x - y
    for t in T:
        d += t ** 2
    return (d ** 0.5)

def knn(Data, X, kn, d, f):
    t = kn / len(Data)
    Prob = []
    for x in X:
        dis = []
        for s in Data:
            dis.append(f(x, s))
        dis.sort()
        v = (dis[kn] * 2) ** d
        Prob.append(t / v)
    return np.array(Prob)

def test_1D_parzen(way=2, humped=2):

    np.random.seed(12)
    for n in [10000]:

        plt.figure(figsize=(13, 6))
        if way == 1:
            if humped == 1:
                Data = np.random.normal(0, 2, n).reshape([-1, 1])
            elif humped == 2:
                Data = np.zeros(n)
                for i in range(n):
                    prob = np.random.rand()
                    if prob>0.5:
                        Data[i] = np.random.normal(-5, 1)
                    else:
                        Data[i] = np.random.normal(5, 2)
        elif way==2:
            # 获取要拟合的分布抽样并排序 Y = 5-10*(1-X)**0.5
            ran = np.random.rand(n)
            ran = 5 - 10 * (1 - ran) ** 0.5
            Data = np.sort(ran)


        X = np.arange(-10, 10, 0.1).reshape([-1, 1])

        h = [0.05, 0.1, 0.5, 1]
        Prob1 = Parzen(Data, X, h=h[0], d=1, f=cube)
        ax = plt.subplot(3, 4, 1)
        ax.set_title("n="+str(n)+" h={:.2f}".format(h[0]))
        ax.plot(X, Prob1)
        plt.ylabel("Parzen: cube")

        Prob2 = Parzen(Data, X, h=h[1], d=1, f=cube)
        ax = plt.subplot(3, 4, 2)
        ax.set_title("n="+str(n)+" h={:.1f}".format(h[1]))
        ax.plot(X, Prob2)

        Prob3 = Parzen(Data, X, h=h[2], d=1, f=cube)
        ax = plt.subplot(3, 4, 3)
        ax.set_title("n="+str(n)+" h={:.1f}".format(h[2]))
        ax.plot(X, Prob3)

        Prob4 = Parzen(Data, X, h=h[3], d=1, f=cube)
        ax = plt.subplot(3, 4, 4)
        ax.set_title("n="+str(n)+" h={:.1f}".format(h[3]))
        ax.plot(X, Prob4)

        Prob1 = Parzen(Data, X, h=h[0], d=1, f=gaussian_1D)
        ax = plt.subplot(3, 4, 5)
        ax.set_title("n="+str(n)+" h={:.2f}".format(h[0]))
        ax.plot(X, Prob1)
        plt.ylabel("Parzen: gaussian")

        Prob2 = Parzen(Data, X, h=h[1], d=1, f=gaussian_1D)
        ax = plt.subplot(3, 4, 6)
        ax.set_title("n="+str(n)+" h={:.1f}".format(h[1]))
        ax.plot(X, Prob2)

        Prob3 = Parzen(Data, X, h=h[2], d=1, f=gaussian_1D)
        ax = plt.subplot(3, 4, 7)
        ax.set_title("n="+str(n)+" h={:.1f}".format(h[2]))
        ax.plot(X, Prob3)

        Prob4 = Parzen(Data, X, h=h[3], d=1, f=gaussian_1D)
        ax = plt.subplot(3, 4, 8)
        ax.set_title("n="+str(n)+" h={:.1f}".format(h[3]))
        ax.plot(X, Prob4)


        Prob1 = Parzen(Data, X, h=h[0], d=1, f=exp)
        ax = plt.subplot(3, 4, 9)
        ax.set_title("n="+str(n)+" h={:.2f}".format(h[0]))
        ax.plot(X, Prob1)
        plt.ylabel("Parzen: Epanechnikov")

        Prob2 = Parzen(Data, X, h=h[1], d=1, f=exp)
        ax = plt.subplot(3, 4, 10)
        ax.set_title("n="+str(n)+" h={:.1f}".format(h[1]))
        ax.plot(X, Prob2)

        Prob3 = Parzen(Data, X, h=h[2], d=1, f=exp)
        ax = plt.subplot(3, 4, 11)
        ax.set_title("n="+str(n)+" h={:.1f}".format(h[2]))
        ax.plot(X, Prob3)


        Prob4 = Parzen(Data, X, h=h[3], d=1, f=exp)
        ax = plt.subplot(3, 4, 12)
        ax.set_title("n="+str(n)+" h={:.1f}".format(h[3]))
        ax.plot(X, Prob4)

        plt.show()


def test_1D_parzen_h(way=1, humped=2):


    np.random.seed(12)
    for n in [1000]:

        plt.figure(figsize=(13, 6))
        if way == 1:
            if humped == 1:
                Data = np.random.normal(0, 2, n).reshape([-1, 1])
            elif humped == 2:
                Data = np.zeros(n)
                for i in range(n):
                    prob = np.random.rand()
                    if prob>0.5:
                        Data[i] = np.random.normal(-5, 1)
                    else:
                        Data[i] = np.random.normal(5, 2)
        elif way==2:
            # 获取要拟合的分布抽样并排序 Y = 5-10*(1-X)**0.5
            ran = np.random.rand(n)
            ran = 5 - 10 * (1 - ran) ** 0.5
            Data = np.sort(ran)


        X = np.arange(-10, 10, 0.1).reshape([-1, 1])

        h = [0.1, 0.5, 1, 4]
        Prob1 = Parzen(Data, X, h=h[0], d=1, f=cube)
        ax = plt.subplot(3, 4, 1)
        ax.set_title("n="+str(n)+" h={:.2f}".format(h[0]))
        ax.plot(X, Prob1)
        plt.ylabel("Parzen: cube")

        Prob2 = Parzen(Data, X, h=h[1], d=1, f=cube)
        ax = plt.subplot(3, 4, 2)
        ax.set_title("n="+str(n)+" h={:.1f}".format(h[1]))
        ax.plot(X, Prob2)

        Prob3 = Parzen(Data, X, h=h[2], d=1, f=cube)
        ax = plt.subplot(3, 4, 3)
        ax.set_title("n="+str(n)+" h={:.1f}".format(h[2]))
        ax.plot(X, Prob3)

        Prob4 = Parzen(Data, X, h=h[3], d=1, f=cube)
        ax = plt.subplot(3, 4, 4)
        ax.set_title("n="+str(n)+" h={:.1f}".format(h[3]))
        ax.plot(X, Prob4)

        Prob1 = Parzen(Data, X, h=h[0], d=1, f=gaussian_1D)
        ax = plt.subplot(3, 4, 5)
        ax.set_title("n="+str(n)+" h={:.2f}".format(h[0]))
        ax.plot(X, Prob1)
        plt.ylabel("Parzen: gaussian")

        Prob2 = Parzen(Data, X, h=h[1], d=1, f=gaussian_1D)
        ax = plt.subplot(3, 4, 6)
        ax.set_title("n="+str(n)+" h={:.1f}".format(h[1]))
        ax.plot(X, Prob2)

        Prob3 = Parzen(Data, X, h=h[2], d=1, f=gaussian_1D)
        ax = plt.subplot(3, 4, 7)
        ax.set_title("n="+str(n)+" h={:.1f}".format(h[2]))
        ax.plot(X, Prob3)

        Prob4 = Parzen(Data, X, h=h[3], d=1, f=gaussian_1D)
        ax = plt.subplot(3, 4, 8)
        ax.set_title("n="+str(n)+" h={:.1f}".format(h[3]))
        ax.plot(X, Prob4)


        Prob1 = Parzen(Data, X, h=h[0], d=1, f=exp)
        ax = plt.subplot(3, 4, 9)
        ax.set_title("n="+str(n)+" h={:.2f}".format(h[0]))
        ax.plot(X, Prob1)
        plt.ylabel("Parzen: Epanechnikov")

        Prob2 = Parzen(Data, X, h=h[1], d=1, f=exp)
        ax = plt.subplot(3, 4, 10)
        ax.set_title("n="+str(n)+" h={:.1f}".format(h[1]))
        ax.plot(X, Prob2)

        Prob3 = Parzen(Data, X, h=h[2], d=1, f=exp)
        ax = plt.subplot(3, 4, 11)
        ax.set_title("n="+str(n)+" h={:.1f}".format(h[2]))
        ax.plot(X, Prob3)


        Prob4 = Parzen(Data, X, h=h[3], d=1, f=exp)
        ax = plt.subplot(3, 4, 12)
        ax.set_title("n="+str(n)+" h={:.1f}".format(h[3]))
        ax.plot(X, Prob4)

        plt.show()


def test_2D_parzen(way=3):

    for n in [10, 100, 1000, 10000]:
        if way==1:
            Data = np.random.randn(n, 2)


        elif way==2:
            datax = np.hstack([np.random.randn(n) * 3 - 1,
                               np.random.randn(n) * 3 + 1])
            datay = np.hstack([np.random.randn(n) * 1 + 2,
                               np.random.randn(n) * 1 - 2])
            Data = np.concatenate([datax[..., np.newaxis], datay[..., np.newaxis]], axis=1)

        elif way==3:
            Data = np.zeros((n, 2))
            for i in range(n):

                cov = [[1, 0], [0, 1]]
                prob = np.random.rand()
                if prob > 0.5:
                    Data[i] = np.random.multivariate_normal((-2, -2), cov)
                else:
                    Data[i] = np.random.multivariate_normal((2, 2), cov)



        x = np.arange(-5, 5, 0.1)
        y = np.arange(-5, 5, 0.1)
        X = []
        for i in x:
            for j in y:
                X.append([i, j])
        X = np.array(X)

        plt.figure(figsize=(12, 6))

        h=[0.1, 0.5, 1, 2]
        Prob1 = Parzen(Data, X, h=h[0], d=2, f=cube)
        ax = plt.subplot(2, 4, 1, projection='3d')
        ax.set_title("n={:d} h={:.2f}".format(n, h[0]))
        ax.plot_trisurf(X[:, 0], X[:, 1], Prob1, cmap='rainbow')
        ax.text2D(0.5,1,"Parzen: cube", rotation=90, va='center')

        Prob2 = Parzen(Data, X, h=h[1], d=2, f=cube)
        ax = plt.subplot(2, 4, 2, projection='3d')
        ax.set_title("n={:d} h={:.2f}".format(n, h[1]))
        ax.plot_trisurf(X[:, 0], X[:, 1], Prob2, cmap='rainbow')

        Prob3 = Parzen(Data, X, h=h[2], d=2, f=cube)
        ax = plt.subplot(2, 4, 3, projection='3d')
        ax.set_title("n={:d} h={:.2f}".format(n, h[2]))
        ax.plot_trisurf(X[:, 0], X[:, 1], Prob3, cmap='rainbow')

        Prob3 = Parzen(Data, X, h=h[3], d=2, f=cube)
        ax = plt.subplot(2, 4, 4, projection='3d')
        ax.set_title("n={:d} h={:.2f}".format(n, h[3]))
        ax.plot_trisurf(X[:, 0], X[:, 1], Prob3, cmap='rainbow')



        Prob1 = Parzen(Data, X, h=h[0], d=2, f=guaasin_ND)
        ax = plt.subplot(2, 4, 5, projection='3d')
        ax.set_title("n={:d} h={:.2f}".format(n, h[0]))
        ax.plot_trisurf(X[:, 0], X[:, 1], Prob1, cmap='rainbow')
        ax.text2D(1, 5, "Parzen: Gaussian", rotation=90, va='center')

        Prob2 = Parzen(Data, X, h=h[1], d=2, f=guaasin_ND)
        ax = plt.subplot(2, 4, 6, projection='3d')
        ax.set_title("n={:d} h={:.2f}".format(n, h[1]))
        ax.plot_trisurf(X[:, 0], X[:, 1], Prob2, cmap='rainbow')

        Prob3 = Parzen(Data, X, h=h[2], d=2, f=guaasin_ND)
        ax = plt.subplot(2, 4, 7, projection='3d')
        ax.set_title("n={:d} h={:.2f}".format(n, h[2]))
        ax.plot_trisurf(X[:, 0], X[:, 1], Prob3, cmap='rainbow')

        Prob4 = Parzen(Data, X, h=h[3], d=2, f=guaasin_ND)
        ax = plt.subplot(2, 4, 8, projection='3d')
        ax.set_title("n={:d} h={:.2f}".format(n, h[3]))
        ax.plot_trisurf(X[:, 0], X[:, 1], Prob4, cmap='rainbow')


        plt.show()


def test_1D_pk(n=10000, humped=2):
    Data = np.ones(n)
    if humped==1:
        Data = np.random.normal(0, 1, n).reshape([-1, 1])
    elif humped==2:
        Data1 = np.random.normal(1, 1, n).reshape([-1, 1])
        Data2 = np.random.normal(8, 2, n).reshape([-1, 1])

        Data = 0.5*Data1 + 0.5*Data2


    X = np.arange(-5, 5, 0.1).reshape([-1, 1])
    Prob1 = Parzen(Data, X, h=0.01, d=1, f=cube)

    ax = plt.subplot(3, 3, 1)
    ax.set_title("h = 0.01")
    ax.plot(X, Prob1)
    plt.ylabel("Parzen: cube")

    Prob2 = Parzen(Data, X, h=0.1, d=1, f=cube)
    ax = plt.subplot(3, 3, 2)
    ax.set_title("h = 0.1")
    ax.plot(X, Prob2)

    Prob3 = Parzen(Data, X, h=1, d=1, f=cube)
    ax = plt.subplot(3, 3, 3)
    ax.set_title("h = 1")
    ax.plot(X, Prob3)

    Prob1 = Parzen(Data, X, h=0.01, d=1, f=gaussian_1D)
    ax = plt.subplot(3, 3, 4)
    ax.set_title("h = 0.01")
    ax.plot(X, Prob1)
    plt.ylabel("Parzen: gaussian")

    Prob2 = Parzen(Data, X, h=0.1, d=1, f=gaussian_1D)
    ax = plt.subplot(3, 3, 5)
    ax.set_title("h = 0.1")
    ax.plot(X, Prob2)

    Prob3 = Parzen(Data, X, h=1, d=1, f=gaussian_1D)
    ax = plt.subplot(3, 3, 6)
    ax.set_title("h = 1")
    ax.plot(X, Prob3)

    Prob1 = knn(Data, X, kn=10, d=1, f=eudistance)
    ax = plt.subplot(3, 3, 7)
    ax.set_title("kn = 10")
    ax.plot(X, Prob1)

    plt.ylabel("knn")

    Prob2 = knn(Data, X, kn=30, d=1, f=eudistance)
    ax = plt.subplot(3, 3, 8)
    ax.set_title("kn = 30")
    ax.plot(X, Prob2)

    Prob3 = knn(Data, X, kn=100, d=1, f=eudistance)
    ax = plt.subplot(3, 3, 9)
    ax.set_title("kn = 100")
    ax.plot(X, Prob3)
    plt.show()


def test_2D_pk():
    Data = np.random.randn(1000, 2)
    x = np.arange(-5, 5, 0.1)
    y = np.arange(-5, 5, 0.1)
    n = len(x)
    X = []
    for i in x:
        for j in y:
            X.append([i, j])
    X = np.array(X)
    # datax = np.hstack([np.random.randn(n) * 2 - 3,
    #                    np.random.randn(n) * 2 + 5])
    # datay = np.hstack([np.random.randn(n) * 6 + 4,
    #                    np.random.randn
    #                    (n) * 5 - 4])
    # xi = np.array([1, 4])
    # xv, yv = datax, datay
    # pos = np.vstack([datax, datay])

    Prob1 = Parzen(Data, X, h=0.02, d=2, f=cube)
    ax = plt.subplot(2, 3, 1, projection='3d')
    ax.set_title("Parzen: h = 0.02")
    ax.plot_trisurf(X[:, 0], X[:, 1], Prob1, cmap='rainbow')

    Prob2 = Parzen(Data, X, h=0.2, d=2, f=cube)
    ax = plt.subplot(2, 3, 2, projection='3d')
    ax.set_title("Parzen: h = 0.2")
    ax.plot_trisurf(X[:, 0], X[:, 1], Prob2, cmap='rainbow')

    Prob3 = Parzen(Data, X, h=2, d=2, f=cube)
    ax = plt.subplot(2, 3, 3, projection='3d')
    ax.set_title("Parzen: h = 2")
    ax.plot_trisurf(X[:, 0], X[:, 1], Prob3, cmap='rainbow')

    Prob1 = knn(Data, X, kn=10, d=2, f=eudistance)
    ax = plt.subplot(2, 3, 4, projection='3d')
    ax.set_title("knn: kn = 10")
    ax.plot_trisurf(X[:, 0], X[:, 1], Prob1, cmap='rainbow')

    Prob2 = knn(Data, X, kn=40, d=2, f=eudistance)
    ax = plt.subplot(2, 3, 5, projection='3d')
    ax.set_title("knn: kn = 40")
    ax.plot_trisurf(X[:, 0], X[:, 1], Prob2, cmap='rainbow')

    Prob3 = knn(Data, X, kn=100, d=2, f=eudistance)
    ax = plt.subplot(2, 3, 6, projection='3d')
    ax.set_title("knn: kn = 100")
    ax.plot_trisurf(X[:, 0], X[:, 1], Prob3, cmap='rainbow')
    plt.show()


def main():
    test_1D_pk()
    test_2D_pk()


if __name__ == '__main__':
    test_2D_parzen()
