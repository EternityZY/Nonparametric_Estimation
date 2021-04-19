import numpy as np
import matplotlib.pyplot as plt


class TwoNomal():
    def __init__(self, mu1, mu2, sigma1, sigma2):
        self.mu1 = mu1
        self.sigma1 = sigma1
        self.mu2 = mu2
        self.sigma2 = sigma2

    def doubledensity(self, x):
        mu1 = self.mu1
        sigma1 = self.sigma1
        mu2 = self.mu2
        sigma2 = self.sigma1
        N1 = np.sqrt(2 * np.pi * np.power(sigma1, 2))
        fac1 = np.power(x - mu1, 2) / np.power(sigma1, 2)
        density1 = np.exp(-fac1 / 2) / N1

        N2 = np.sqrt(2 * np.pi * np.power(sigma2, 2))
        fac2 = np.power(x - mu2, 2) / np.power(sigma2, 2)
        density2 = np.exp(-fac2 / 2) / N2
        # print(density1,density2)
        density = 0.5 * density2 + 0.5 * density1
        return density


if __name__ == '__main__':
    N2 = TwoNomal(-2, 1, 1, 1)

    # 创建等差数列作为X
    X = np.arange(-5, 5, 0.05)
    # print(X)
    Y = N2.doubledensity(X)
    plt.plot(X,Y,'b-',linewidth=3)

    plt.show()