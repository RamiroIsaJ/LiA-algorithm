import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import math


class BenchFunctions:
    def __init__(self, dim=2, size_max=100):
        if dim < 2:
            self.dimension = 2
        elif dim > 3:
            self.dimension = 3
        else:
            self.dimension = dim
        self.size_max = size_max

    def mesh_G(self, low, up):
        temp = [np.linspace(low, up, self.size_max) for _ in range(self.dimension)]
        grid = np.meshgrid(*temp)
        assert (len(grid) == self.dimension)
        return np.array(grid)

    def function_0(self):
        # ackley function
        low_r, up_r = -5.12,  5.12
        space = np.linspace(low_r, up_r, self.size_max)
        grid = self.mesh_G(low_r, up_r)
        if self.dimension == 2:
            f1 = np.zeros((self.size_max, self.size_max))
            f2 = np.zeros((self.size_max, self.size_max))
        else:
            f1 = np.zeros((self.size_max, self.size_max, self.size_max))
            f2 = np.zeros((self.size_max, self.size_max, self.size_max))

        for i in range(self.dimension):
            f1 += np.power(grid[i], 2)
            f2 += np.cos(2 * math.pi * grid[i])

        f = -20 * np.exp(-0.2 * np.sqrt((1/self.dimension)*f1)) - np.exp((1/self.dimension)*f2) + 20 + np.finfo(float).eps
        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        # Axes3D.plot_surface(ax, grid[0], grid[1], f, rstride=1, cstride=1, cmap=cm.coolwarm, edgecolor='none')
        return f, space

    def function_1(self):
        # sphere function
        low_r, up_r = -100,  100
        space = np.linspace(low_r, up_r, self.size_max)
        grid = self.mesh_G(low_r, up_r)
        if self.dimension == 2:
            f1 = np.zeros((self.size_max, self.size_max))
        else:
            f1 = np.zeros((self.size_max, self.size_max, self.size_max))

        for i in range(self.dimension):
            f1 += np.power(grid[i], 2)

        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        # Axes3D.plot_surface(ax, grid[0], grid[1], f1, rstride=1, cstride=1, cmap=cm.coolwarm, edgecolor='none')
        return f1, space

    def function_2(self):
        # rosenbrock function
        low_r, up_r = -6, 6
        space = np.linspace(low_r, up_r, self.size_max)
        grid = self.mesh_G(low_r, up_r)
        if self.dimension == 2:
            f1 = np.zeros((self.size_max, self.size_max))
        else:
            f1 = np.zeros((self.size_max, self.size_max, self.size_max))

        for i in range(self.dimension-1):
            f1 += 100*np.power(grid[i+1]-np.power(grid[i], 2), 2) + np.power(1-grid[i], 2)

        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        # Axes3D.plot_surface(ax, grid[0], grid[1], f1, rstride=1, cstride=1, cmap=cm.coolwarm, edgecolor='none')
        return f1, space

    def function_3(self):
        # rastrigin function
        low_r, up_r = -5.12, 5.12
        space = np.linspace(low_r, up_r, self.size_max)
        grid = self.mesh_G(low_r, up_r)
        if self.dimension == 2:
            f1 = np.zeros((self.size_max, self.size_max))
        else:
            f1 = np.zeros((self.size_max, self.size_max, self.size_max))

        for i in range(self.dimension):
            f1 += np.power(grid[i], 2) - 10 * np.cos(2 * math.pi * grid[i]) + 10
        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        # Axes3D.plot_surface(ax, grid[0], grid[1], f1, rstride=1, cstride=1, cmap=cm.coolwarm, edgecolor='none')
        return f1, space

    def function_4(self):
        # cigar function
        low_r, up_r = -100, 100
        space = np.linspace(low_r, up_r, self.size_max)
        grid = self.mesh_G(low_r, up_r)
        if self.dimension == 2:
            f1 = np.zeros((self.size_max, self.size_max))
        else:
            f1 = np.zeros((self.size_max, self.size_max, self.size_max))

        for i in range(self.dimension):
            if i == 0:
                f1 += np.power(grid[i], 2)
            else:
                f1 += np.power(10, 6) * np.power(grid[i], 2)
        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        # Axes3D.plot_surface(ax, grid[0], grid[1], f1, rstride=1, cstride=1, cmap=cm.coolwarm, edgecolor='none')
        return f1, space

    def function_5(self):
        # griewank function
        low_r, up_r = -600, 600
        space = np.linspace(low_r, up_r, self.size_max)
        grid = self.mesh_G(low_r, up_r)
        if self.dimension == 2:
            f1 = np.zeros((self.size_max, self.size_max))
            f2 = np.ones((self.size_max, self.size_max))
        else:
            f1 = np.zeros((self.size_max, self.size_max, self.size_max))
            f2 = np.ones((self.size_max, self.size_max, self.size_max))

        for i in range(self.dimension):
            f1 += np.power(grid[i], 2)/4000
            f2 *= np.cos(grid[i]/np.sqrt(i+1))

        f = 1 + f1 - f2
        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        # Axes3D.plot_surface(ax, grid[0], grid[1], f, rstride=1, cstride=1, cmap=cm.coolwarm, edgecolor='none')
        return f, space

    def function_6(self):
        # schwefel function
        low_r, up_r = -500, 500
        space = np.linspace(low_r, up_r, self.size_max)
        grid = self.mesh_G(low_r, up_r)
        if self.dimension == 2:
            f1 = np.zeros((self.size_max, self.size_max))
        else:
            f1 = np.zeros((self.size_max, self.size_max, self.size_max))

        for i in range(self.dimension):
            f1 += grid[i] * np.sin(np.sqrt(np.abs(grid[i])))

        f = 418.9829*self.dimension - f1
        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        # Axes3D.plot_surface(ax, grid[0], grid[1], f, rstride=1, cstride=1, cmap=cm.coolwarm, edgecolor='none')
        return f, space

    def function_7(self):
        # drop wave function
        low_r, up_r = -5.12, 5.12
        space = np.linspace(low_r, up_r, self.size_max)
        grid = self.mesh_G(low_r, up_r)
        if self.dimension == 2:
            f1 = np.zeros((self.size_max, self.size_max))
        else:
            f1 = np.zeros((self.size_max, self.size_max, self.size_max))

        for i in range(self.dimension):
            f1 += np.power(grid[i], 2)

        f = -((1+np.cos(12*np.sqrt(f1))) / ((1/self.dimension)*f1+2))
        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        # Axes3D.plot_surface(ax, grid[0], grid[1], f, rstride=1, cstride=1, cmap=cm.coolwarm, edgecolor='none')
        return f, space

    def function_8(self):
        # levy function
        low_r, up_r = -10, 10
        space = np.linspace(low_r, up_r, self.size_max)
        grid = self.mesh_G(low_r, up_r)
        if self.dimension == 2:
            f = np.zeros((self.size_max, self.size_max))
        else:
            f = np.zeros((self.size_max, self.size_max, self.size_max))

        w1 = np.array(1 + ((grid[0] - 1) / 4))
        wd = np.array(1 + ((grid[self.dimension-1] - 1) / 4))
        for i in range(self.dimension-1):
            w = np.array(1 + ((grid[i] - 1) / 4))
            f += (np.power(w-1, 2))*(1+10*np.power(np.sin(math.pi*w+1), 2)) + \
                 (np.power(wd-1, 2))*(1+10*np.power(np.sin(math.pi*wd+1), 2))
        f += np.power(np.sin(math.pi*w1), 2)
        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        # Axes3D.plot_surface(ax, grid[0], grid[1], f, rstride=1, cstride=1, cmap=cm.coolwarm, edgecolor='none')
        return f, space

    def function_9(self):
        # lang function
        low_r, up_r = 0, 10
        space = np.linspace(low_r, up_r, self.size_max)
        grid = self.mesh_G(low_r, up_r)
        if self.dimension == 2:
            f = np.zeros((self.size_max, self.size_max))
        else:
            f = np.zeros((self.size_max, self.size_max, self.size_max))

        A = np.array([[3, 5], [5, 2], [2, 1], [1, 4], [7, 9]])
        c = np.array([1, 2, 5, 2, 3])

        for i in range(5):
            if self.dimension == 2:
                w = np.zeros((self.size_max, self.size_max))
            else:
                w = np.zeros((self.size_max, self.size_max, self.size_max))

            for j in range(self.dimension):
                w += np.power(grid[j] - A[i, j], 2)

            f += c[i] * np.exp((-1/math.pi)*w) * np.cos(math.pi*w)

        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        # Axes3D.plot_surface(ax, grid[0], grid[1], f, rstride=1, cstride=1, cmap=cm.coolwarm, edgecolor='none')
        return f, space

    def function_10(self):
        # Himmelblau
        low_r, up_r = -5, 5
        space = np.linspace(low_r, up_r, self.size_max)
        grid = self.mesh_G(low_r, up_r)
        f1 = np.zeros((self.size_max, self.size_max))
        f2 = np.zeros((self.size_max, self.size_max))

        f1 = np.power(np.power(grid[0], 2) + grid[1] - 11, 2)
        f2 = np.power(np.power(grid[1], 2) + grid[0] - 7, 2)

        f = f1 + f2
        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        # Axes3D.plot_surface(ax, grid[0], grid[1], f, rstride=1, cstride=1, cmap=cm.coolwarm, edgecolor='none')
        return f, space

    def function_11(self):
        # Beale
        low_r, up_r = -4.5, 4.5
        space = np.linspace(low_r, up_r, self.size_max)
        grid = self.mesh_G(low_r, up_r)

        f1 = np.power(1.5 - grid[0] + grid[0]*grid[1], 2)
        f2 = np.power(2.25 - grid[0] + grid[0]*np.power(grid[1], 2), 2)
        f3 = np.power(2.625 - grid[0] + grid[0]*np.power(grid[1], 3), 2)

        f = f1 + f2 + f3
        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        # Axes3D.plot_surface(ax, grid[0], grid[1], f, rstride=1, cstride=1, cmap=cm.coolwarm, edgecolor='none')
        return f, space


if __name__ == '__main__':
    env = BenchFunctions()
    env.function_11()
    # Z = env.lang_f()
    # plt.figure()
    # plt.imshow(Z, cmap=cm.coolwarm, vmin=abs(Z).min(), vmax=abs(Z).max())
    # plt.show()
