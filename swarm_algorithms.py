import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import math
import operator
import random
from numpy.random import uniform


class PSO:
    def __init__(self, n_part=20, dim=2, wi=0.5, wf=2.0, c1=2.0, c2=2.0, d_max=5, itr=50, size_t=100, func=None):
        self.n_part = n_part
        self.dim = dim
        self.wi = wi
        self.wf = wf
        self.d_max = d_max
        self.c1 = c1
        self.c2 = c2
        self.itr = itr
        self.part = np.zeros((self.n_part, self.dim+1))
        self.velo = np.zeros((self.n_part, self.dim))
        self.size_t = size_t
        self.v_min = 0
        self.v_max = self.size_t-1
        if func is None:
            self.func = np.empty((0, 0))
        else:
            self.func = func

    def mesh_G(self, low, up):
        temp = [np.linspace(low, up, self.size_t) for _ in range(self.dim)]
        grid = np.meshgrid(*temp)
        assert (len(grid) == self.dim)
        return grid

    def sphere_f(self):
        low_r, up_r = -100,  100
        grid = self.mesh_G(low_r, up_r)
        f = np.zeros((self.size_t, self.size_t))
        for i in range(self.dim):
            f += np.power(grid[i], 2)
        self.func = f

    def ackeley_f(self):
        low_r, up_r = -5.12,  5.12
        grid = self.mesh_G(low_r, up_r)
        f1 = np.zeros((self.size_t, self.size_t))
        f2 = np.zeros((self.size_t, self.size_t))
        for i in range(self.dim):
            f1 += np.power(np.array(grid[i]), 2)
            f2 += np.cos(2 * math.pi * np.array(grid[i]))
        self.func = -20 * np.exp(-0.2 * np.sqrt(0.5 * f1)) - np.exp(0.5 * f2) + 20 + np.finfo(float).eps

    def rand_ini(self):
        for i in range(self.n_part):
            p = np.zeros((1, self.dim), dtype=int)
            for d in range(self.dim):
                p[0, d] = np.round(self.v_min + random.random() * (self.v_max - self.v_min))
            self.part[i, :self.dim] = p[0, :]
            self.part[i, self.dim] = self.func[tuple(p[0, :])]

    def pso_min(self):
        pos_min, g_best = min(enumerate(self.part[:, self.dim]), key=operator.itemgetter(1))
        loc_best = self.part
        par_best = self.part[pos_min, :]
        fit_r = [g_best]
        # plt.figure()
        # plt.ion()
        for itr in range(self.itr):
            # graph function
            # plt.clf()
            # plt.imshow(self.func, cmap=cm.coolwarm, vmin=abs(self.func).min(), vmax=abs(self.func).max())

            w = self.wf - (self.wf - self.wi) * (itr / self.itr)
            r1 = random.random()
            r2 = random.random()

            for i in range(self.n_part):

                for d in range(self.dim):
                    self.velo[i, d] = w * self.velo[i, d] + self.c1 * (r1 * (loc_best[i, d] - self.part[i, d])) + \
                                      self.c2 * (r2 * (par_best[d] - self.part[i, d]))
                    self.velo[i, d] = ((self.velo[i, d] <= -self.d_max) * -self.d_max) + \
                                      ((self.velo[i, d] > -self.d_max) * self.velo[i, d])
                    self.velo[i, d] = ((self.velo[i, d] >= self.d_max) * self.d_max) + \
                                      ((self.velo[i, d] < self.d_max) * self.velo[i, d])

                part = np.zeros((1, self.dim), dtype=int)
                for d in range(self.dim):
                    part[0, d] = np.round(self.part[i, d] + self.velo[i, d])
                    # control limits
                    part[0, d] = ((part[0, d] <= self.v_min) * self.v_min) + \
                                 ((part[0, d] > self.v_min) * part[0, d])
                    part[0, d] = ((part[0, d] >= self.v_max) * self.v_max) + \
                                 ((part[0, d] < self.v_max) * part[0, d])
                self.part[i, :self.dim] = part[0, :]
                self.part[i, self.dim] = self.func[tuple(part[0, :])]

            for i in range(self.n_part):
                if self.part[i, self.dim] < loc_best[i, self.dim]:
                    loc_best[i, :] = self.part[i, :]

            # min value in this iteration
            pos_min, val_min = min(enumerate(self.part[:, self.dim]), key=operator.itemgetter(1))
            if val_min < g_best:
                g_best = val_min
                par_best[:] = self.part[pos_min, :]
            fit_r.append(g_best)
            # print(par_best)
            # plt.scatter(self.part[:, 0], self.part[:, 1], marker='o', c='b', s=5)
            # plt.scatter(par_best[0], par_best[1], marker='*', c='r', s=5)
            # plt.pause(0.1)
        # plt.ioff()
        # plt.show()
        return fit_r


class FWA:
    def __init__(self, n_fw=20, dim=2, m=30, a=0.04, b=0.8, a_s=40, m_s=2, itr=50, size_t=100, func=None):
        self.n_fw = n_fw
        self.dim = dim
        self.fw = np.zeros((self.n_fw, self.dim+1))
        self.m = m
        self.a = a
        self.b = b
        self.a_s = a_s
        self.m_s = m_s
        self.itr = itr
        self.size_t = size_t
        self.v_min = 0
        self.v_max = self.size_t-1
        if func is None:
            self.func = np.empty((0, 0))
        else:
            self.func = func

    def mesh_G(self, low, up):
        temp = [np.linspace(low, up, self.size_t) for _ in range(self.dim)]
        grid = np.meshgrid(*temp)
        assert (len(grid) == self.dim)
        return grid

    def sphere_f(self):
        low_r, up_r = -100, 100
        grid = self.mesh_G(low_r, up_r)
        f = np.zeros((self.size_t, self.size_t))
        for i in range(self.dim):
            f += np.power(grid[i], 2)
        self.func = f

    def ackeley_f(self):
        low_r, up_r = -5.12, 5.12
        grid = self.mesh_G(low_r, up_r)
        f1 = np.zeros((self.size_t, self.size_t))
        f2 = np.zeros((self.size_t, self.size_t))
        for i in range(self.dim):
            f1 += np.power(np.array(grid[i]), 2)
            f2 += np.cos(2 * math.pi * np.array(grid[i]))
        self.func = -20 * np.exp(-0.2 * np.sqrt(0.5 * f1)) - np.exp(0.5 * f2) + 20 + np.finfo(float).eps

    def rand_ini(self):
        for i in range(self.n_fw):
            p = np.zeros((1, self.dim), dtype=int)
            for d in range(self.dim):
                p[0, d] = np.round(self.v_min + random.random() * (self.v_max - self.v_min))
            self.fw[i, :self.dim] = p[0, :]
            self.fw[i, self.dim] = self.func[tuple(p[0, :])]

    def fwa_min(self):
        pos_min, _ = min(enumerate(self.fw[:, self.dim]), key=operator.itemgetter(1))
        fw_best = self.fw[pos_min, :]
        fit_r = [fw_best[self.dim]]
        # plt.figure()
        # plt.ion()
        for itr in range(self.itr):
            # graph function
            # plt.clf()
            # plt.imshow(self.func, cmap=cm.coolwarm, vmin=abs(self.func).min(), vmax=abs(self.func).max())
            # plt.scatter(self.fw[:, 0], self.fw[:, 1], marker='o', c='b', s=5)

            max_pos, y_max = max(enumerate(self.fw[:, self.dim]), key=operator.itemgetter(1))
            min_pos, y_min = min(enumerate(self.fw[:, self.dim]), key=operator.itemgetter(1))

            # explosion width sparks
            S = self.m * ((y_max - self.fw[:, self.dim] + np.finfo(float).eps) /
                          (sum(y_max - self.fw[:, self.dim]) + np.finfo(float).eps))
            A = self.a_s * ((self.fw[:, self.dim] - y_min + np.finfo(float).eps) /
                            (sum(self.fw[0:, 1] - y_min) + np.finfo(float).eps))

            # sparks limits to avoid worth effects
            for i in range(self.n_fw):
                if S[i] < (self.a * self.m):
                    S[i] = round(self.a * self.m)
                elif S[i] > (self.b * self.m) and (self.a < self.b < 1):
                    S[i] = round(self.b * self.m)
                else:
                    S[i] = round(S[i])

            # min explosion
            A_min = self.v_min - (((self.v_min - self.v_max) / self.itr) * math.sqrt((2 * self.itr - itr) * itr))

            # generate uniform sparks
            for i in range(self.n_fw):
                x_sp = self.fw[i, :]
                # min distance
                if A[i] < A_min:
                    A[i] = A_min
                # dimension generation
                lon = int(S[i])
                z = np.random.randint(2, size=lon)
                for k in range(lon):
                    if z[k] == 1:
                        fw = np.zeros((1, self.dim), dtype=int)
                        for d in range(self.dim):
                            h = A[i] * random.uniform(-1, 1)
                            fw[0, d] = np.round(self.fw[i, d] + h)
                            # control limits
                            fw[0, d] = ((fw[0, d] <= self.v_min) * self.v_min) + \
                                       ((fw[0, d] > self.v_min) * fw[0, d])
                            fw[0, d] = ((fw[0, d] >= self.v_max) * self.v_max) + \
                                       ((fw[0, d] < self.v_max) * fw[0, d])
                        x_sp[:self.dim] = fw[0, :]
                        x_sp[self.dim] = self.func[tuple(fw[0, :])]
                        self.fw = np.concatenate((self.fw, x_sp[np.newaxis, :]), axis=0)

            # plt.scatter(self.fw[self.n_fw:, 0], self.fw[self.n_fw:, 1], marker='o', c='y', s=5)
            # n = self.fw.__len__()

            # generation gaussian sparks
            for i in range(self.m_s):
                pos = random.randint(0, self.n_fw-1)
                g_sp = self.fw[pos, :]
                lon = int(S[pos])
                z = np.random.randint(2, size=lon)
                g = np.random.randn()
                for k in range(0, lon):
                    if z[k] == 1:
                        fw = np.zeros((1, self.dim), dtype=int)
                        for d in range(self.dim):
                            fw[0, d] = np.round(g_sp[d] * g)
                            # control limits
                            fw[0, d] = ((fw[0, d] <= self.v_min) * self.v_min) + \
                                       ((fw[0, d] > self.v_min) * fw[0, d])
                            fw[0, d] = ((fw[0, d] >= self.v_max) * self.v_max) + \
                                       ((fw[0, d] < self.v_max) * fw[0, d])
                        g_sp[:self.dim] = fw[0, :]
                        g_sp[self.dim] = self.func[tuple(fw[0, :])]
                        self.fw = np.concatenate((self.fw, g_sp[np.newaxis, :]), axis=0)

            # plt.scatter(self.fw[n:, 0], self.fw[n:, 1], marker='o', c='g', s=5)

            self.fw = np.array(sorted(self.fw, key=lambda a_entry: a_entry[self.dim], reverse=False))
            self.fw = self.fw[:self.n_fw, :]
            best_fw = self.fw[0, :]
            if best_fw[self.dim] < fw_best[self.dim]:
                fw_best[:] = best_fw[:]
            fit_r.append(fw_best[self.dim])
            # print(fw_best)
            # plt.scatter(fw_best[0], fw_best[1], marker='*', c='r', s=5)
            # plt.pause(0.5)
        # plt.ioff()
        # plt.show()
        return fit_r


class BEE:
    def __init__(self, bee_t=20, dim=2, k=200, limit=30, itr=50, size_t=100, func=None):
        self.bee_t = bee_t
        self.dim = dim
        self.bees = np.zeros((self.bee_t, self.dim+1))
        self.k = k
        self.limit = limit
        self.itr = itr
        self.size_t = size_t
        self.v_min = 0
        self.v_max = self.size_t-1
        if func is None:
            self.func = np.empty((0, 0))
        else:
            self.func = func

    def mesh_G(self, low, up):
        temp = [np.linspace(low, up, self.size_t) for _ in range(self.dim)]
        grid = np.meshgrid(*temp)
        assert (len(grid) == self.dim)
        return grid

    def sphere_f(self):
        low_r, up_r = -100, 100
        grid = self.mesh_G(low_r, up_r)
        f = np.zeros((self.size_t, self.size_t))
        for i in range(self.dim):
            f += np.power(grid[i], 2)
        self.func = f

    def ackeley_f(self):
        low_r, up_r = -5.12, 5.12
        grid = self.mesh_G(low_r, up_r)
        f1 = np.zeros((self.size_t, self.size_t))
        f2 = np.zeros((self.size_t, self.size_t))
        for i in range(self.dim):
            f1 += np.power(np.array(grid[i]), 2)
            f2 += np.cos(2 * math.pi * np.array(grid[i]))
        self.func = -20 * np.exp(-0.2 * np.sqrt(0.5 * f1)) - np.exp(0.5 * f2) + 20 + np.finfo(float).eps

    def chaotic_ini(self):
        p = np.zeros((self.bee_t, self.dim), dtype=int)
        op = np.zeros((self.bee_t, self.dim), dtype=int)
        ch_k = np.zeros((self.k, self.dim))
        for i in range(self.bee_t):
            for j in range(self.dim):
                ch_k[0, j] = uniform(0, 1)
                for k in range(1, self.k):
                    ch_k[k, j] = np.sin(math.pi * ch_k[k-1, j])

                p[i, j] = np.round(self.v_min + ch_k[self.k-1, j] * (self.v_max - self.v_min))

        for i in range(self.bee_t):
            for j in range(self.dim):
                op[i, j] = self.v_min + self.v_max - p[i, j]

        bees_p = np.concatenate((p, op), axis=0)
        perm_pos = np.random.permutation(len(bees_p))

        for i in range(self.bee_t):
            self.bees[i, :self.dim] = bees_p[perm_pos[i], :]
            self.bees[i, self.dim] = self.func[tuple(bees_p[perm_pos[i], :])]

    def bee_min(self):
        trial = np.zeros((self.bee_t, 1))
        fit = np.zeros((self.bee_t, 1))
        prob = np.zeros((self.bee_t, 1))
        pos_min, _ = min(enumerate(self.bees[:, self.dim]), key=operator.itemgetter(1))
        bee_best = self.bees[pos_min, :]
        fit_r = [bee_best[self.dim]]

        # plt.figure()
        # plt.ion()
        for itr in range(self.itr):
            # graph function
            # plt.clf()
            # plt.imshow(self.func, cmap=cm.coolwarm, vmin=abs(self.func).min(), vmax=abs(self.func).max())
            # plt.scatter(self.bees[:, 0], self.bees[:, 1], marker='*', c='b', s=5)

            for i in range(self.bee_t):
                pos1 = i
                while True:
                    if pos1 == i:
                        pos1 = random.randint(0, self.bee_t-1)
                    else:
                        break
                pos2 = i
                while True:
                    if pos2 == pos1 or pos2 == i:
                        pos2 = random.randint(0, self.bee_t-1)
                    else:
                        break

                br1 = self.bees[pos1, :]
                br2 = self.bees[pos2, :]

                j = random.randrange(0, self.dim)
                V = np.zeros((1, self.dim), dtype=int)
                V[0, :] = self.bees[i, :self.dim]
                val = self.func[tuple(V[0, :])]
                V[0, j] = np.round(bee_best[j] + random.uniform(-1, 1) * (br1[j] - br2[j]))
                # control limits
                for d1 in range(self.dim):
                    V[0, d1] = ((V[0, d1] <= self.v_min) * self.v_min) + ((V[0, d1] > self.v_min) * V[0, d1])
                    V[0, d1] = ((V[0, d1] >= self.v_max) * self.v_max) + ((V[0, d1] < self.v_max) * V[0, d1])
                val_n = self.func[tuple(V[0, :])]
                if val_n <= val:
                    self.bees[i, :self.dim] = V[0, :]
                    self.bees[i, self.dim] = val_n
                    trial[i] = 1
                else:
                    trial[i] += 1

                if self.bees[i, self.dim] > 0:
                    fit[i] = 1 / (1 + self.bees[i, self.dim])
                else:
                    fit[i] = 1 + abs(self.bees[i, self.dim])

            # compute probabilities
            for i in range(self.bee_t):
                prob[i] = fit[i] / np.sum(fit)

            for i in range(self.bee_t):
                r = random.random()
                if r < prob[i]:
                    pos1 = i
                    while True:
                        if pos1 == i:
                            pos1 = random.randint(0, self.bee_t-1)
                        else:
                            break
                    pos2 = i
                    while True:
                        if pos2 == pos1 or pos2 == i:
                            pos2 = random.randint(0, self.bee_t-1)
                        else:
                            break

                    br1 = self.bees[pos1, :]
                    br2 = self.bees[pos2, :]

                    j = random.randrange(0, self.dim)
                    V = np.zeros((1, self.dim), dtype=int)
                    V[0, :] = self.bees[i, :self.dim]
                    val = self.func[tuple(V[0, :])]
                    V[0, j] = np.round(bee_best[j] + random.uniform(-1, 1) * (br1[j] - br2[j]))
                    # control limits
                    for d1 in range(self.dim):
                        V[0, d1] = ((V[0, d1] <= self.v_min) * self.v_min) + ((V[0, d1] > self.v_min) * V[0, d1])
                        V[0, d1] = ((V[0, d1] >= self.v_max) * self.v_max) + ((V[0, d1] < self.v_max) * V[0, d1])
                    val_n = self.func[tuple(V[0, :])]

                    if val_n <= val:
                        self.bees[i, :self.dim] = V[0, :]
                        self.bees[i, self.dim] = val_n
                        trial[i] = 1
                    else:
                        trial[i] += 1
            _, t_max = max(enumerate(trial), key=operator.itemgetter(1))

            if t_max > self.limit:
                self.chaotic_ini()
                print('New bees generation')

            # best bee
            min_pos, val_b = min(enumerate(self.bees[:, self.dim]), key=operator.itemgetter(1))
            if val_b < bee_best[self.dim]:
                bee_best = self.bees[min_pos, :]
            fit_r.append(bee_best[self.dim])
            # print(bee_best)
            # plt.scatter(bee_best[0], bee_best[1], marker='o', c='r', s=10)
            # plt.pause(1)
        # plt.ioff()
        # plt.show()
        return fit_r


if __name__ == "__main__":
    env = BEE()
    env.ackeley_f()
    env.chaotic_ini()
    env.bee_min()
