import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import math
import operator
import random
from copy import deepcopy
from benchmark_functions import BenchFunctions as Func


class LIA:
    def __init__(self, nt=20, dim=2, k=100, max_p=10, min_p=4, min_step=2.0, max_step=4.0, itr=100, size_t=100,
                 func=None, tar_m=None, ini='C', spa=None):
        self.nt = nt
        self.dim = dim
        self.k = k
        self.max_p = max_p
        self.min_p = min_p
        self.min_step = max_step
        self.max_step = min_step
        self.itr = itr
        self.size_t = size_t
        self.b_min = 1
        self.b_max = self.size_t - 1
        self.light = np.zeros((self.nt, self.dim + 1))
        self.loc_best = np.ones((self.nt, self.dim + 1)) * np.inf
        self.loc_max_b = np.ones((self.nt, self.dim + 1))
        self.tar_m = tar_m
        self.ini = ini
        self.space = spa

        if func is None:
            self.ackeley_f()
        else:
            self.func = func

    def mesh_G(self, low, up):
        temp = [np.linspace(low, up, self.size_t) for _ in range(self.dim)]
        grid = np.meshgrid(*temp)
        assert (len(grid) == self.dim)
        return grid

    def ackeley_f(self):
        low_r, up_r = -5.12,  5.12
        grid = self.mesh_G(low_r, up_r)
        f1 = np.zeros((self.size_t, self.size_t))
        f2 = np.zeros((self.size_t, self.size_t))
        for i in range(self.dim):
            f1 += np.power(np.array(grid[i]), 2)
            f2 += np.cos(2 * math.pi * np.array(grid[i]))
        self.func = -20 * np.exp(-0.2 * np.sqrt(0.5 * f1)) - np.exp(0.5 * f2) + 20 + np.finfo(float).eps

    def rand_init(self):
        for i in range(self.nt):
            p = np.zeros((1, self.dim), dtype=int)
            for d in range(self.dim):
                p[0, d] = np.round(self.b_min + np.random.random() * (self.b_max - self.b_min))
            self.light[i, :self.dim] = p[0, :]
            self.light[i, self.dim] = self.func[tuple(p[0, :])]

    def flashes_init(self, rep=20, prob=0.5):
        light_s = np.round(self.b_min + np.random.rand(self.nt, self.dim) * (self.b_max - self.b_min))
        # --------------------------------------------------------------------------------------------------
        plt.figure()
        plt.imshow(self.func, cmap=cm.coolwarm, vmin=abs(self.func).min(), vmax=abs(self.func).max())
        plt.scatter(light_s[:, 1], light_s[:, 0], marker='o', c='r', s=5)
        plt.pause(0.01)
        plt.ion()
        # --------------------------------------------------------------------------------------------------
        for i in range(rep - 1):
            # ----------------------------------------------------------------------------------------------
            plt.clf()
            plt.imshow(self.func, cmap=cm.coolwarm, vmin=abs(self.func).min(), vmax=abs(self.func).max())
            # ----------------------------------------------------------------------------------------------
            light = np.round(self.b_min + np.random.rand(self.nt, self.dim) * (self.b_max - self.b_min))
            light_s = np.vstack([light_s, light])
            # -----------------------------------------------------------------------------------------------
            plt.scatter(light[:, 1], light[:, 0], marker='o', c='r', s=5)
            plt.pause(0.01)
        plt.ioff()
        # ---------------------------------------------------------------------------------------------------
        light_s = np.array(light_s.astype(int))
        f_light_s = np.array([self.func[tuple(x)] for x in light_s])
        idx_max = np.argmax(f_light_s) if len(f_light_s) else None
        idx_min = np.argmin(f_light_s) if len(f_light_s) else None
        x_min, f_max, f_min = light_s[idx_min], f_light_s[idx_max], f_light_s[idx_min]
        # --------------------------------------------------------------------
        # normalization fitness function
        # --------------------------------------------------------------------
        fit_n = (f_light_s - f_min) / (f_max - f_min)
        p_prob = []
        while len(p_prob) < self.nt * self.nt:
            p_prob = np.where(fit_n < prob)[0].tolist()
            prob += 0.05
        # --------------------------------------------------------------------
        p_prob = np.random.permutation(p_prob)
        p_prob = p_prob[:self.nt * self.nt]
        n_rep = 20 * self.dim
        val_ref = 0
        light_f = 0
        for i1 in range(n_rep):
            data, div_, prob_sum = [], [], []
            pos_f = np.random.permutation(p_prob).reshape((self.nt, self.nt))
            for j in range(self.nt):
                light_f1 = x_min
                for i in range(0, self.nt - 1):
                    light_f1 = np.vstack([light_f1, light_s[pos_f[j, i]]])

                prob_sum.append(np.sum(fit_n[pos_f[j, :]]))
                data.append(light_f1)
                center = np.sum(light_f1, axis=0) / self.nt
                diversity = (1 / self.nt) * np.sum(np.sqrt((1 / self.dim) * np.sum(np.power(light_f1 - center, 2))))
                div_.append(diversity)

            data, div_, prob_sum = np.array(data), np.array(div_), np.array(prob_sum)
            idx_div = np.argmax(div_)
            val_div = div_[idx_div]
            if val_div > val_ref and prob_sum[idx_div] / self.nt <= prob:
                light_f = np.array(data[idx_div, :, :])
                val_ref = val_div

        light_f = light_f.astype(int)
        for i in range(self.nt):
            self.light[i, :self.dim] = light_f[i, :]
            self.light[i, self.dim] = self.func[tuple(light_f[i, :])]

        # ---------------------------------------------------------------------------------------------------
        plt.figure()
        plt.imshow(self.func, cmap=cm.coolwarm, vmin=abs(self.func).min(), vmax=abs(self.func).max())
        plt.scatter(self.light[:, 1], self.light[:, 0], marker='o', c='b', s=5)
        plt.show()

        # ---------------------------------------------------------------------------------------------------

    def branches_min(self, light, n_points, x_best_, step, fc=1):
        n_points = np.abs(n_points)
        tot_light = np.zeros((n_points, self.dim+1))
        # ----------------------------------------
        # direction for main branch
        # ----------------------------------------
        dir_ = np.ones(self.dim)
        for d in range(self.dim):
            diff = (x_best_[d] - light[d])
            if diff != 0:
                dir_[d] = diff / np.abs(diff)
            else:
                if np.random.rand() > 0.5:
                    dir_[d] *= -1
        # --------------------------------------------------------------------------------------------------
        # main branch
        # --------------------------------------------------------------------------------------------------
        tot_light[0, :] = light

        if np.random.rand() > 0.70:
            for i in range(1, n_points):
                rad = step * np.random.randint(-1, 2, self.dim)
                aux = tot_light[i-1, :self.dim] + rad
                # ----------------------------------
                # mutation
                # ----------------------------------
                ram1 = np.random.rand()
                if ram1 > 0.5:
                    aux += random.normalvariate(0, 0.2)

                aux = np.ceil(aux)
                aux = ((aux <= self.b_min) * self.b_min) + ((aux > self.b_min) * aux)
                aux = ((aux >= self.b_max) * self.b_max) + ((aux < self.b_max) * aux)
                # ------------------------------
                # add to array of lightning
                # ------------------------------
                aux = aux.astype(int)
                tot_light[i, :self.dim] = aux[:]
                tot_light[i, self.dim] = self.func[tuple(aux[:])]
        else:
            for i in range(1, n_points):
                rad = step * dir_
                aux = tot_light[i-1, :self.dim] + rad
                # ----------------------------------
                # mutation
                # ----------------------------------
                ram1 = np.random.rand()
                if ram1 > 0.5:
                    aux += random.normalvariate(0, 0.2)

                aux = np.ceil(aux)
                aux = ((aux <= self.b_min) * self.b_min) + ((aux > self.b_min) * aux)
                aux = ((aux >= self.b_max) * self.b_max) + ((aux < self.b_max) * aux)
                # ------------------------------
                # add to array of lightning
                # ------------------------------
                aux = aux.astype(int)
                tot_light[i, :self.dim] = aux[:]
                tot_light[i, self.dim] = self.func[tuple(aux[:])]

        # --------------------------------------------------------------------------------------------------
        # secondary branches
        # --------------------------------------------------------------------------------------------------
        aux_l = np.zeros(self.dim+1)
        beta = 0.3
        # number of points to branches
        r_points = int(np.round(n_points * 2))

        for p in range(n_points):
            # ------------------------------------------------------------------------------
            # best fitness in the lightning
            # ------------------------------------------------------------------------------
            _, fit_max = min(enumerate(tot_light[:, self.dim]), key=operator.itemgetter(1))
            # ------------------------------------------------------------------------------
            # roulette wheel selection
            # ------------------------------------------------------------------------------
            prob = np.exp(-beta * tot_light[p, self.dim] / (fit_max + np.finfo(float).eps))
            r1 = np.random.rand()
            if prob > r1:
                for p1 in range(r_points):
                    rad = step * np.random.randint(-1, 2, self.dim)
                    # -----------------------------------
                    # start or continue branch
                    # -----------------------------------
                    if p1 == 0:
                        aux = tot_light[p, :self.dim] + rad
                    else:
                        aux = tot_light[-1, :self.dim] + rad
                    # -----------------------------------
                    # mutation
                    # -----------------------------------
                    ram1 = np.random.rand()
                    if ram1 > 0.5:
                        aux += random.normalvariate(0, 0.2)
                    aux = np.ceil(aux)
                    aux = ((aux <= self.b_min) * self.b_min) + ((aux > self.b_min) * aux)
                    aux = ((aux >= self.b_max) * self.b_max) + ((aux < self.b_max) * aux)
                    # ------------------------------
                    # add to array of lightning
                    # ------------------------------
                    aux = aux.astype(int)
                    aux_l[:self.dim] = aux[:]
                    aux_l[self.dim] = self.func[tuple(aux[:])]

                    tot_light = np.append(tot_light, [aux_l], axis=0)

        return tot_light

    # -----------------------------------------------------------------------------------
    # bounds according to exponential function
    # -----------------------------------------------------------------------------------
    def groups_exp(self, max_val, min_val, ng):
        x = np.arange(0, ng, 1)
        a = max_val
        b = np.exp(np.log(min_val / a) / ng)
        return np.array(np.round(a * np.power(b, x)))

    def lia_min(self):
        # -------------------------------------
        # initialization negative charges
        # -------------------------------------
        self.flashes_init()
        # --------------------------------------------
        # obtain accuracy for steps values
        # --------------------------------------------
        ranges_itr = np.array([0.0, 0.3, 0.6, 0.8, 1.0])
        val_itr = np.round(self.itr * ranges_itr)
        n_groups = len(ranges_itr)
        g_step_min = self.groups_exp(np.round(self.min_step / 2), self.min_step, n_groups)
        g_step_max = self.groups_exp(np.round(self.max_step / 2), self.max_step, n_groups)
        # print(g_step_min, g_step_max)

        fit_r, best_pot, ide = [], [], 0
        min_step, max_step = self.min_step, self.max_step
        plt.figure()
        plt.ion()
        for itr in range(self.itr):
            # print(itr)
            # -------------------------------------------------------------------------------------------
            # graph objective function
            # -------------------------------------------------------------------------------------------
            plt.clf()
            plt.imshow(self.func, cmap=cm.coolwarm, vmin=abs(self.func).min(), vmax=abs(self.func).max())

            # -----------------------------------------------------------------------------------
            # update global max and min potentials
            # -----------------------------------------------------------------------------------
            pos_max, f_max = min(enumerate(self.light[:, self.dim]), key=operator.itemgetter(1))
            _, f_min = max(enumerate(self.light[:, self.dim]), key=operator.itemgetter(1))

            # --------------------------------
            # current global best potential
            # --------------------------------
            if itr == 0:
                best_pot = deepcopy(self.light[pos_max, :])
            else:
                if self.light[pos_max, self.dim] < best_pot[self.dim]:
                    best_pot = deepcopy(self.light[pos_max, :])

            fit_r.append(best_pot[self.dim])
            # print(best_pot)
            # print(self.space[int(best_pot[1])], self.space[int(best_pot[0])])

            # ---------------------------------------------------------------------------------------------
            # compute the number of points and step for branches
            # ---------------------------------------------------------------------------------------------
            fi_ = (self.light[:, self.dim] - f_min) / (f_max - f_min + np.finfo(float).eps)
            points = np.round(self.min_p + fi_ * (self.max_p - self.min_p))
            # ---------------------------------------------------------------------------
            # use accuracy factor to step values according to groups
            # ---------------------------------------------------------------------------
            if itr == val_itr[ide]:
                # print('------------range-----------: ' + str(ide))
                min_step = g_step_min[ide]
                max_step = g_step_max[ide]
                ide += 1
            steps = np.round(min_step + fi_ * (max_step - min_step))

            # ---------------------------------------------------------------------------
            # generate branches
            # ---------------------------------------------------------------------------
            for i in range(self.nt):

                branch = self.branches_min(self.light[i, :], int(points[i]), best_pot[:self.dim], steps[i], fc=1)
                # ------------------------------------------------------------------------
                # update best local potential
                # ------------------------------------------------------------------------
                pos_m, _ = min(enumerate(branch[:, self.dim]), key=operator.itemgetter(1))
                if self.loc_best[i, self.dim] > branch[pos_m, self.dim]:
                    self.loc_best[i, :] = branch[pos_m, :]
                # ------------------------------------------------------------------------
                # update second best local potential
                # ------------------------------------------------------------------------
                pos_m1, _ = max(enumerate(branch[:, self.dim]), key=operator.itemgetter(1))
                self.loc_max_b[i, :] = branch[pos_m1, :]

                plt.scatter(branch[:, 1], branch[:, 0], marker='o', c='g', s=2)

            # -----------------------------------------------------------
            # update new positions of potentials
            # -----------------------------------------------------------
            for i in range(self.nt):
                new_p1, new_p2 = np.zeros(self.dim+1), np.zeros(self.dim+1)
                aux_p1, aux_p2 = self.loc_best[i, :self.dim], self.light[i, :self.dim]
                aux_p1, aux_p2 = aux_p1.astype(int), aux_p2.astype(int)
                # ----------------------------------------------------------------------------------------
                # compute delta and update aux potential
                # ----------------------------------------------------------------------------------------
                aux_p1 = aux_p1 + np.random.rand(self.dim)*(best_pot[:self.dim] - self.loc_best[i, :self.dim])
                aux_p1 = ((aux_p1 <= self.b_min) * self.b_min) + ((aux_p1 > self.b_min) * aux_p1)
                aux_p1 = ((aux_p1 >= self.b_max) * self.b_max) + ((aux_p1 < self.b_max) * aux_p1)
                aux_p1 = aux_p1.astype(int)
                new_p1[:self.dim] = aux_p1[:]
                new_p1[self.dim] = self.func[tuple(aux_p1[:])]
                # ----------------------------------------------------------------------------------------
                aux_p2 = aux_p2 + np.random.rand(self.dim) * (best_pot[:self.dim] - self.light[i, :self.dim])
                aux_p2 = ((aux_p2 <= self.b_min) * self.b_min) + ((aux_p2 > self.b_min) * aux_p2)
                aux_p2 = ((aux_p2 >= self.b_max) * self.b_max) + ((aux_p2 < self.b_max) * aux_p2)
                aux_p2 = aux_p2.astype(int)
                new_p2[:self.dim] = aux_p2[:]
                new_p2[self.dim] = self.func[tuple(aux_p2[:])]
                # ---------------------------------------------
                # update new position
                # ---------------------------------------------
                if new_p1[self.dim] < self.loc_best[i, self.dim]:
                    self.light[i, :] = new_p1[:]
                elif new_p2[self.dim] < self.loc_best[i, self.dim]:
                    self.light[i, :] = new_p2[:]
                elif self.loc_best[i, self.dim] < self.light[i, self.dim]:
                    self.light[i, :] = self.loc_best[i, :]
                else:
                    self.light[i, :] = self.loc_max_b[i, :]

            plt.scatter(self.light[:, 1], self.light[:, 0], marker='o', c='black', s=5)
            plt.scatter(self.loc_best[:, 1], self.loc_best[:, 0], marker='o', c='r', s=5)
            plt.scatter(best_pot[1], best_pot[0], marker='o', c='y', s=5)
            plt.pause(0.1)
        plt.ioff()
        return fit_r


if __name__ == "__main__":

    dime = 2
    size = 500
    F = Func(dim=dime, size_max=size)
    f1, space = F.function_3()

    env = LIA(nt=20, itr=100, ini='R', func=f1, spa=space, size_t=size)
    f = env.lia_min()

    fig1 = plt.figure(2)
    axf = fig1.add_subplot(1, 1, 1)
    axf.plot(np.array(f), c='b', label='LiA-fitness')
    plt.legend(loc='best')
    plt.title('Optimization function')
    plt.ylabel('Fitness value')
    plt.xlabel('Iterations')
    axf.grid()
    plt.show()



