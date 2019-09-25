import numpy as np
import random
from copy import copy

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
#                               Solver with collective intelligence algorithm called LIA
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------
# generate branches
# ------------------------------------------------------------------------------------
def branches(fun, light, points, x_best_, f_max, step, dim, l_bound, u_bound):
    # ----------------------------------------
    # direction for main branch
    # ----------------------------------------
    dir_ = np.ones(dim)
    for d in range(dim):
        diff = (x_best_[d] - light[d])
        if diff != 0:
            dir_[d] = diff / np.abs(diff)
        else:
            if np.random.rand() > 0.5:
                dir_[d] *= -1
    # --------------------------------------------------------------------------------
    # main branch
    # --------------------------------------------------------------------------------
    tot_p = np.zeros((points, dim))
    tot_p[0, :] = light

    if np.random.rand() > 0.70:
        for i in range(1, points):
            rad = step * np.random.randint(-1, 2, dim)
            tot_p[i] = tot_p[i-1] + rad
            # ----------------------------------
            # mutation
            # ----------------------------------
            ram1 = np.random.rand()
            if ram1 > 0.5:
                tot_p[i] += random.normalvariate(0, 0.03)
    else:
        for i in range(1, points):
            rad = step * dir_
            tot_p[i] = tot_p[i-1] + rad
            # ----------------------------------
            # mutation
            # ----------------------------------
            ram1 = np.random.rand()
            if ram1 > 0.5:
                tot_p[i] += random.normalvariate(0, 0.03)

    tot_p = ((tot_p <= l_bound) * l_bound) + ((tot_p > l_bound) * tot_p)
    tot_p = ((tot_p >= u_bound) * u_bound) + ((tot_p < u_bound) * tot_p)
    # -----------------------------------------------------------------------------------
    # secondary branches
    # -----------------------------------------------------------------------------------
    beta = 0.3
    r_points = int(np.round(points * 0.8))
    for i in range(points):
        f1 = [fun(x) for x in tot_p]
        f1 = np.array(f1)
        idx_max = np.argmin(f1) if len(f1) else None
        if idx_max is not None:
            f_max1 = f1[idx_max]
        else:
            f_max1 = f_max
        # --------------------------------------
        # roulette wheel selection
        # --------------------------------------
        prob = np.exp(-beta * f1[i] / (f_max1 + np.finfo(float).eps))
        r1 = np.random.rand()
        if prob > r1:
            # ------------------------------------
            # generate secondary branches
            # ------------------------------------
            for j in range(r_points):
                rad = step * np.random.randint(-1, 2, dim)
                # -----------------------------------
                # start or continue branch
                # -----------------------------------
                if j == 0:
                    aux = tot_p[i] + rad
                else:
                    aux = tot_p[-1] + rad
                # -----------------------------------
                # mutation
                # -----------------------------------
                ram1 = np.random.rand()
                if ram1 > 0.5:
                    aux += random.normalvariate(0, 0.03)

                aux = ((aux <= l_bound) * l_bound) + ((aux > l_bound) * aux)
                aux = ((aux >= u_bound) * u_bound) + ((aux < u_bound) * aux)

                tot_p = np.append(tot_p, [aux], axis=0)

    return tot_p


# -----------------------------------------------------------------------------------------------------------
# particle random initialization using flashes
# -----------------------------------------------------------------------------------------------------------
def flashes_lia(fun, n_part, dime, low_b, up_b, rep=5, p_b=0.5):
    # --------------------------------------------------------------------
    # generate random groups of initialization particles
    # --------------------------------------------------------------------
    light_s = low_b + (up_b - low_b) * np.random.rand(n_part, dime)
    for i in range(rep-1):
        light = low_b + (up_b - low_b) * np.random.rand(n_part, dime)
        light_s = np.vstack([light_s, light])
    # --------------------------------------------------------------------
    # normalization fitness function
    # --------------------------------------------------------------------
    fit_s = [fun(x) for x in light_s]
    fit_s = np.array(fit_s)
    idx_max = np.argmax(fit_s) if len(fit_s) else None
    idx_min = np.argmin(fit_s) if len(fit_s) else None
    x_min, f_max, f_min = light_s[idx_min], fit_s[idx_max], fit_s[idx_min]
    fit_n = (fit_s - f_min) / (f_max - f_min)
    # ----------------------------------------------------------------------
    # select better positions according to selected probability
    # ----------------------------------------------------------------------
    p_prob = []
    while len(p_prob) < n_part*n_part:
        p_prob = np.where(fit_n < p_b)[0].tolist()
        p_b += 0.05
    # --------------------------------------------------------------------
    # permutation to generate random groups
    # --------------------------------------------------------------------
    p_prob = np.random.permutation(p_prob)
    p_prob = p_prob[:n_part*n_part]
    # --------------------------------------------------------------------
    n_rep, val_ref = 10 * dime, 0
    light_f = []
    # --------------------------------------------------------------------
    # iterative process to select better particles
    # --------------------------------------------------------------------
    for i1 in range(n_rep):
        data, div_, prob_sum = [], [], []
        pos_f = np.random.permutation(p_prob).reshape((n_part, n_part))
        for j in range(n_part):
            light_f1 = x_min
            for i in range(0, n_part-1):
                light_f1 = np.vstack([light_f1, light_s[pos_f[j, i]]])
            prob_sum.append(np.sum(fit_n[pos_f[j, :]]))
            data.append(light_f1)
            # ------------------------------------------------------------------------------------------
            # compute diversity factor of initialization group
            # ------------------------------------------------------------------------------------------
            center = np.sum(light_f1, axis=0) / n_part
            diversity = (1/n_part) * np.sum(np.sqrt((1/dime) * np.sum(np.power(light_f1 - center, 2))))
            div_.append(diversity)

        data, div_, prob_sum = np.array(data), np.array(div_), np.array(prob_sum)
        idx_div = np.argmax(div_)
        val_div = div_[idx_div]
        # ----------------------------------------------------------
        # select the better group
        # ----------------------------------------------------------
        if val_div > val_ref and prob_sum[idx_div] / n_part <= p_b:
            light_f = data[idx_div, :, :]
            val_ref = val_div
    return np.array(light_f)


# -----------------------------------------------------------------------------------
# steps accuracy according to iteration group
# -----------------------------------------------------------------------------------
def steps_hdp(fi_i, step_ini_, step_fin_, nt_, fact_):
    # ------------------------------------------
    # generate a list of steps from step_ini
    # ------------------------------------------
    step = groups_exp(step_fin_, step_ini_, nt_) * fact_
    # ------------------------------------------
    # assign steps according to fitness sorted
    # ------------------------------------------
    fi_s = np.argsort(fi_i)
    new_steps = np.zeros(nt_)
    for p in range(nt_):
        new_steps[fi_s[p]] = step[p]

    return new_steps


# -----------------------------------------------------------------------------------
# bounds according to exponential and lineal functions
# -----------------------------------------------------------------------------------
def groups_exp(min_val, max_val, ng):
    x = np.arange(0, ng, 1)
    a = max_val
    b = np.exp(np.log(min_val / a) / ng)
    return np.array(a * np.power(b, x))


def groups_lin(min_val, max_val, ng):
    x = np.arange(0, ng, 1)
    a = max_val
    b = (min_val - a) / ng
    return np.array(a + b * x)


# ------------------------------------------------------------------------------------
# main procedure
# ------------------------------------------------------------------------------------
def lia(fun, nt=20, min_p=4, max_p=12, stp_min=1e-4, stp_max=1e-4, itr=100, ide_dim=0, n_gr=4):
    max_s, min_s = stp_min, stp_max
    l_bound, u_bound = np.array(fun.lower_bounds), np.array(fun.upper_bounds)
    dim, f_max, f_min = len(l_bound), None, None
    x_max, x_best, f_best, leg = None, None, None, None
    itr_done, ide, step_ini, step_fin = 0, 0, 0, 0
    a, b = 0.0, 1.0
    factors = [2.5, 2.5, 1.5, 1.5]
    fact = factors[ide_dim]
    # --------------------------------------------
    # obtain accuracy for steps values
    # --------------------------------------------
    # ranges_itr = np.array([0.0, 0.30, 0.60, 0.80, 1.0])
    # val_itr = np.round(itr * ranges_itr)
    ini_i = [np.round(0.02*itr), np.round(0.03*itr), np.round(0.05*itr), np.round(0.05*itr)]
    y = ini_i[ide_dim]
    itr_group = np.round(groups_lin(itr, y, n_gr))
    itr_group = np.insert(itr_group, 0, 0)
    print(itr_group)
    # --------------------------------------------------------------
    # groups for steps
    # --------------------------------------------------------------
    steps_group = groups_exp(min_s, max_s, n_gr)
    steps_group = np.insert(steps_group, len(steps_group), min_s)
    print(steps_group)
    # --------------------------------------------
    # to break iterations
    # --------------------------------------------
    fit_p, fit_i = 0, 1
    max_rep, cont_rep, ini_break = np.round(0.1*itr), 0, np.round(0.6*itr)
    # -------------------------------------------------------------
    # random initializer "negative charges"
    # -------------------------------------------------------------
    ret = [5, 15, 20, 25]
    prob = [0.4, 0.4, 0.3, 0.3]
    rt, pb = ret[ide_dim], prob[ide_dim]
    lights = flashes_lia(fun, nt, dim, l_bound, u_bound, rep=rt*5, p_b=pb)
    # lights = l_bound + (u_bound - l_bound) * np.random.rand(nt, dim)
    loc_best, loc_max_b = copy(lights), copy(lights)
    cont, fact_p = 0, 1e-1

    # --------------------------------------------------------------------------------
    # start iterative LIA algorithm
    # --------------------------------------------------------------------------------
    for k in range(itr):
        if fun.number_of_constraints > 0:
            contra = [fun.constraint(x) for x in lights]  # call constraints
            fitness = [fun(x) for i, x in enumerate(lights) if np.all(contra[i] <= 0)]
        else:
            fitness = [fun(x) for x in lights]

        fitness = np.array(fitness)
        # ---------------------------------------------------------------
        # Find global max and min
        # ---------------------------------------------------------------
        idx_max = np.argmin(fitness) if len(fitness) else None
        idx_min = np.argmax(fitness) if len(fitness) else None
        if idx_max is not None:
            x_max, f_max = lights[idx_max], fitness[idx_max]

        if idx_min is not None:
            x_min, f_min = lights[idx_min], fitness[idx_min]

        # ----------------------------------------------------
        # best potential
        # ----------------------------------------------------
        if k == 0:
            x_best, f_best = copy(x_max), copy(f_max)
        else:
            if f_max < f_best:
                x_best, f_best = copy(x_max), copy(f_max)

        # ----------------------------------------------------
        # compare current and previous fitness value
        # ----------------------------------------------------
        if k % 2 == 0:
            fit_p = f_best
        else:
            fit_i = f_best

        if k > ini_break:
            if fit_p == fit_i:
                cont_rep += 1
            else:
                cont_rep = 0

        # ---------------------------------------------------------------------------
        # compute the number of points and step for branches
        # ---------------------------------------------------------------------------
        fi_ = a + ((np.array(fitness) - f_min) / (f_max - f_min + np.finfo(float).eps)) * (b-a)
        # print(fi_)
        points = np.round(min_p + fi_ * (max_p - min_p))

        # ---------------------------------------------------------------------------
        # compute the accuracy factor to step values
        # ---------------------------------------------------------------------------
        if k == itr_group[ide]:
            print('----------------------------------------------')
            print('------------  Group of steps  ----------- :  ' + str(ide))
            print('----------------------------------------------')
            fact += ide
            if ide < len(steps_group)-3:
                step_ini = steps_group[ide]
                step_fin = steps_group[ide+2]

            steps_n = steps_hdp(fi_, step_ini, step_fin, nt, fact_p)
            if ide < len(steps_group)-1:
                ide += 1

        # steps = min_s + fi_ * (max_s - min_s)
        # print('steps ------: ' + str(steps))
        # print('steps_n ------: ' + str(steps_n))

        # ----------------------------------
        # function value of loc best
        # ----------------------------------
        f_loc = [fun(x) for x in loc_best]
        # ----------------------------------------------------------------------------------------
        # generate branches
        # ----------------------------------------------------------------------------------------
        for i in range(nt):
            points_ = int(points[i])
            branch = branches(fun, lights[i], points_, x_best, f_best, steps_n[i], dim, l_bound, u_bound)
            branch = np.array(branch)
            # --------------------------------------------------------------
            # update best local positions
            # --------------------------------------------------------------
            f1 = [fun(x) for x in branch]
            f1 = np.array(f1)
            idx_max = np.argmin(f1) if len(f1) else None
            if f1[idx_max] < f_loc[i]:
                loc_best[i] = branch[idx_max]
            # --------------------------------------------------------------
            # select the maximum point from branch
            # --------------------------------------------------------------
            f2 = [fun(x) for x in branch[1:]]
            f2 = np.array(f2)
            idx_min = np.argmin(f2) if len(f1) else None
            loc_max_b[i] = branch[idx_min]

        # update steps in each iteration
        steps_n += steps_hdp(fi_, step_ini, step_fin, nt, fact_p)

        # -----------------------------------------------------------------------------
        # update new positions of potentials
        # ------------------------------------------------------------------------------
        for i in range(nt):
            aux_lights = copy(lights[i])
            aux_bt_loc = copy(loc_best[i])
            # --------------------------------------------------------------------
            # new points to evaluate performance
            # --------------------------------------------------------------------
            aux_lights = aux_lights + np.random.rand(dim) * (x_best - lights[i])
            aux_bt_loc = aux_bt_loc + np.random.rand(dim) * (x_best - loc_best[i])

            # -------------------------------------------------
            # current fitness values
            # -------------------------------------------------
            f_aux_1, f_aux_2 = fun(aux_lights), fun(aux_bt_loc)
            f_bt_lo, f_light = fun(loc_best[i]), fun(lights[i])

            # ----------------------------------------------------------
            # assign new positions
            # ----------------------------------------------------------
            if f_aux_2 < f_bt_lo:
                lights[i] = aux_bt_loc
                # print('Light... aux_loc: ' + str(i) + ' : ' + str(aux_bt_loc))
            elif f_aux_1 < f_bt_lo:
                lights[i] = aux_lights
                # print('Light... aux_light: ' + str(i) + ' : ' + str(aux_lights))
            elif f_bt_lo < f_light:
                lights[i] = loc_best[i]
                # print('Light... loc_best: ' + str(i) + ' : ' + str(loc_best[i]))
            else:
                lights[i] = loc_max_b[i]
                # print('Light... loc_max: ' + str(i) + ' : ' + str(loc_max_b[i]))
        # ----------------------------------------------------
        # exit condition for large iterations
        # ----------------------------------------------------
        print(x_best, f_best, k)
        itr_done = k
        if fun.final_target_hit:
            leg = ['Optimization terminated successfully.']
            break
        elif cont_rep > max_rep:
            leg = ['Repetition fitness limit exceeded.']
            break
        else:
            leg = ['Iteration limit exceeded.']

    output = np.append(np.array([x_best, f_best, itr_done]), leg, axis=0)
    print(output)

    return output



