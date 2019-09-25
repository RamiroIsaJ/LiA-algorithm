from LiA_min_optimization import LIA
from swarm_algorithms import PSO
from swarm_algorithms import FWA
from swarm_algorithms import BEE
from benchmark_functions import BenchFunctions as Func


# ------------------------------------------------------
# principal parameters
# ------------------------------------------------------
d = 2
size = 200
F = Func(dim=d, size_max=size)
ite = 200
test = 50
# ------------------------------------------------------
# Ackley function and initialization
# ------------------------------------------------------
f1, _ = F.function_0()
# ------------------------------------------------------
t_lia, t_pso, t_fwa, t_bee = [], [], [], []

print('Start function 0')
for rep in range(test):
    print(rep)
    # --------------------------------------------------
    lia = LIA(dim=d, itr=ite, size_t=size, func=f1, ini='R')
    fit_lia = lia.lia_min()
    t_lia.append(fit_lia)
# -------------------------------------------------------------------------
# save values in txt file
# -------------------------------------------------------------------------
with open('Function_0_LiA_3', 'w') as f:
    for w in t_lia:
        f.write(str(w)[1: -1] + "\n")

print('Save all files function 0')
# ------------------------------------------------------

# ------------------------------------------------------
# principal parameters
# ------------------------------------------------------
d = 3
size = 200
F = Func(dim=d, size_max=size)
ite = 200
test = 50
# ------------------------------------------------------
# Sphere function and initialization
# ------------------------------------------------------
f1, _ = F.function_1()
# ------------------------------------------------------
t_lia, t_pso, t_fwa, t_bee = [], [], [], []

print('Start function 1')
for rep in range(test):
    print(rep)
    # --------------------------------------------------
    lia = LIA(dim=d, itr=ite, size_t=size, func=f1, ini='R')
    fit_lia = lia.lia_min()
    t_lia.append(fit_lia)
# -------------------------------------------------------------------------
# save values in txt file
# -------------------------------------------------------------------------
with open('Function_1_LiA_3', 'w') as f:
    for w in t_lia:
        f.write(str(w)[1: -1] + "\n")

print('Save all files function 1')
# ------------------------------------------------------

# ------------------------------------------------------
# principal parameters
# ------------------------------------------------------
d = 2
size = 200
F = Func(dim=d, size_max=size)
ite = 200
test = 50
# ------------------------------------------------------
# Rosenbrock function and initialization
# ------------------------------------------------------
f1, _ = F.function_2()
# ------------------------------------------------------
t_lia, t_pso, t_fwa, t_bee = [], [], [], []

print('Start function 2')
for rep in range(test):
    print(rep)
    # --------------------------------------------------
    lia = LIA(dim=d, itr=ite, size_t=size, func=f1, ini='R')
    fit_lia = lia.lia_min()
    t_lia.append(fit_lia)
# -------------------------------------------------------------------------
# save values in txt file
# -------------------------------------------------------------------------
with open('Function_2_LiA_3', 'w') as f:
    for w in t_lia:
        f.write(str(w)[1: -1] + "\n")

print('Save all files function 2')
# ------------------------------------------------------

# ------------------------------------------------------
# principal parameters
# ------------------------------------------------------
d = 3
size = 200
F = Func(dim=d, size_max=size)
ite = 200
test = 50
# ------------------------------------------------------
# Rastrigin function and initialization
# ------------------------------------------------------
f1, _ = F.function_3()
# ------------------------------------------------------
t_lia, t_pso, t_fwa, t_bee = [], [], [], []

print('Start function 3')
for rep in range(test):
    print(rep)
    # --------------------------------------------------
    lia = LIA(dim=d, itr=ite, size_t=size, func=f1, ini='R')
    fit_lia = lia.lia_min()
    t_lia.append(fit_lia)
# -------------------------------------------------------------------------
# save values in txt file
# -------------------------------------------------------------------------
with open('Function_3_LiA_3', 'w') as f:
    for w in t_lia:
        f.write(str(w)[1: -1] + "\n")

print('Save all files function 3')
# ------------------------------------------------------

# ------------------------------------------------------
# principal parameters
# ------------------------------------------------------
d = 2
size = 200
F = Func(dim=d, size_max=size)
ite = 200
test = 50
# ------------------------------------------------------
# Cigar function and initialization
# ------------------------------------------------------
f1, _ = F.function_4()
# ------------------------------------------------------
t_lia, t_pso, t_fwa, t_bee = [], [], [], []

print('Start function 4')
for rep in range(test):
    print(rep)
    # --------------------------------------------------
    lia = LIA(dim=d, itr=ite, size_t=size, func=f1, ini='R')
    fit_lia = lia.lia_min()
    t_lia.append(fit_lia)
    '''
    # --------------------------------------------------
    pso = PSO(dim=d, itr=ite, size_t=size, func=f1)
    pso.rand_ini()
    fit_pso = pso.pso_min()
    t_pso.append(fit_pso)
    # --------------------------------------------------
    fwa = FWA(dim=d, itr=ite, size_t=size, func=f1)
    fwa.rand_ini()
    fit_fwa = fwa.fwa_min()
    t_fwa.append(fit_fwa)
    # --------------------------------------------------
    bee = BEE(dim=d, itr=ite, size_t=size, func=f1)
    bee.chaotic_ini()
    fit_bee = bee.bee_min()
    t_bee.append(fit_bee)
    '''
# -------------------------------------------------------------------------
# save values in txt file
# -------------------------------------------------------------------------
with open('Function_4_LiA_3', 'w') as f:
    for w in t_lia:
        f.write(str(w)[1: -1] + "\n")
'''
with open('Function_4_PSO', 'w') as f:
    for w in t_pso:
        f.write(str(w)[1: -1] + "\n")

with open('Function_4_FWA', 'w') as f:
    for w in t_fwa:
        f.write(str(w)[1: -1] + "\n")

with open('Function_4_BEE', 'w') as f:
    for w in t_bee:
        f.write(str(w)[1: -1] + "\n")
'''

print('Save all files function 4')
# ------------------------------------------------------

# ------------------------------------------------------
# principal parameters
# ------------------------------------------------------
d = 3
size = 200
F = Func(dim=d, size_max=size)
ite = 200
test = 50
# ------------------------------------------------------
# Griewank function and initialization
# ------------------------------------------------------
f1, _ = F.function_5()
# ------------------------------------------------------
t_lia, t_pso, t_fwa, t_bee = [], [], [], []

print('Start function 5')
for rep in range(test):
    print(rep)
    # --------------------------------------------------
    lia = LIA(dim=d, itr=ite, size_t=size, func=f1, ini='R')
    fit_lia = lia.lia_min()
    t_lia.append(fit_lia)
# -------------------------------------------------------------------------
# save values in txt file
# -------------------------------------------------------------------------
with open('Function_5_LiA_3', 'w') as f:
    for w in t_lia:
        f.write(str(w)[1: -1] + "\n")

print('Save all files function 5')
# ------------------------------------------------------

# ------------------------------------------------------
# principal parameters
# ------------------------------------------------------
d = 2
size = 200
F = Func(dim=d, size_max=size)
ite = 200
test = 50
# ------------------------------------------------------
# Schwefel function and initialization
# ------------------------------------------------------
f1, _ = F.function_6()
# ------------------------------------------------------
t_lia, t_pso, t_fwa, t_bee = [], [], [], []

print('Start function 6')
for rep in range(test):
    print(rep)
    # --------------------------------------------------
    lia = LIA(dim=d, itr=ite, size_t=size, func=f1, ini='R')
    fit_lia = lia.lia_min()
    t_lia.append(fit_lia)
# -------------------------------------------------------------------------
# save values in txt file
# -------------------------------------------------------------------------
with open('Function_6_LiA_3', 'w') as f:
    for w in t_lia:
        f.write(str(w)[1: -1] + "\n")

print('Save all files function 6')
# ------------------------------------------------------

# ------------------------------------------------------
# principal parameters
# ------------------------------------------------------
d = 3
size = 200
F = Func(dim=d, size_max=size)
ite = 200
test = 50
# ------------------------------------------------------
# Drop wave function and initialization
# ------------------------------------------------------
f1, _ = F.function_7()
# ------------------------------------------------------
t_lia, t_pso, t_fwa, t_bee = [], [], [], []

print('Start function 7')
for rep in range(test):
    print(rep)
    # --------------------------------------------------
    lia = LIA(dim=d, itr=ite, size_t=size, func=f1, ini='R')
    fit_lia = lia.lia_min()
    t_lia.append(fit_lia)
# -------------------------------------------------------------------------
# save values in txt file
# -------------------------------------------------------------------------
with open('Function_7_LiA_3', 'w') as f:
    for w in t_lia:
        f.write(str(w)[1: -1] + "\n")

print('Save all files function 7')
# ------------------------------------------------------

# ------------------------------------------------------
# principal parameters
# ------------------------------------------------------
d = 2
size = 200
F = Func(dim=d, size_max=size)
ite = 200
test = 50
# ------------------------------------------------------
# Levy function and initialization
# ------------------------------------------------------
f1, _ = F.function_8()
# ------------------------------------------------------
t_lia, t_pso, t_fwa, t_bee = [], [], [], []

print('Start function 8')
for rep in range(test):
    print(rep)
    # --------------------------------------------------
    lia = LIA(dim=d, itr=ite, size_t=size, func=f1, ini='R')
    fit_lia = lia.lia_min()
    t_lia.append(fit_lia)
# -------------------------------------------------------------------------
# save values in txt file
# -------------------------------------------------------------------------
with open('Function_8_LiA_3', 'w') as f:
    for w in t_lia:
        f.write(str(w)[1: -1] + "\n")

print('Save all files function 8')
# ------------------------------------------------------

# ------------------------------------------------------
# principal parameters
# ------------------------------------------------------
d = 2
size = 200
F = Func(dim=d, size_max=size)
ite = 200
test = 50
# ------------------------------------------------------
# Lang function and initialization
# ------------------------------------------------------
f1, _ = F.function_9()
# ------------------------------------------------------
t_lia, t_pso, t_fwa, t_bee = [], [], [], []

print('Start function 9')
for rep in range(test):
    print(rep)
    # --------------------------------------------------
    lia = LIA(dim=d, itr=ite, size_t=size, func=f1, ini='R')
    fit_lia = lia.lia_min()
    t_lia.append(fit_lia)
# -------------------------------------------------------------------------
# save values in txt file
# -------------------------------------------------------------------------
with open('Function_9_LiA_3', 'w') as f:
    for w in t_lia:
        f.write(str(w)[1: -1] + "\n")

print('Save all files function 9')

# ------------------------------------------------------
# principal parameters
# ------------------------------------------------------
d = 2
size = 200
F = Func(dim=d, size_max=size)
ite = 200
test = 50
# ------------------------------------------------------
# Himmelblau function and initialization
# ------------------------------------------------------
f1, _ = F.function_10()
# ------------------------------------------------------
t_lia, t_pso, t_fwa, t_bee = [], [], [], []

print('Start function 10')
for rep in range(test):
    print(rep)
    # --------------------------------------------------
    lia = LIA(dim=d, itr=ite, size_t=size, func=f1, ini='R')
    fit_lia = lia.lia_min()
    t_lia.append(fit_lia)
    # --------------------------------------------------
    pso = PSO(dim=d, itr=ite, size_t=size, func=f1)
    pso.rand_ini()
    fit_pso = pso.pso_min()
    t_pso.append(fit_pso)
    # --------------------------------------------------
    fwa = FWA(dim=d, itr=ite, size_t=size, func=f1)
    fwa.rand_ini()
    fit_fwa = fwa.fwa_min()
    t_fwa.append(fit_fwa)
    # --------------------------------------------------
    bee = BEE(dim=d, itr=ite, size_t=size, func=f1)
    bee.chaotic_ini()
    fit_bee = bee.bee_min()
    t_bee.append(fit_bee)
# -------------------------------------------------------------------------
# save values in txt file
# -------------------------------------------------------------------------
with open('Function_10_LiA', 'w') as f:
    for w in t_lia:
        f.write(str(w)[1: -1] + "\n")

with open('Function_10_PSO', 'w') as f:
    for w in t_pso:
        f.write(str(w)[1: -1] + "\n")

with open('Function_10_FWA', 'w') as f:
    for w in t_fwa:
        f.write(str(w)[1: -1] + "\n")

with open('Function_10_BEE', 'w') as f:
    for w in t_bee:
        f.write(str(w)[1: -1] + "\n")

print('Save all files function 10')

# ------------------------------------------------------
# principal parameters
# ------------------------------------------------------
d = 2
size = 200
F = Func(dim=d, size_max=size)
ite = 200
test = 50
# ------------------------------------------------------
# Beale function and initialization
# ------------------------------------------------------
f1, _ = F.function_11()
# ------------------------------------------------------
t_lia, t_pso, t_fwa, t_bee = [], [], [], []

print('Start function 11')
for rep in range(test):
    print(rep)
    # --------------------------------------------------
    lia = LIA(dim=d, itr=ite, size_t=size, func=f1, ini='R')
    fit_lia = lia.lia_min()
    t_lia.append(fit_lia)
    # --------------------------------------------------
    pso = PSO(dim=d, itr=ite, size_t=size, func=f1)
    pso.rand_ini()
    fit_pso = pso.pso_min()
    t_pso.append(fit_pso)
    # --------------------------------------------------
    fwa = FWA(dim=d, itr=ite, size_t=size, func=f1)
    fwa.rand_ini()
    fit_fwa = fwa.fwa_min()
    t_fwa.append(fit_fwa)
    # --------------------------------------------------
    bee = BEE(dim=d, itr=ite, size_t=size, func=f1)
    bee.chaotic_ini()
    fit_bee = bee.bee_min()
    t_bee.append(fit_bee)
# -------------------------------------------------------------------------
# save values in txt file
# -------------------------------------------------------------------------
with open('Function_11_LiA', 'w') as f:
    for w in t_lia:
        f.write(str(w)[1: -1] + "\n")

with open('Function_11_PSO', 'w') as f:
    for w in t_pso:
        f.write(str(w)[1: -1] + "\n")

with open('Function_11_FWA', 'w') as f:
    for w in t_fwa:
        f.write(str(w)[1: -1] + "\n")

with open('Function_11_BEE', 'w') as f:
    for w in t_bee:
        f.write(str(w)[1: -1] + "\n")

print('Save all files function 11')
