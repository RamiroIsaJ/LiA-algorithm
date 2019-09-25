import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import kruskal

dir_path = os.path.dirname(os.path.realpath(__file__))
path = dir_path + "/PruebasF/"
ide = 10
v_lia, v_pso, v_fwa, v_bee = [], [], [], []
# ---------------------------------------------------------------------------------
# read txt files
# ---------------------------------------------------------------------------------

for line in open(path+'Function_'+str(ide)+'_LiA'):
    v_lia.append([float(val) for val in line.rstrip('\n').split(',') if val != ''])

for line in open(path+'Function_'+str(ide)+'_PSO'):
    v_pso.append([float(val) for val in line.rstrip('\n').split(',') if val != ''])

for line in open(path+'Function_'+str(ide)+'_FWA'):
    v_fwa.append([float(val) for val in line.rstrip('\n').split(',') if val != ''])

for line in open(path+'Function_'+str(ide)+'_BEE'):
    v_bee.append([float(val) for val in line.rstrip('\n').split(',') if val != ''])
# ----------------------------------------------------------------------------------
# compute mean of each algorithm
# ----------------------------------------------------------------------------------
v_lia = np.array(v_lia)
mean_lia = np.mean(v_lia, axis=0)
# ----------------------------------------------------------------------------------
v_pso = np.array(v_pso)
mean_pso = np.mean(v_pso, axis=0)
# ----------------------------------------------------------------------------------
v_fwa = np.array(v_fwa)
mean_fwa = np.mean(v_fwa, axis=0)
# ----------------------------------------------------------------------------------
v_bee = np.array(v_bee)
mean_bee = np.mean(v_bee, axis=0)
# ----------------------------------------------------------------------------------
# non-parametric test
# ----------------------------------------------------------------------------------
data1, data2, data3, data4 = v_pso[:, -1], v_fwa[:, -1],  v_bee[:, -1],  v_lia[:, -1]
# ----------------------------------------------------------------------------------
m_lia = np.mean(data4)
std_lia = np.sqrt(np.sum(np.power(data4 - m_lia, 2)) / 50)
print(m_lia, std_lia)
# ----------------------------------------------------------------------------------
m_pso = np.mean(data1)
std_pso = np.sqrt(np.sum(np.power(data1 - m_pso, 2)) / 50)
print(m_pso, std_pso)
# ----------------------------------------------------------------------------------
m_fwa = np.mean(data2)
std_fwa = np.sqrt(np.sum(np.power(data2 - m_fwa, 2)) / 50)
print(m_fwa, std_fwa)
# ----------------------------------------------------------------------------------
m_bee = np.mean(data3)
std_bee = np.sqrt(np.sum(np.power(data3 - m_bee, 2)) / 50)
print(m_bee, std_bee)
# ----------------------------------------------------------------------------------
stat, p = kruskal(data1, data2, data3, data4)
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
    print('Same distributions (fail to reject H0)')
else:
    print('Different distributions (reject H0)')
# ----------------------------------------------------------------------------------
# plot boxes
# ----------------------------------------------------------------------------------
data = [data4, data1, data2, data3]
fig_1 = plt.figure()
ax = fig_1.add_subplot(111)
bp = ax.boxplot(data)
ax.set_xticklabels(['LiA', 'PSO', 'FWA', 'ABC'])
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
# ----------------------------------------------------------------------------------
# plot results
# ----------------------------------------------------------------------------------
fig = plt.figure()
plt.plot(mean_lia, '-r', label='LiA')
plt.plot(mean_pso, '--g', label='PSO')
plt.plot(mean_fwa, '--b', label='FWA')
plt.plot(mean_bee, '--c', label='ABC')
plt.legend(loc='best')
plt.ylabel('Best fitness value')
plt.xlabel('Iterations')
plt.title('Performance comparision')
plt.grid()
plt.show()
