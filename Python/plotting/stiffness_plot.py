import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit as cv

#-----CONSTANTS----------------------------#
kB = 1.38064852e-23
T = 293.15
lpix = 4.68e-8


matplotlib.rcParams['figure.dpi']   = 150
matplotlib.rcParams['font.family']  = "serif"

def fit_func(x, a):
    return a*x

#------BEAD MEASURMENT----------------------#
bead_power      = np.asarray([0, 10, 20, 30, 40, 40])
bead_x_stiff    = np.asarray([2.3652e-5, 2.4569e-7, 1.665e-5, 8.3591e-7, 1.9969e-5, 9.6086e-5])
bead_y_stiff    = np.asarray([8.2974e-5, 2.9425e-7, 1.6591e-5, 9.7104e-7, 1.7268e-5, 8.9602e-5])
bead_t_stiff    = np.sqrt( np.asarray(bead_x_stiff)**2 + np.asarray(bead_y_stiff)**2 )
bead_x_err = lpix*((kB*T)**(-1/2))*(bead_x_stiff*1e-3)**(3/2)
bead_y_err = lpix*((kB*T)**(-1/2))*(bead_y_stiff*1e-3)**(3/2)


to_fit_total = np.sqrt(  np.asarray([1.665e-5, 1.9969e-5, 9.6086e-5])**2 + np.asarray([1.6591e-5, 1.7268e-5, 8.9602e-5]))

#------FUNCTION FIT-------------------------#
param_b_x, trash = cv(fit_func, np.asarray([20, 40, 40]), np.asarray([1.665e-5, 1.9969e-5, 9.6086e-5]) )
param_b_y, trash = cv(fit_func, np.asarray([20, 40, 40]), np.asarray([1.6591e-5, 1.7268e-5, 8.9602e-5]) )
param_b_t, trash = cv(fit_func, np.asarray([20, 40, 40]), to_fit_total)

#------TRAP MEASURMENT----------------------#
trap_power      = np.asarray([0, 5, 10, 20, 30, 40])
trap_x_stiff    = np.asarray([3.6463e-7, 8.2428e-5, 0.00010879, 0.00030305, 0.00061455, 0.00075512])
trap_y_stiff    = np.asarray([5.55e-7, 4.1822e-5, 2.5336e-5, 0.00011004, 0.00016901, 0.00025401])
trap_t_stiff    = np.sqrt( np.asarray(trap_x_stiff)**2 + np.asarray(trap_y_stiff)**2 )
trap_x_err = lpix*((kB*T)**(-1/2))*(trap_x_stiff*1e-3)**(3/2)
trap_y_err = lpix*((kB*T)**(-1/2))*(trap_y_stiff*1e-3)**(3/2)

#------FUNCTION FIT-------------------------#
param_t_x, cov_t_x = cv(fit_func, trap_power, trap_x_stiff, sigma=trap_x_err)
param_t_y, cov_t_y = cv(fit_func, trap_power, trap_y_stiff, sigma=trap_y_err)
#param_t_t, cov_t_t = cv(fit_func, trap_power, trap_t_stiff, sigma=trap_t_err)


matplotlib.rcParams['lines.linestyle']=''

X_b = np.linspace(np.min(bead_power),np.max(bead_power), 50)

fig, ax1 = plt.subplots()
ax1.errorbar(bead_power, bead_y_stiff, yerr=bead_y_err, fmt='', label=r"$k_y$", marker=".", color='black')
ax1.errorbar(bead_power, bead_x_stiff, yerr=bead_x_err, fmt='', label=r"$k_x$", marker=".", color='red')
#ax1.errorbar(bead_power, bead_t_stiff, yerr=bead_t_err, fmt='', label=r"$k_{tot}$", marker=".", color='blue')

Y_1 = fit_func(X_b, *param_b_y)
Y_2 = fit_func(X_b, *param_b_x)
#Y_3 = fit_func(np.linspace(0-0.5,40), *param_b_t)

ax1.plot(X_b, fit_func(X_b, *param_b_y), linestyle='--', color='black')
ax1.plot(X_b, fit_func(X_b, *param_b_x), linestyle='--', color='red')

ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax1.set_xlabel(r'Beam output power [$mW$]')
ax1.set_ylabel(r"Stiffness $k_x$, $k_y$ [$pN/nm$]")
fig.set_size_inches((5,4))
plt.subplots_adjust(left=0.14, bottom=0.14, right=0.99)
#plt.semilogy()
ax1.legend()
fig.tight_layout()
plt.savefig('plots/beam.png',dpi=200)
plt.close()



matplotlib.rcParams['lines.linestyle']=''

X_t = np.linspace(np.min(trap_power),np.max(trap_power), 50)

plt.errorbar(trap_power, trap_x_stiff, yerr=trap_x_err, fmt="", label=r"$k_x$", marker=".", color='black')
plt.errorbar(trap_power, trap_y_stiff, yerr=trap_y_err, fmt="", label=r"$k_y$", marker=".", color='red')
#plt.errorbar(trap_power, trap_t_stiff, yerr=trap_t_err, fmt="", label=r"$k_{tot}$", marker=".", color='blue')

plt.plot(X_t, fit_func(X_t, *param_t_x), linestyle='--', color='black')
plt.plot(X_t, fit_func(X_t, *param_t_y), linestyle='--', color='red')
#plt.plot(X_t, fit_func(X_t, *param_t_t), linestyle='--', color='blue')


plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.xlabel(r'Beam output power [$mW$]')
plt.ylabel(r"Stiffness [$pN/nm$]")
plt.gcf().set_size_inches((5,4))
plt.subplots_adjust(left=0.14, bottom=0.14, right=0.99)
#plt.semilogy()
plt.legend()
plt.tight_layout()
plt.savefig('plots/trap.png',dpi=200)

print(param_b_x, param_b_y)
print(param_t_x, param_t_y)

print(bead_x_err)
print(bead_y_err)

print(trap_x_err)
print(trap_y_err)