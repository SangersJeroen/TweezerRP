import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['figure.dpi']   = 150
matplotlib.rcParams['font.family']  = "serif"

#------BEAD MEASURMENT----------------------#
bead_power      = [0, 10, 20, 30, 40, 40]
bead_x_stiff    = [2.3652e-5, 2.4569e-7, 1.665e-5, 8.3591e-7, 1.9969e-5, 9.6086e-5]
bead_y_stiff    = [8.2974e-5, 2.9425e-7, 1.6591e-5, 9.7104e-7, 1.7268e-5, 8.9602e-5]
bead_t_stiff    = np.sqrt( np.asarray(bead_x_stiff)**2 + np.asarray(bead_y_stiff)**2 )

#------TRAP MEASURMENT----------------------#
trap_power      = [0, 5, 10, 20, 30, 40]
trap_x_stiff    = [3.6463e-7, 8.2428e-5, 0.00010879, 0.00030305, 0.00061455, 0.00075512]
trap_y_stiff    = [5.55e-7, 4.1822e-5, 2.5336e-5, 0.00011004, 0.00016901, 0.00025401]
trap_t_stiff    = np.sqrt( np.asarray(trap_x_stiff)**2 + np.asarray(trap_y_stiff)**2 )

plt.close()
plt.plot(bead_power, bead_x_stiff, label=r"$k_x$", marker=".")
plt.plot(bead_power, bead_y_stiff, label=r"$k_y$", marker=".")
plt.plot(bead_power, bead_t_stiff, label=r"$k_{tot}$", marker=".")
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.xlabel(r'Beam output power [$mW$]')
plt.ylabel(r"Stiffness [$pN/nm$]")
plt.gcf().set_size_inches((5,4))
plt.subplots_adjust(left=0.14, bottom=0.14, right=0.99)
plt.semilogy()
plt.legend()
plt.show()

plt.close()
plt.scatter(trap_power, trap_x_stiff, label=r"$k_x$", marker=".", color='black')
plt.scatter(trap_power, trap_y_stiff, label=r"$k_y$", marker=".", color='red')
plt.scatter(trap_power, trap_t_stiff, label=r"$k_{tot}$", marker=".", color='blue')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.xlabel(r'Beam output power [$mW$]')
plt.ylabel(r"Stiffness [$pN/nm$]")
plt.gcf().set_size_inches((5,4))
plt.subplots_adjust(left=0.14, bottom=0.14, right=0.99)
plt.semilogy()
plt.legend()
plt.show()