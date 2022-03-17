import numpy as np
import matplotlib.pyplot as plt

d = np.loadtxt("fifth_force_limits.txt", delimiter=",")
d2 = np.loadtxt("review_2015.txt", delimiter=",", skiprows=1)
d3 = np.loadtxt("eot_wash2020.txt", delimiter=",", skiprows=1)

yd = 5.8e-19 * np.sqrt(d[:,1])
yd2 = 5.8e-19 * np.sqrt(d2[:,1])
yd3 = 5.8e-19 * np.sqrt(d3[:,1])

#plt.figure()
#plt.loglog(d[:,0], d[:,1])

plt.figure()
plt.loglog(0.2/(d[:,0]*1e6), yd)
plt.loglog(0.2/(d2[:,0]*1e6), yd2, label="2015 Review")
plt.loglog(0.2/(d3[:,0]*1e6), yd3, label="Eot-wash 2020")
plt.xlabel("Mediator mass (eV)")
plt.ylabel("Neutron coupling, $y_n$")
plt.grid(True)
plt.legend()
plt.savefig("yn_plot.png", transparent=True)

plt.figure()
plt.loglog(d[:,0]*1e6, yd)
plt.loglog(d2[:,0]*1e6, yd2, label="2015 Review")
plt.loglog(d3[:,0]*1e6, yd3, label="Eot-wash 2020")
plt.xlabel("1/(Mediator mass) (um)")
plt.ylabel("Neutron coupling, $y_n$")
plt.grid(True)
plt.legend()
plt.savefig("yn_plot_um.png") #, transparent=True)

#np.savetxt("fifth_force_lims_eV.txt", np.vstack((0.2/(d[:,0]*1e6), yd)).T)

np.savez("fifth_force_limits.npz",x = 0.2/(d[:,0]*1e6), y = yd) 

plt.show()
