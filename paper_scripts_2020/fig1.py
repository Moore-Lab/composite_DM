import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sp
import scipy.optimize as opt

SI_to_GeV = 1.87e18
distance = 0.0033

# [stackdati, -stackdato, tvec[:680]-0.02, tempt[120:], tt[120:]+0.5e-3, tvec, indatf, outdatf, mon], mon refer to the electrical impulses

A = np.load("fig1_info_filter_65to115Hz.npy", encoding='latin1', allow_pickle=True)
B = np.load("fig1_info_filter_5to350Hz.npy", encoding='latin1', allow_pickle=True)

energy = (1.602e-19) * 3.2 * 200. * 1e-4 * SI_to_GeV / distance

print (energy)

# plt.figure()
# plt.plot(A[2], A[0])
# plt.plot(A[2], A[1])
# plt.plot(A[4], A[3]/2., "k:")
#
# plt.figure()
# plt.plot(B[2], B[0])
# plt.plot(B[2], B[1])
# plt.plot(B[4], B[3]/2., "k:")
#
# plt.figure()
# plt.plot(A[5], A[6])
# plt.plot(A[5], -A[7])
# plt.plot(A[4] + 11.9869, A[3]/2., "k:")
# #plt.plot(A[5], 1e-10*A[8])
#
# plt.figure()
# plt.plot(B[5], B[6])
# plt.plot(B[5], -B[7])
# plt.plot(B[4] + 11.9869, B[3]/2., "k:")

###############

# fig, axs = plt.subplots(2, 2, sharex='col', sharey='row', gridspec_kw={'hspace': 0, 'wspace': 0})
#
# (ax1, ax2), (ax3, ax4) = axs
#
# ax3.plot(A[2], A[0])
# ax3.plot(A[2], A[1])
# ax3.plot(A[4], A[3]/2., "k:")
# ax3.set_xlim(-0.015, 0.040)
# ax3.set_ylim(-3e-9, 3e-9)
# ax3.vlines(0.0, -1e-8, 1e-8, colors="grey", alpha = 0.5)
#
# ax1.plot(B[2], B[0])
# ax1.plot(B[2], B[1])
# ax1.plot(B[4], B[3]/2., "k:")
# ax1.set_xlim(-0.015, 0.060)
# ax1.set_ylim(-4e-9, 5e-9)
# ax1.vlines(0.0, -1e-8, 1e-8, colors="grey", alpha = 0.5)
#
# ax2.plot(B[5]-11.9869, B[6])
# ax2.plot(B[5]-11.9869, -B[7])
# ax2.plot(B[4], B[3]/2., "k:")
# ax2.set_xlim(-0.015, 0.060)
# ax2.set_ylim(-4e-9, 5e-9)
# ax2.vlines(0.0, -1e-8, 1e-8, colors="grey", alpha = 0.5)
#
# ax4.plot(A[5] - 11.9869, A[6])
# ax4.plot(A[5] - 11.9869, -A[7])
# ax4.plot(A[4], A[3]/2., "k:")
# ax4.set_xlim(-0.015, 0.040)
# ax4.set_ylim(-3e-9, 3e-9)
# ax4.vlines(0.0, -1e-8, 1e-8, colors="grey", alpha = 0.5)
#
# for ax in axs.flat:
#     ax.label_outer()

############

fig, axs = plt.subplots(2, 1, sharex='col', sharey='row', figsize=(5, 2.5))
#fig, axs = plt.subplots(2, 1, sharex='col', sharey='row', gridspec_kw={'hspace': 0, 'wspace': 0}, figsize=(5, 2.5))

(ax1, ax2) = axs

ax1.plot((B[5]-11.9869)*1e3, 1e9*B[6], label = "In-loop")
ax1.plot((B[5]-11.9869)*1e3, -1e9*B[7], label = "Out-of-loop")
ax1.plot(B[4]*1e3, 1e9*B[3]/2., "k:")
ax1.set_xlim(-0.06*1e3, 0.060*1e3)
ax1.set_ylim(-3, 4)
ax1.vlines(0.0, -10, 10, colors="grey", alpha = 0.5)
ax1.legend(frameon=False, bbox_to_anchor=(0.65, 0.47))


ax2.plot((A[5] - 11.9869)*1e3, 1e9*A[6])
ax2.plot((A[5] - 11.9869)*1e3, -1e9*A[7])
ax2.plot(A[4]*1e3, 1e9*A[3]/2., "k:")
ax2.set_xlim(-0.06*1e3, 0.060*1e3)
ax2.set_ylim(-2, 2)
ax2.vlines(0.0, -10, 10, colors="grey", alpha = 0.5)

fig.text(0.02, 0.55, 'Displacement [nm]', va='center', ha='center', rotation='vertical')
plt.xlabel("Time [ms]")
#plt.ylabel("Displacement [nm]")

#plt.tight_layout(pad=0)

for ax in axs.flat:
    ax.label_outer()

plt.show()