import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

## calculate dR/dq for a massless mediator
mxlist = [5e3,]

vesc = 1.815e-3 ## galactic escape velocity
v0 = 7.34e-4 ## v0 parameter from Zurek group paper
ve = 8.172e-4 ## ve parameter from Zurek group paper
vmin = 1e-6
nvels = 100

rhoDM = 0.3 # dark matter mass density, GeV/cm^3

N_T = 3e14 ## number neutrons
alpha_n = 1.2e-8


hbarc = 0.2e-13 # GeV cm
c = 3e10 # cm/s


qq = np.linspace(0.05, 20, 10000)

conv_fac = hbarc**2 * c * 3600 * 24 ## to give rate in cts/(GeV day)

## now calculate the velocity distribution, following https://journals.aps.org/prd/pdf/10.1103/PhysRevD.100.035025
## note there is a typo in the sign of the exponential in the above paper
N0 = np.pi**1.5 * v0**3 * ( erf(vesc/v0) - 2/np.sqrt(np.pi) * (vesc/v0) * np.exp(-(vesc/v0)**2))
def f_halo(v):
    f1 = np.exp( -(v+ve)**2/v0**2 )*(np.exp(4*v*ve/v0**2)-1)
    f2 = np.exp( -(v-ve)**2/v0**2 ) - np.exp(-vesc**2/v0**2)

    f = np.zeros_like(v)

    g1 = v < (vesc - ve)
    g2 = np.logical_and( vesc-ve < v, v < vesc + ve)
    f[g1] = f1[g1]
    f[g2] = f2[g2]

    return f * np.pi * v * v0**2/(N0 * ve)

def f_halo_dan(v):
    return 4*np.pi*v**2 * np.exp(-v**2/v0**2)/N0

vel_list = np.linspace(vmin, vesc, nvels)

for mx in mxlist:

    nX = rhoDM/mx
    #vmins = qq/(2*mx)
    #dRdq = (N_T*alpha_n)**2/(2*np.pi)/qq**3 * nX * eta(vmins) * conv_fac

    dRdq = np.zeros(len(qq))
    dRdq_in = np.zeros(len(qq))
    dRdq_mat = np.zeros((len(qq), nvels))
    dRdq_in_mat = np.zeros((len(qq), nvels))
    for vidx, vel in enumerate(vel_list):

        p = mx * vel

        sigvals = (N_T*alpha_n)**2 * 8*np.pi * 1/qq**3 * 1/vel * f_halo(vel)
        Ecm = 0.5*mx*vel**2
        k = alpha_n * N_T  ## 4pi is already in the def of alpha
        bmin = 5e-4/hbarc ## 5 um radius
        qmax = 2*mx*vel/np.sqrt(4*Ecm**2 * bmin**2/k**2 + 1)
        ## account for vmin at a given q
        sigvals[qq > 2*mx*vel] = 0

        ## cross section for point like sphere
        dRdq_mat[:,vidx] = sigvals * nX * conv_fac 
        ## cross section outside sphere only
        sigvals[qq > qmax] = 0
        dRdq_in_mat[:,vidx] = sigvals * nX * conv_fac 

    for j in range( len(qq) ):
        dRdq[j] = np.trapz(dRdq_mat[j,:], vel_list)
        dRdq_in[j] = np.trapz(dRdq_in_mat[j,:], vel_list)

    plt.figure()
    plt.semilogy( qq, dRdq)
    plt.xlim([0,5])
    plt.plot( qq, dRdq_in)
    plt.xlabel("q [GeV]")
    plt.ylabel("dR/dq [cts/(GeV day)]")
    plt.title("Mass = %.2f"%mx)
    plt.show()
