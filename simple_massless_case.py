import numpy as np
from scipy import integrate
from scipy import special
import matplotlib.pyplot as plt

pi = np.pi
# natural units
k = 1.2e-8 * 3e14
rho = 2.3e-42
Mass = 5e3
radius = 2.5e10
c = 299792458.
v0 = 220e3/c
vesc = 544e3/c
ve = 245e3/c

y_correction = (3600.*24.)/(6.58e-25) ## make counts/(GeV*day)

vmin = 0.2/(2*Mass)
vmax = (vesc + ve)*1.01

def dsdq(v,q,k):
    return 8.*pi*(k**2)/((v**2) * (q**3))

def fv(v, v0, ve, vesc):
    N0 = (pi**1.5)*(v0**3)*( special.erf(vesc/v0) - (2./pi**0.5)*(vesc/v0)*np.exp(-(vesc/v0)**2) )

    f = np.piecewise(v, [v <= vesc - ve, vesc - ve < v and v <= vesc + ve, v > vesc + ve], [(pi*v*(v0**2)/ve)*(  np.exp(-((v+ve)/v0)**2)  )*( np.exp( 4.*v*ve/v0**2 ) - 1. ),
                                                                                            (pi*v*(v0**2)/ve)*(  np.exp( -((v-ve)/v0)**2 ) - np.exp(-(vesc/v0)**2)  ),
                                                                                            0.])

    f = f/N0
    return f

def dRdq(v0, ve, vesc, vmin, vmax, rho, Mass): #### q has to be renerated on the go, depending on v


    vlist = np.linspace(vmin, vmax, 20000)
    dv = vlist[1] - vlist[0]
    l = len(vlist)
    R = 0
    for v in vlist:
        qlist = np.arange(2*Mass*vmin, 2*Mass*v, 0.01)
        r = v * fv(v, v0, ve, vesc) * dsdq(v, qlist, k)
        rextra = np.zeros(l - len(r))
        r = np.concatenate((r, rextra))
        R = r + R # this is the step that integrates in v, for each q

    R = dv*R * rho / Mass

    qextra = np.zeros(len(R) - len(qlist))
    qlist = np.concatenate((qlist, qextra))

    return [qlist, R]

# this bad version is wrong, but it is a good cross check for q in between \sim 0.5 to 1
# def dRdqbad(v0, ve, vesc, vmin, vmax, rho, Mass): #### q has to be renerated on the go, depending on v
#
#     qlist = np.linspace(1e-2, 5., 1000)
#     R = []
#     Rerr = []
#     for qq in qlist:
#         func = lambda v : v*fv(v, v0, ve, vesc) * dsdq(v, qq, k)
#         r = integrate.quad(func, vmin, vmax)
#         R.append(r[0])
#         Rerr.append(r[1])
#
#     R = np.array(R)*rho/Mass
#     Rerr = np.array(Rerr) * rho / Mass
#
#     return [R, Rerr]

R = dRdq(v0, ve, vesc, vmin, vmax, rho, Mass)

vmax2 = np.sqrt(2.*k/(Mass*radius)) # max speed so that none DM goes inside the sphere, regardless of the impact parameter
print ("vmax2 in m/s = ", vmax2*c)
print ("Pmax2 ", vmax2*Mass)
R2 = dRdq(v0, ve, vesc, vmin, vmax2, rho, Mass)

np.save("limit_case_fernando.npy", [[R[0], y_correction*R[1]], [R2[0], y_correction*R2[1]]])


plt.figure()
plt.plot(R[0], y_correction*R[1])
plt.plot(R2[0], y_correction*R2[1])
plt.xlim(0, 5)
plt.ylim(1e-6, 3e6)

plt.yscale("log")
plt.show()