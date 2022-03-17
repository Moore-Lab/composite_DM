import glob, os, h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp
import matplotlib.mlab as mlab
from scipy.optimize import curve_fit
import datetime as dt
import matplotlib.dates
from statsmodels.stats.proportion import proportion_confint
import matplotlib.cm as cmx
import matplotlib.colors as colors
from scipy.special import erf
from scipy.interpolate import UnivariateSpline

a = np.load("dm_data_paper.npz", encoding='latin1', allow_pickle=True)
lcf = np.load("limit_case_fernando.npy", encoding='latin1', allow_pickle=True)

bc = a["dp"]
h = a["cts"]
expo = a["expo"]

def gauss(x, a, b, c):
    g = c*np.exp(-0.5*((x-a)/b)**2)
    return g

#good = np.logical_and(bc > 0.0, bc < 1.5)

sigma = []
for j in h:
    if j == 0:
        sigma.append(1.)
    else:
        sigma.append((j/expo)**0.5)
popt, pcov = curve_fit(gauss, bc, h/expo, sigma = sigma, p0 = [0.001, 0.14, 2e5])


print (popt)
print (pcov)


#xx = np.load("drdq_massless_1.2e-8.npy", encoding='latin1', allow_pickle=True)

#print (xx)

plt.figure()
plt.errorbar( bc, 20*h/expo, yerr=20*np.sqrt(h)/expo, fmt='.')
plt.plot(bc, 20.*gauss(bc, *popt))
plt.plot(lcf[0][0], lcf[0][1])
plt.plot(lcf[1][0], lcf[1][1])
plt.yscale('log')

plt.legend()
plt.ylim(1e-3, 5e6)
plt.xlim(0, 5)

plt.figure()
plt.errorbar( bc, (h/expo - gauss(bc, *popt)), yerr=np.sqrt(h)/expo, fmt='k.')
plt.yscale('log')

plt.show()