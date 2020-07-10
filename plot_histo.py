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

a = np.load("dm_data.npz", encoding='latin1', allow_pickle=True)

bc = a["dp"]
h = a["cts"]
expo = a["expo"]

def gauss(x, a, b, c):
    g = c*np.exp(-0.5*((x-a)/b)**2)
    return g

sigma = []
for j in h:
    if j == 0:
        sigma.append(1.)
    else:
        sigma.append((j/expo)**0.5)
popt, pcov = curve_fit(gauss, bc[1:], h[1:]/expo, sigma = sigma[1:], p0 = [0.01, 0.14, 2e5])

print (popt)
print (pcov)

plt.figure()
plt.errorbar( bc, h/expo, yerr=np.sqrt(h)/expo, fmt='k.')
plt.plot(bc, gauss(bc, *popt))
plt.yscale('log')
plt.ylim(0.1, 6e5)
plt.xlim(0, 5)

#plt.figure()
#plt.errorbar( bc, h/expo - gauss(bc, *popt), yerr=np.sqrt(h)/expo, fmt='k.')
#plt.yscale('log')

plt.show()