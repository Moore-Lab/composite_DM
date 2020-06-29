import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.optimize import curve_fit

## find the combined paramters from both calibrations

qvals = np.linspace(0.05, 100, 10000)

f1 = np.load("calibration_file_20200615.npz")
f2 = np.load("calibration_file_20200619.npz")

rp1 = f1['reconeff_params']
rp2 = f2['reconeff_params']
rx1, ry1, re1 = f1['rx'], f1['ry'], f1['re']
rx2, ry2, re2 = f2['rx'], f2['ry'], f2['re']

def ffnerf(x, A1, mu1, sig1, A2, mu2, sig2):
    return A1*(1+erf((x-mu1)/(np.sqrt(2)*sig1)))/2 + A2*(1+erf((np.log(x)-mu2)/(np.sqrt(2)*sig2)))/2

eff_corr_vec1 = ffnerf( qvals, *rp1 )
eff_corr_vec2 = ffnerf( qvals, *rp2 )

eff_corr_vec1[eff_corr_vec1 > 1] = 1
eff_corr_vec2[eff_corr_vec2 > 1] = 1

cdat = (ry1 + ry2)/2
cerr = np.sqrt( re1**2 + re2**2)/2

print(rp1)
bpi, bci = curve_fit(ffnerf, rx1, cdat, sigma=cerr, p0=rp1)

np.save("combined_recon_fit.npy", bpi)

plt.figure()
plt.errorbar( rx1, ry1, yerr=re1, fmt='b.')
plt.errorbar( rx2, ry2, yerr=re2, fmt='r.')
plt.errorbar( rx2, cdat, yerr=cerr, fmt='k.')
plt.plot(qvals, ffnerf(qvals, *bpi), 'k', label='com')
#plt.plot(qvals, eff_corr_vec1)
#plt.plot(qvals, eff_corr_vec2)
#plt.plot(qvals, (eff_corr_vec1+eff_corr_vec2)/2)
plt.show()
