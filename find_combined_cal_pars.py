import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.optimize import curve_fit

## find the combined paramters from both calibrations

qvals = np.linspace(0, 14, 10000)

f1 = np.load("calibration_file_20200615.npz")
f2 = np.load("calibration_file_20200619.npz")
f3 = np.load("calibration_file_20200617.npz")

rp1 = f1['reconeff_params']
rp2 = f2['reconeff_params']
rp3 = f3['reconeff_params']

rx1, ry1, re1 = f1['rx'], f1['ry'], f1['re']
rx2, ry2, re2 = f2['rx'], f2['ry'], f2['re']
rx3, ry3, re3 = f3['rx'], f3['ry'], f3['re']

def ffnerf(x, A1, mu1, sig1, A2, mu2, sig2):
    return A1*(1.+erf((x-mu1)/(np.sqrt(2.)*sig1)))/2. + A2*(1.+erf((np.log(x)-mu2)/(np.sqrt(2.)*sig2)))/2.

def ffnerf2(x, A1, mu1, sig1, A2):
    return A1*(1.+erf((x-mu1)/(np.sqrt(2.)*sig1)))/2. + A2

eff_corr_vec1 = ffnerf2( qvals, *rp1 )
eff_corr_vec2 = ffnerf2( qvals, *rp2 )
eff_corr_vec3 = ffnerf2( qvals, *rp3 )

eff_corr_vec1[eff_corr_vec1 > 1] = 1
eff_corr_vec2[eff_corr_vec2 > 1] = 1
eff_corr_vec3[eff_corr_vec3 > 1] = 1

cdat = (ry1 + ry2 + ry3)/3.
cerr = np.sqrt( re1**2 + re2**2 + re3**2 )/3.

#print(rp1)
#bpi, bci = curve_fit(ffnerf, rx1, cdat, p0 = [1.46, 0.35, 0.46, -0.46, 4.7, -0.05])
bpi, bci = curve_fit(ffnerf2, rx1, cdat, p0 = [1.46, 0.35, 0.46, 0.3])
print (bpi)

print(bpi)

np.save("combined_recon_fit.npy", bpi)

plt.figure()
plt.errorbar( rx1, ry1, yerr=re1, fmt='.')
plt.errorbar( rx2, ry2, yerr=re2, fmt='.')
plt.errorbar( rx3, ry3, yerr=re3, fmt='.')
plt.errorbar( rx2, cdat, yerr=cerr, fmt='k.')
plt.plot(qvals, ffnerf2(qvals, *bpi), 'k', label='com')
#plt.plot(qvals, eff_corr_vec1)
#plt.plot(qvals, eff_corr_vec2)
#plt.plot(qvals, (eff_corr_vec1+eff_corr_vec2)/2)


### find also the energy bias calculation
bcaly1, bcale1 = f1['bias_cal_data'], f1['bias_cal_err']
bcaly2, bcale2 = f2['bias_cal_data'], f2['bias_cal_err']
bcaly3, bcale3 = f3['bias_cal_data'], f3['bias_cal_err']

cdat = (bcaly1 + bcaly2 + bcaly3)/3
cerr = np.sqrt( bcale1 + bcale2 + bcale3)/3

## add 0 energy point from analyze_kicks:
#cdat = np.hstack((0.51, cdat))
#cerr = np.hstack((0.02, cerr))
#rx1 = np.hstack((0, rx1))

def cfit(x,A,mu,sig):
    return x + np.abs(A)*(1.+erf((mu-x)/sig))
#spars = [0.299019, 0.56726896, 0.93185983]
spars = [ 0.52721076, -0.04189387, 1.62850192]
ecbp, ecbc = curve_fit(cfit, rx1, cdat,  sigma=cerr, p0=spars,  maxfev=10000)


plt.figure()
plt.errorbar( rx2, bcaly1, yerr=bcale1, fmt='.')
plt.errorbar( rx2, bcaly2, yerr=bcale2, fmt='.')
plt.errorbar( rx2, bcaly3, yerr=bcale3, fmt='.')
plt.errorbar( rx1, cdat, yerr=cerr, fmt='k.')
plt.plot(qvals, cfit(qvals, *ecbp), 'k')

np.save("combined_recon_cal.npy", ecbp)

plt.show()
