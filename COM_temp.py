import glob, os, h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp
import matplotlib.mlab as mlab
from scipy.optimize import curve_fit
import datetime as dt
import matplotlib.dates
from scipy.special import erf
import scipy.stats
import scipy.optimize as opt

pi = np.pi

kb = 1.38e-23

Mass = 1.03e-12

Fs = 10000

important_npy = "/Volumes/My Passport for Mac/DM measurements/20200615/20200615_to/important_npy"

data = ["/Volumes/My Passport for Mac/DM measurements/20200615/20200615_to/DM_20200615",]

def get_num(s):
    num = s.split('_')[-1][:-3]
    return(float(num))

def get_measurement(fname, vtom_in, vtom_out):
    _, fext = os.path.splitext(fname)
    if (fext == ".h5"):
        print (str(fname))
        f = h5py.File(fname, 'r')
        dset = f['beads/data/pos_data']
        dat = np.transpose(dset)

        dat = dat * 10. / (2 ** 15 - 1)

    xin = dat[:, 0] - np.mean(dat[:, 0])
    xout = dat[:, 4] - np.mean(dat[:, 4])

    return [vtom_in*xin, -1.*vtom_out*xout]

def get_v_to_m_and_fressonance(folder_npy):
    namein = str(folder_npy) + "/v2tom2_in.npy"
    nameout = str(folder_npy) + "/v2tom2_out.npy"
    Lin = np.sqrt(np.load(namein, encoding='latin1'))
    Lout = np.sqrt(np.load(nameout, encoding='latin1'))
    namefress = str(folder_npy) + "/info_outloop.npy"
    fress = np.load(namefress, encoding='latin1', allow_pickle=True)[7]
    return [Lin, Lout, np.abs(fress)]

def get_gammas(folder_npy):
    name = str(folder_npy) + "/gammas.npy"
    g = np.load(name, encoding='latin1')
    return g[2] # index 2 is the combined gamma

vtom_in, vtom_out, f0 = get_v_to_m_and_fressonance(important_npy)
gam = get_gammas(important_npy)

b,a = sp.butter(3, np.array([65., 115.])/(Fs/2), btype='bandpass')

flist = []
for d in data:
    f = sorted(glob.glob(d + "/*.h5"), key=get_num)
    for ff in f:
        flist.append(ff)

def merge_xout(flist):
    xout = get_measurement(flist[0], vtom_in, vtom_out)[1]
    aux1, f = mlab.psd(np.real(xout), Fs=Fs, NFFT=2 ** 18)
    xoutpsd = np.zeros(len(aux1))
    j = 0.
    for i in flist[20:22]: #20:22
        xout = get_measurement(i, vtom_in, vtom_out)[1]
        aux, f = mlab.psd(np.real(xout), Fs=Fs, NFFT = 2 ** 18)
        xoutpsd = xoutpsd + aux
        j = j + 1.
    xoutpsd = 1.*xoutpsd/j

    return [f, np.sqrt(xoutpsd)]


def harmonic2(f, A, C):
    f0 = get_v_to_m_and_fressonance(important_npy)[2]
    gamma = get_gammas(important_npy)
    w0 = 2.*np.pi*np.abs(f0)
    w = 2.*np.pi*f
    gamma = 2.0*np.pi*gamma

    a1 = 1.*np.abs(A)
    a3 = 1.*(w0**2 - w**2)**2 + (w*gamma)**2

    s = 1.*a1/a3

    return np.sqrt(s) + C

f, psd = merge_xout(flist)



fit_points1 = np.logical_and(f > 0, f < 41.)
fit_points2 = np.logical_and(f > 54.5, f < 54.7)
fit_points3 = np.logical_and(f > 55.18, f < 55.44)
fit_points4 = np.logical_and(f > 56.5, f < 64.5)
fit_points5 = np.logical_and(f > 72.6, f < 73.0)
fit_points6 = np.logical_and(f > 72.6, f < 73.51)
fit_points7 = np.logical_and(f > 74.3, f < 74.7)
fit_points8 = np.logical_and(f > 75.6, f < 76.0)
fit_points9 = np.logical_and(f > 78.7, f < 79.5)
fit_points10 = np.logical_and(f > 80.21, f < 81.46)
fit_points11 = np.logical_and(f > 82.66, f < 83.55)
fit_points12 = np.logical_and(f > 86.0, f < 86.4)
fit_points13 = np.logical_and(f > 88.9, f < 89.2)

fit_points14 = np.logical_and(f > 96.94, f < 97.4)


fit_points15 = np.logical_and(f > 108, f < 109)
fit_points16 = np.logical_and(f > 118, f < 5000)


fit_points_n1 = np.logical_and(f > 39.5, f < 39.9)



fit = fit_points1 + fit_points2 + fit_points3 + fit_points4 + fit_points5 + \
      fit_points6 + fit_points7 + fit_points8 + fit_points9 + fit_points10 + \
      fit_points11 + fit_points12 + fit_points13 + fit_points14 + fit_points15 + \
      fit_points16 + fit_points_n1
fit = np.logical_not(fit)

p0 = np.array([5e-13, 0.5e-11])
popt, pcov = opt.curve_fit(harmonic2, f[fit][1:-1], psd[fit][1:-1], p0 = p0)#, bounds=([5e-14, 0.5e-11], [1e-12, 2e-11]))

print (popt)


plt.figure()
plt.loglog(f, psd)
plt.loglog(f[fit][1:-1], psd[fit][1:-1])
plt.loglog(f, harmonic2(f, *popt))

print (gam)

Temp = np.abs(popt[0]) * Mass / (2. * kb * (2.*np.pi*gam))
Temperr_fit = np.abs(pcov[0][0]**0.5) * Mass / (2. * kb * (2.*np.pi*gam))

print("temp [uK]:", 1e6 * Temp)
print("temperr_fit [uK]:", 1e6 * Temperr_fit)

plt.show()