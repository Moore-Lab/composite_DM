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

a = np.load("random_times_corr.npy", encoding='latin1', allow_pickle=True)

def histogram(c, bins):
    h, b = np.histogram(c, bins = bins)
    bc = np.diff(b)/2. + b[:-1]
    return [h, bc]

def gauss(x, a, b, c):
    g = c*np.exp(-0.5*((x-a)/b)**2)
    return g

def labview_time_to_datetime(lt):

    ### Convert a labview timestamp (i.e. time since 1904) to a
    ### more useful format (pytho datetime object)

    ## first get number of seconds between Unix time and Labview's
    ## arbitrary starting time
    lab_time = dt.datetime(1904, 1, 1, 0, 0, 0)
    nix_time = dt.datetime(1970, 1, 1, 0, 0, 0)
    delta_seconds = (nix_time - lab_time).total_seconds()

    if isinstance(lt, float):
        lab_dt = dt.datetime.fromtimestamp(lt - delta_seconds)
        return lab_dt

    else:
        Lab_dt = []
        for i in lt:
            lab_dt = dt.datetime.fromtimestamp(i - delta_seconds)
            Lab_dt.append(lab_dt)
        return Lab_dt

def getdata(fname, gain_error=1.0):
    ### Get bead data from a file.  Guesses whether it's a text file
    ### or a HDF5 file by the file extension

    _, fext = os.path.splitext( fname )
    f = h5py.File(fname,'r')
    dset = f['beads/data/pos_data']
    dat = np.transpose(dset)
    dat = dat / 3276.7 ## hard coded scaling from DAQ
    attribs = dset.attrs
    time = dset.attrs["Time"]

    return dat, attribs, f, time

def getcorr_time_seconds(a):
    corrin = []
    corrout = []
    T = []
    for i in range(len(a)):
        c = float(a[i][1])
        d = float(a[i][2])
        corrin.append(c)
        corrout.append(d)
        time = getdata(a[i][0])[-1] + float(a[i][2])/1e4
        T.append(time)
    return [np.array(corrin), np.array(corrout), np.array(T)]

def getcorr_timedate(a):
    corrin, corrout, T = getcorr_time_seconds(a)
    T = labview_time_to_datetime(T)
    return [corrin, corrout, T]

def plot_histogram(a, Nblocks, nbins):
    total = len(a)
    N = int(total/Nblocks)
    newcorrin = []
    newcorrout = []
    newtime = []
    i = 0
    for k in range(Nblocks):
        try:
            Blocks = a[i*N:(k+1)*N]
        except:
            Blocks = a[i*N:-1]
        i = i + 1
        corrin, corrout, time = getcorr_time_seconds(Blocks)
        newcorrin.append(corrin)
        newcorrout.append(corrout)
        newtime.append(time)

    corr_eachin = []
    corr_err_eachin = []
    corr_eachout = []
    corr_err_eachout = []
    mean_time = []

    plt.figure()
    for m in newcorrin:
        print ("len", m)
        h, bc = histogram(m, nbins)
        sigma = []
        for j in h:
            if j == 0:
                sigma.append(1.)
            else:
                sigma.append(j ** 0.5)
        popt, pcov = curve_fit(gauss, bc, h, sigma=sigma)
        corr_eachin.append(np.abs(popt[1]))
        corr_err_eachin.append(pcov[1][1]**0.5)

        plt.plot(bc, h, ".")
        energy = np.linspace(2. * np.min(bc), 2. * np.max(bc), 100)
        plt.plot(energy, gauss(energy, *popt))

    for m in newcorrout:
        h, bc = histogram(m, nbins)
        sigma = []
        for j in h:
            if j == 0:
                sigma.append(1.)
            else:
                sigma.append(j ** 0.5)
        popt, pcov = curve_fit(gauss, bc, h, sigma=sigma)
        corr_eachout.append(np.abs(popt[1]))
        corr_err_eachout.append(pcov[1][1]**0.5)

        plt.plot(bc, h, ".")
        energy = np.linspace(2. * np.min(bc), 2. * np.max(bc), 100)
        plt.plot(energy, gauss(energy, *popt))

    for n in newtime:
        mean_time.append(np.mean(n))

    mean_time = np.array(mean_time)

    mean_time = labview_time_to_datetime(mean_time)

    plt.figure()
    plt.errorbar(mean_time, corr_eachin, yerr = corr_err_eachin, fmt = ".", label = "inloop")
    plt.errorbar(mean_time, corr_eachout, yerr=corr_err_eachout, fmt=".", label="inloop")

    return []


plot_histogram(a, 24, nbins = 11)

corrin, corrout, T = getcorr_timedate(a)

hin, bcin = histogram(corrin, 15)
hout, bcout = histogram(corrout, 15)

sigmain = []
for j in hin:
    if j == 0:
        sigmain.append(1.)
    else:
        sigmain.append(j**0.5)

sigmaout = []
for j in hout:
    if j == 0:
        sigmaout.append(1.)
    else:
        sigmaout.append(j**0.5)

poptin, pcovin = curve_fit(gauss, bcin, hin, sigma = sigmain)
poptout, pcovout = curve_fit(gauss, bcout, hout, sigma = sigmaout)

energy = np.linspace(2.*np.min(bcout), 2.*np.max(bcout), 100)

plt.figure()
plt.errorbar(bcin, hin, yerr = sigmain, fmt = ".", label = "inloop")
plt.semilogy(energy, gauss(energy, *poptin))
plt.errorbar(bcout, hout, yerr = sigmaout, fmt = ".", label = "outloop")
plt.semilogy(energy, gauss(energy, *poptout))
plt.ylim(0.9, 1.4*np.max(hin))
plt.legend()

print ("sigma total inloop= ", np.abs(poptin[1]))
print ("sigma total err inloop= ", pcovin[1][1]**0.5)
print ("sigma total outloop= ", np.abs(poptout[1]))
print ("sigma total err outloop= ", pcovout[1][1]**0.5)

plt.figure()
plt.plot(T, corrin, ".", label = "inloop")
plt.plot(T, corrout, ".", label = "outloop")
plt.legend()

plt.show()

#time = labview_time_to_datetime(getdata(a[i][0])[-1] + float(a[i][2]) / 1e4)