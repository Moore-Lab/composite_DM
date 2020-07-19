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

# the ecbp is fixed down below, this is a clear choice that accomodates well the data and dont affect the limit plot by a visible amount

distance = 3.99e-3 # in m
disterr = 0.05*1e-3 # in m
Fs = 1e4
mass = 1.03e-12 # kg
#flen = 524288  # file length, samples
SI_to_GeV = 1.87e18
tthr = 0.050 ## time threshold in s for which to look for coincidences with calibration pulses (this is big to get random rate)
repro = True # Set true to reprocess data, false to read from file
Fernando_path = False
calculate_index = False # use true only if change filter or index...

Make_npy_FIG1 = False # use it as false for calibration, true for figure for the paper

calibration_date = "20200617"

if Fernando_path:
    data_list = ["/Volumes/My Passport for Mac/DM measurements/" + calibration_date + "/kick/0.1ms/0.1V",
                 "/Volumes/My Passport for Mac/DM measurements/" + calibration_date + "/kick/0.1ms/0.2V",
                 "/Volumes/My Passport for Mac/DM measurements/" + calibration_date + "/kick/0.1ms/0.4V",
                 "/Volumes/My Passport for Mac/DM measurements/" + calibration_date + "/kick/0.1ms/0.8V",
                 "/Volumes/My Passport for Mac/DM measurements/" + calibration_date + "/kick/0.1ms/1.6V",
                 "/Volumes/My Passport for Mac/DM measurements/" + calibration_date + "/kick/0.1ms/3.2V",
                 "/Volumes/My Passport for Mac/DM measurements/" + calibration_date + "/kick/0.1ms/6.4V"]
    if Make_npy_FIG1:
        data_list = ["/Volumes/My Passport for Mac/DM measurements/20200615/kick/0.1ms/3.2V",]

    path1 = "/Volumes/My Passport for Mac/DM measurements/" + calibration_date + "/important_npy"
    path2 = path1
else:
    data_list = ["data/"+calibration_date+"_to/kick/0.1ms/0.1V",
                 "data/"+calibration_date+"_to/kick/0.1ms/0.2V",
                "data/"+calibration_date+"_to/kick/0.1ms/0.4V",
                "data/"+calibration_date+"_to/kick/0.1ms/0.8V",
                "data/"+calibration_date+"_to/kick/0.1ms/1.6V",
                "data/"+calibration_date+"_to/kick/0.1ms/3.2V",
                 "data/"+calibration_date+"_to/kick/0.1ms/6.4V"]
    # data_list = ["/Users/dcmoore/Desktop/0.1ms/0.1V",
    #              "/Users/dcmoore/Desktop/0.1ms/0.2V",
    #              "/Users/dcmoore/Desktop/0.1ms/0.4V",
    #              "/Users/dcmoore/Desktop/0.1ms/0.8V",
    #              "/Users/dcmoore/Desktop/0.1ms/1.6V",
    #              "/Users/dcmoore/Desktop/0.1ms/3.2V",
    #              "/Users/dcmoore/Desktop/0.1ms/6.4V"]
    path1 = "data/20200615_to/calibration1e_HiZ_20200615"
    #path1 = "/Users/dcmoore/Desktop/important_npy"
    path2 = "data"
    #path2 = path1

# data_list = ["data/20200619/kick/0.1ms/0.1V",
#              "data/20200619/kick/0.1ms/0.2V",
#              "data/20200619/kick/0.1ms/0.4V",
#              "data/20200619/kick/0.1ms/0.8V",
#              "data/20200619/kick/0.1ms/1.6V",
#              "data/20200619/kick/0.1ms/3.2V",
#              "data/20200619/kick/0.1ms/6.4V"]

# data_list = ["data/20200619/kick/0.1ms/0.1V",
#              "data/20200619/kick/0.1ms/0.2V",
#              "data/20200619/kick/0.1ms/0.4V",
#              "data/20200619/kick/0.1ms/0.8V",
#              "data/20200619/kick/0.1ms/1.6V",
#              "data/20200619/kick/0.1ms/3.2V",
#              "data/20200619/kick/0.1ms/6.4V"]

def histogram(c, bins):
    h, b = np.histogram(c, bins = bins)
    bc = np.diff(b)/2. + b[:-1]
    return [h, bc]

def gauss(x, a, b, c):
    g = c*np.exp(-0.5*((x-a)/b)**2)
    return g

# DO NOT USE eleminate_noisy_peaks as it removes real signals
def eleminate_noisy_peaks(timestream, plot):
    fft_ = np.fft.fft(timestream)
    freq = np.fft.fftfreq(len(timestream), 1./Fs)
    fft2 = np.abs(fft_)**2
    if True:
        fit_points1 = np.logical_and(freq > -5001, freq < 40.)
        fit_points2 = np.logical_and(freq > 54.5, freq < 54.7)
        fit_points3 = np.logical_and(freq > 55.18, freq < 55.44)
        fit_points4 = np.logical_and(freq > 56.5, freq < 64.5)
        fit_points5 = np.logical_and(freq > 72.6, freq < 73.0)
        fit_points6 = np.logical_and(freq > 72.6, freq < 73.51)
        fit_points7 = np.logical_and(freq > 74.3, freq < 74.7)
        fit_points8 = np.logical_and(freq > 75.6, freq < 76.0)
        fit_points9 = np.logical_and(freq > 78.7, freq < 79.5)
        fit_points10 = np.logical_and(freq > 80.21, freq < 81.46)
        fit_points11 = np.logical_and(freq > 82.66, freq < 83.55)
        fit_points12 = np.logical_and(freq > 86.0, freq < 86.4)
        fit_points13 = np.logical_and(freq > 88.9, freq < 89.2)
        fit_points14 = np.logical_and(freq > 96.94, freq < 97.4)
        fit_points15 = np.logical_and(freq > 108, freq < 109)
        fit_points16 = np.logical_and(freq > 118, freq < 5001)
        fit_points_n1 = np.logical_and(freq > 39.5, freq < 39.9)
        fit = fit_points1 + fit_points2 + fit_points3 + fit_points4 + fit_points5 + \
          fit_points6 + fit_points7 + fit_points8 + fit_points9 + fit_points10 + \
          fit_points11 + fit_points12 + fit_points13 + fit_points14 + fit_points15 + \
          fit_points16 + fit_points_n1
        fit = np.logical_not(fit)
    p0 = np.array([5e-13, (1e-11) ** 2, 30.])
    popt, pcov = curve_fit(harmonic3, freq[fit][1:-1], fft2[fit][1:-1], p0=p0)

    aux = np.abs(harmonic3(freq, *popt) - fft2)/harmonic3(freq, *popt)

    newfft2 = []
    newfft = []
    for i in range(len(aux)):
        if 1./aux[i] < 0.1:
            newfft2.append(0.)
            newfft.append(0.)
        else:
            newfft2.append(fft2[i])
            newfft.append(fft_[i])

    newfft2 = np.array(newfft2)
    newtimestream = np.fft.ifft(np.array(newfft))
    newtimestream = np.real(newtimestream)

    if plot:
        plt.figure()
        plt.semilogy(freq, fft2)
        plt.semilogy(freq, 1./aux)
        plt.semilogy(freq, harmonic3(freq, *popt))
        plt.figure()
        plt.loglog(freq, fft2)
        plt.loglog(freq, newfft2)
        plt.figure()
        plt.plot(timestream)
        plt.plot(newtimestream)
        plt.show()

    return newtimestream # #

if not calculate_index:
    precalculated_index_badfreq = np.load("important_index_badfreq_calibration.npy")

def eleminate_noisy_peaks_nofit(timestream, plot, calculate_index):
    fft_ = np.fft.fft(timestream)
    freq = np.fft.fftfreq(len(timestream), 1./Fs)

    if True:
        fit_points1 = np.logical_and(freq > 39., freq < 40.)
        fit_points2 = np.logical_and(freq > 54.5, freq < 54.7)
        fit_points3 = np.logical_and(freq > 55.18, freq < 55.44)
        fit_points4 = np.logical_and(freq > 56.3, freq < 57.5)
        fit_points4_2 = np.logical_and(freq > 59.33, freq < 64.5)
        fit_points5 = np.logical_and(freq > 72.6, freq < 73.0)
        fit_points6 = np.logical_and(freq > 72.6, freq < 73.3)
        fit_points7 = np.logical_and(freq > 74.3, freq < 74.7)
        fit_points8 = np.logical_and(freq > 75.6, freq < 76.0)
        fit_points9 = np.logical_and(freq > 78.85, freq < 79.05)
        fit_points9_2 = np.logical_and(freq > 79.25, freq < 79.4)
        fit_points10 = np.logical_and(freq > 80.21, freq < 80.58)
        fit_points10_2 = np.logical_and(freq > 80.9, freq < 81.2)
        fit_points11 = np.logical_and(freq > 82.75, freq < 83.05)
        fit_points11_2 = np.logical_and(freq > 83.23, freq < 83.55)
        fit_points12 = np.logical_and(freq > 86.1, freq < 86.25)
        fit_points13 = np.logical_and(freq > 88.9, freq < 89.2)
        # fit_points14 = np.logical_and(freq > 96.94, freq < 97.4)
        fit_points15 = np.logical_and(freq > 108, freq < 109)
        fit_points16 = np.logical_and(freq > 118, freq < 122.)
        fit_points17 = np.logical_and(freq > 24, freq < 33)
        fit_points18 = np.logical_and(freq > 7.4, freq < 8.5)
        fit_points19 = np.logical_and(freq > 20.4, freq < 20.9)
        fit_points20 = np.logical_and(freq > 172., freq < 181.)
        fit_points21 = np.logical_and(freq > 11.26, freq < 11.31)
        fit_points22 = np.logical_and(freq > 48.6, freq < 49.01)
        fit_points_n1 = np.logical_and(freq > 39.5, freq < 39.9)

        fit = fit_points1 + fit_points2 + fit_points3 + fit_points4 + fit_points5 + \
              fit_points6 + fit_points7 + fit_points8 + fit_points9 + fit_points10 + \
              fit_points11 + fit_points12 + fit_points13 + fit_points15 + \
              fit_points16 + fit_points17 + fit_points18 + fit_points19 + fit_points20 + fit_points21 + fit_points_n1 + \
              fit_points4_2 + fit_points9_2 + fit_points10_2 + fit_points11_2 + fit_points22

    badfreq = np.concatenate((freq[fit], -freq[fit]))
    if plot:
        plt.figure()
        plt.loglog(freq, np.abs(fft_))

    newfft = fft_

    if calculate_index:
        index = [i for i, x in enumerate(freq) if x in badfreq]
        np.save("important_index_badfreq_calibration.npy", index)
    else:
        index = precalculated_index_badfreq

    newfft[index] = np.zeros(len(index))

    newtimestream = np.fft.ifft(newfft)
    newtimestream = np.real(newtimestream)

    if plot:
        plt.loglog(freq, np.abs(newfft))
        plt.figure()
        plt.plot(timestream)
        plt.plot(newtimestream)
        plt.show()

    return [newtimestream, index]

def eleminate_noisy_peaks_nofit_templateonly(template, plot, index, lenmeasurement):

    lentemp = len(template)
    pad = np.zeros(lenmeasurement - lentemp)
    template = np.concatenate((template, pad))

    if plot:
        plt.figure()
        plt.plot(template)

    fft_ = np.fft.fft(template)
    freq = np.fft.fftfreq(len(template), 1./Fs)

    if plot:
        plt.figure()
        plt.loglog(freq, np.abs(fft_))

    newfft = fft_

    newfft[index] = np.zeros(len(index))

    if plot:
        plt.loglog(freq, np.abs(newfft))

    newtemplate = np.fft.ifft(newfft)
    newtemplate = np.real(newtemplate)
    newtemplate = newtemplate[0:lentemp]

    if plot:
        plt.figure()
        plt.plot(newtemplate, label = "new")
        plt.legend()
        plt.show()

    return newtemplate

def get_color_map( n ):
    jet = plt.get_cmap('jet') 
    cNorm  = colors.Normalize(vmin=0, vmax=n-1)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    outmap = []
    for i in range(n):
        outmap.append( scalarMap.to_rgba(i) )
    return outmap


def getdata(fname, gain_error=1.0):
    ### Get bead data from a file.  Guesses whether it's a text file
    ### or a HDF5 file by the file extension

    _, fext = os.path.splitext( fname )
    if( fext == ".h5"):
        try:
            f = h5py.File(fname,'r')
            dset = f['beads/data/pos_data']
            dat = np.transpose(dset)
            dat = dat / 3276.7 ## hard coded scaling from DAQ
            attribs = dset.attrs

        except (KeyError, IOError):
            print("Warning, got no keys for: ", fname)
            dat = []
            attribs = {}
            f = []
    else:
        dat = np.loadtxt(fname, skiprows = 5, usecols = [2, 3, 4, 5])
        attribs = {}
        f = []

    return dat, attribs, f

def get_num(s):
    num = s.split('_')[-1][:-3]
    return(float(num))

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

def make_template(Fs, f0, gamma_total, dp, Mass):
    w0 = 2. * np.pi * f0
    g = 2. * np.pi * (gamma_total/2)
    #time_template = np.arange(-flen/(2*Fs), flen/(2*Fs), 1./Fs)
    time_template = np.arange(-1.5/gamma_total, 1.5/gamma_total, 1./Fs)
    w1 = np.sqrt(w0**2 - g**2)
    a = (dp / (Mass * w1)) * np.exp(-time_template * g) * np.sin(w1 * time_template)
    a[time_template<0] = 0
    return [a, time_template]

#vtom_in, vtom_out, f0 = get_v_to_m_and_fressonance("data")
vtom_in, vtom_out, f0 = get_v_to_m_and_fressonance(path1)

## fix cal for now
#vtom_in *= np.sqrt(2)
#vtom_out *= np.sqrt(2)

gam = get_gammas(path2)
temp = make_template(Fs, f0, gam, 1, mass)

#tempt = np.hstack((np.zeros(500), temp[0]))
tempt = temp[0]
b,a = sp.butter(3, np.array([10., 190.])/(Fs/2), btype='bandpass')
#if Make_npy_FIG1:
#    b,a = sp.butter(3, np.array([2., 1200.])/(Fs/2), btype='bandpass')
b2,a2 = sp.butter(3, (f0/2)/(Fs/2), btype='lowpass')
tempf = sp.filtfilt(b,a,tempt)

if not calculate_index:
    cdaux = getdata((glob.glob(data_list[0] + "/*.h5")[0]))
    dataux = cdaux[0]
    lenofmeas = len(dataux[:, 0])
    print (lenofmeas)
    tempf = eleminate_noisy_peaks_nofit_templateonly(tempf, False, precalculated_index_badfreq, lenofmeas)

normf = np.sum( tempf**2 )
#tempt #/= np.sum( tempt**2 )
tempf /= normf

bstop,astop = sp.butter(3, np.array([65., 115.])/(Fs/2), btype='bandstop')
bstop2,astop2 = sp.butter(3, 400./(Fs/2), btype='lowpass')

#tt = np.arange(-flen/(2*Fs), flen/(2*Fs), 1./Fs)
tt = np.arange(-1.5/gam, 1.5/gam, 1./Fs)
tempt = make_template(Fs, f0, gam, 9.6/SI_to_GeV, mass)[0]
tempt = sp.filtfilt(b,a,tempt)
#if not calculate_index and Make_npy_FIG1:
#    tempt = eleminate_noisy_peaks_nofit_templateonly(tempt, False, precalculated_index_badfreq, lenofmeas)

#chi2pts = np.logical_and( tt >= 0, tt<1.5/gam) 
#chi2tempf = tempf[chi2pts]*normf/SI_to_GeV

# def get_chi2(a, d, idx):
#     ll = len(chi2tempf)

#     dec_fac = 1 #int(Fs/(4*115))

#     wfdat = np.roll(d, -idx)[:ll][::dec_fac]

#     chi_neg = np.sum( (wfdat + a*chi2tempf[::dec_fac])**2 )
#     chi_pos = np.sum( (wfdat - a*chi2tempf[::dec_fac])**2 )
    
#     if(True):
#         plt.figure()
#         npts = len(d)
#         tvec = np.linspace(0, (npts-1)/Fs, npts)
#         #plt.plot( tvec[::dec_fac], d[::dec_fac], 'k' )
#         plt.plot( tvec[idx:(idx+ll)][::dec_fac], d[idx:(idx+ll)][::dec_fac], 'bo' )
#         if( chi_neg < chi_pos ):
#             plt.plot( tvec[idx:(idx+ll)], -chi2tempf*a, 'r' )
#         else:
#             plt.plot( tvec[idx:(idx+ll)], chi2tempf*a, 'r' )
#         plt.show()

#     return np.min( (chi_neg, chi_pos) )

#s = np.fft.rfft(tempt)
#norm = np.sum( np.abs(s)**2 )
#nmat, mmat = np.meshgrid( np.linspace(-flen/2, flen/2-1, flen), np.linspace(-flen/2, flen/2-1, flen) )
#print(nmat, mmat)
#phasemat = 

#tempf /= np.sum( tempf**2 )

# plt.figure()
# plt.plot(tempt)
# plt.plot(tempf)
# plt.show()

print(vtom_in, vtom_out, f0, gam )

flist = []
for d in data_list:
    f = sorted(glob.glob(d + "/*.h5"), key=get_num)
    for ff in f:
        flist.append(ff)
        
if(repro):
    ## now load each file and process the data for kicks
    joint_peaks = []
    file_offset = 0
    rand_cts = []
    for fi, f in enumerate(flist[::1]):
        
        print(f)
        cd = getdata(f)

        dat = cd[0]

        ## get the real time of the file
        timestamp = cd[1]['Time']
        fparts = f.split("V/")[0]
        fparts = fparts.split("/")[-1]
        volts = float( fparts )
        
        npts = len(dat[:,0])
        #print(npts)
        tvec = np.linspace(0, (npts-1)/Fs, npts)

        # if(fi == 1):
        #     oldi_psd, oldi_freqs = mlab.psd(dat[:,0], Fs=Fs, NFFT=int(npts/16))
        #     oldo_psd, oldo_freqs = mlab.psd(dat[:,4], Fs=Fs, NFFT=int(npts/16))

        indat = dat[:,0]*vtom_in
        indat -= np.mean(indat)
        outdat = -1.*dat[:,4]*vtom_out ## here I am using - to correct for the 180 phase in the outloop
        outdat -= np.mean(outdat)
        accdat = dat[:,5]*1e-7
        accdat -= np.mean(accdat)

        indatf = sp.filtfilt(b,a,indat)
        outdatf = sp.filtfilt(b,a,outdat)
        accdatf = sp.filtfilt(b,a,accdat)

        indatf_outband = np.std(sp.filtfilt(bstop,astop,indat))
        outdatf_outband = np.std(sp.filtfilt(bstop,astop,outdat))
        indatf2_outband = np.std(sp.filtfilt(bstop2,astop2,sp.filtfilt(bstop,astop,indat)))
        outdatf2_outband = np.std(sp.filtfilt(bstop2,astop2,sp.filtfilt(bstop,astop,outdat)))

        # def harmonic3(f, A, C, gamma):
        #     f0 = get_v_to_m_and_fressonance(path1)[2]
        #     w0 = 2. * np.pi * np.abs(f0)
        #     w = 2. * np.pi * f
        #     gamma = 2.0 * np.pi * gamma
        #     # at this point gamma is \Gamma_0 (latex notation)
        #     a1 = 1. * np.abs(A)
        #     a3 = 1. * (w0 ** 2 - w ** 2) ** 2 + (w * gamma) ** 2
        #
        #     s = 1. * a1 / a3
        #
        #     return np.sqrt(s + C)

        indatf, indexbad = eleminate_noisy_peaks_nofit(indatf, False, calculate_index)
        outdatf, indexbad = eleminate_noisy_peaks_nofit(outdatf, False, calculate_index)
        
        incorr = sp.correlate(indatf, tempf, mode='same')
        outcorr = sp.correlate(outdatf, tempf, mode='same')

        ## now the local maxima
        incorrf = sp.filtfilt(b2,a2,np.sqrt(incorr**2))
        outcorrf = sp.filtfilt(b2,a2,np.sqrt(outcorr**2))

        # plt.figure()
        # plt.plot(tvec, indatf)
        # plt.plot(tvec, outdatf)
        # plt.plot(tvec, accdatf/10)

        ## fudge factor of 1.1 accounts for amplitude loss in filter frequency for finding envelope
        ## don't worry about the fudge factor, it's fine...
        incorrf *= 1.1*SI_to_GeV * np.sqrt(2)
        outcorrf *= 1.1*SI_to_GeV * np.sqrt(2)
        inpeaks_orig = sp.find_peaks(incorrf)[0]
        outpeaks_orig = sp.find_peaks(outcorrf)[0]

        ## now find the true peak in the correlation closest to the smoothed peak
        inpeaks = np.zeros_like(inpeaks_orig)
        outpeaks = np.zeros_like(outpeaks_orig)        

        ## 2 ms shouldn't be hardcoded -- depends on the period of the oscillation (sphere res frequency)
        naround = int(Fs * 1./(2.*f0))
        for j,ip in enumerate(inpeaks_orig):
            if( ip<500 or ip>npts-500 ):
                inpeaks[j] = 0
            else:
                inpeaks[j] = np.argmax( np.abs(incorr[(ip-naround):(ip+naround)]) ) - naround + ip
        for j,ip in enumerate(outpeaks_orig):
            if( ip<500 or ip>npts-500 ):
                outpeaks[j] = npts-1
            else:
                outpeaks[j] = np.argmax( np.abs(outcorr[(ip-naround):(ip+naround)]) ) - naround + ip 
        
        ## now step through the calibration spikes and look for peaks
        mon = dat[:,3]
        ## pstart gives time of the spikes in the monitor signal
        pstart = np.argwhere( np.logical_and(mon<0.1, np.roll(mon,-1)>=0.05) ).flatten()

        pspace = pstart[1]-pstart[0]
        ## keep track of random coincidences
        random_cts_in = 0
        random_cts_out = 0
        ## skip beginning and end of files to prevent edge effects (maybe revisit if this is long enough??)
        for ip in pstart:
            if( ip<1000 or ip>npts-1000): continue
            time_diffs_ip = tvec[inpeaks] - tvec[ip]
            time_diffs_op = tvec[outpeaks] - tvec[ip]

            ## check for the closest outloop peak to the ipp
            #opp_closest = tvec[
            
            ##### in loop sensor only
            ## find the largest peak within the cal window for each
            if(np.any(np.abs(time_diffs_ip) < tthr)):
                #ip_peakvals = np.abs(incorr[inpeaks][np.abs(time_diffs_ip) < tthr])*SI_to_GeV
                ip_peakvals = incorr[inpeaks][np.abs(time_diffs_ip) < tthr] * SI_to_GeV
                ip_peaktimes = time_diffs_ip[np.abs(time_diffs_ip) < tthr]
                ipp, ipt = np.max(ip_peakvals), ip_peaktimes[ np.argmax(ip_peakvals) ] #amplitude and time of pulse

                ## now get the chi2 to the pulse template
                ip_s = inpeaks[np.abs(time_diffs_ip) < tthr][ np.argmax(ip_peakvals) ]
                pdat = indatf[int(ip_s - len(tempf)/2):int(ip_s + len(tempf)/2)]
                ichi2a = np.sum( (pdat - tempf*ipp*normf/SI_to_GeV)**2 )
                ichi2b = np.sum( (pdat + tempf*ipp*normf/SI_to_GeV)**2 )
                ichi2 = np.min([ichi2a, ichi2b])
                
                if(False):
                    plt.close('all')

                    plt.figure()
                    plt.plot(pdat)
                    plt.plot(tempf*ipp*normf/SI_to_GeV)
                    plt.title( "min chi2, %.2e, best idx, %d"%(ichi2, 0) )
                    plt.show()
                    
            else:
                ipp, ipt, ichi2 = np.nan, np.nan, np.nan
            
            ### done with inloop

            ##outloop sensor only
            if(np.any(np.abs(time_diffs_op) < tthr)):
                op_peakvals = outcorr[outpeaks][np.abs(time_diffs_op) < tthr]*SI_to_GeV
                #op_peakvals = np.abs(outcorr[outpeaks][np.abs(time_diffs_op) < tthr]) * SI_to_GeV
                op_peaktimes = time_diffs_op[np.abs(time_diffs_op) < tthr]
                opp, opt = np.max(op_peakvals), op_peaktimes[ np.argmax(op_peakvals) ]

                ## now get the chi2 to the pulse template
                op_s = outpeaks[np.abs(time_diffs_op) < tthr][ np.argmax(op_peakvals) ]
                pdat = outdatf[int(op_s - len(tempf)/2):int(op_s + len(tempf)/2)]
                ochi2a = np.sum( (pdat - tempf*opp*normf/SI_to_GeV)**2 )
                ochi2b = np.sum( (pdat + tempf*opp*normf/SI_to_GeV)**2 )
                ochi2 = np.min([ochi2a, ochi2b])

                if(False):
                    plt.close('all')

                    plt.figure()
                    plt.plot(pdat)
                    plt.plot(tempf*opp*normf/SI_to_GeV)
                    plt.title( "min chi2, %.2e, best idx, %d"%(ichi2, 0) )
                    plt.show()
                
            else:
                opp, opt, ochi2 = np.nan, np.nan, np.nan
            ### done with outloop
            
            
            ## variables: time in seconds of event, inloop amp, inloop time, outloop amp, outloop time, calibration voltage, inloop chi2, outloop chi2
            joint_peaks.append( [file_offset+tvec[ip], ipp, ipt, opp, opt, volts, ichi2, ochi2] )
            #print( joint_peaks[-1] )
            
            # time_diffs_ip = tvec[inpeaks] - tvec[(ip+pspace)%npts]
            # time_diffs_op = tvec[outpeaks] - tvec[(ip+pspace)%npts]
            # if(np.any(np.abs(time_diffs_ip) < tthr)):
            #     random_cts_in += 1
            # if(np.any(np.abs(time_diffs_op) < tthr)):
            #     random_cts_out += 1
            
        file_offset += npts/Fs
        #rand_cts.append( [random_cts_in, random_cts_out, volts] )

        ## plot waveforms for debugging only
        if( False ): # and np.any( incorrf[inpeaks] > 1.5 ) ):
            plt.figure()
            plt.plot(tvec, np.abs(incorr)*SI_to_GeV)
            plt.plot(tvec, np.abs(outcorr)*SI_to_GeV)
            plt.plot(tvec, outcorrf, 'g')
            plt.plot(tvec[outpeaks_orig], outcorrf[outpeaks_orig], 'go', mfc='none')
            plt.plot(tvec[outpeaks], np.abs(outcorr[outpeaks])*SI_to_GeV, 'go')
            plt.plot(tvec, incorrf, 'r')
            plt.plot(tvec[inpeaks_orig], incorrf[inpeaks_orig], 'ro', mfc='none')
            plt.plot(tvec[inpeaks], np.abs(incorr[inpeaks])*SI_to_GeV, 'ro')

            pstart = np.argwhere( np.logical_and(mon<0.1, np.roll(mon,-1)>=0.1) ).flatten()
            mon *= np.max(incorrf[inpeaks])/np.max(mon)

            plt.plot( tvec, mon, 'k')

            plt.figure()
            plt.plot(tvec, (incorr)*SI_to_GeV)
            plt.plot(tvec, (outcorr)*SI_to_GeV)
            plt.plot(tvec, outcorrf, 'g')
            plt.plot(tvec[outpeaks_orig], outcorrf[outpeaks_orig], 'go', mfc='none')
            plt.plot(tvec[outpeaks], (outcorr[outpeaks])*SI_to_GeV, 'go')
            plt.plot(tvec, incorrf, 'r')
            plt.plot(tvec[inpeaks_orig], incorrf[inpeaks_orig], 'ro', mfc='none')
            plt.plot(tvec[inpeaks], (incorr[inpeaks])*SI_to_GeV, 'ro')

            mon = dat[:,3]
            pstart = np.argwhere( np.logical_and(mon<0.1, np.roll(mon,-1)>=0.1) ).flatten()
            mon *= np.max(incorrf[inpeaks])/np.max(mon)

            #plt.figure()
            plt.plot( tvec, mon, 'k')
            
            plt.figure()
            plt.plot( tvec, outdatf)
            plt.plot( tvec, indatf)
      

            #stack up the kicks
            stackdati = np.zeros(680)
            stackdato = np.zeros(680)
            nstack = 0
            for p in pstart:
                if(p+480 > npts): break
                stackdati += indatf[(p-200):(p+480)]
                stackdato += outdatf[(p-200):(p+480)]
                nstack += 1
            stackdati /= nstack
            stackdato /= nstack

            plt.figure()
            plt.plot( tvec[:680]-0.02, stackdati, label='in loop'  )
            plt.plot( tvec[:680]-0.02, stackdato, label='out of loop' )
            #plt.plot( tt+0.5e-3, tempt, label="Fernando's template" )
            plt.plot( tt[120:]+0.5e-3, tempt[120:]/2, 'k:', label="Template" )
            plt.legend(loc="upper right")

            if Make_npy_FIG1:
                np.save("fig1_info_filter_2to1201.npy", [stackdati, -stackdato, tvec[:680]-0.02, tempt, tt+0.5e-3, tvec, indatf, outdatf, mon])

            #newi_psd, newi_freqs = mlab.psd(dat[:,0], Fs=Fs, NFFT=int(npts/16))
            #newo_psd, newo_freqs = mlab.psd(dat[:,4], Fs=Fs, NFFT=int(npts/16))
            
            # plt.figure()
            # plt.loglog(oldi_freqs, oldi_psd)
            # plt.loglog(newi_freqs, newi_psd)
            # plt.figure()
            # plt.loglog(oldo_freqs, oldo_psd)
            # plt.loglog(newo_freqs, newo_psd)
            
            plt.show()

    joint_peaks = np.array(joint_peaks)
    rand_cts = np.array(rand_cts)
    np.savez("calibration_joint_peaks.npz", joint_peaks=joint_peaks, rand_cts=rand_cts)
else:
    jdat = np.load("calibration_joint_peaks.npz")
    joint_peaks = jdat['joint_peaks']
    rand_cts = jdat['rand_cts']

# plt.figure()
# #plt.plot( joint_peaks[:,1], joint_peaks[:,3], 'k.')
# plt.plot( joint_peaks[:,-1], joint_peaks[:,1], 'k.')
# plt.plot( joint_peaks[:,-1], joint_peaks[:,3], 'r.')



### analyze the calibration data to find amplitude calibration, cut efficiency, etc:

vlist = np.array([0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4])
eff_list = np.zeros((len(vlist),6))
mean_list = np.zeros((len(vlist),6))

fig_in = plt.figure()
fig_out = plt.figure()
def ffn(x,A,mu,sig,C):
    return A*np.exp(-(x-mu)**2/(2*sig**2)) + C

cols = get_color_map(len(vlist))

## list of kicks in GeV/c (what we expect)
gev_list = 200*vlist*1e-4/(distance)*(1.6e-19) * SI_to_GeV

gev_errl = 200*vlist*1e-4/(distance + disterr)*(1.6e-19) * SI_to_GeV
gev_errh = 200*vlist*1e-4/(distance - disterr)*(1.6e-19) * SI_to_GeV

print("cal amplitudes")
print(gev_list)
print((gev_errh-gev_errl)/2)

for i,v in enumerate(vlist):
    print("Working on volts: ", v)
    cdat = joint_peaks[joint_peaks[:,5] == v, :]

    ## get the mean value for each voltage of the kick amplitude (before final calibration)
    tot_cts = len(cdat[:,1])
    #gpts1 = cdat[:,1] > 0
    #gpts2 = cdat[:,3] > 0
    gpts1 = np.logical_not(np.isnan(cdat[:,1])) ##### -
    gpts2 = np.logical_not(np.isnan(cdat[:,3])) ##### -

    std1 = np.std( cdat[gpts1,1] )/np.sqrt(np.sum(gpts1))
    std2 = np.std( cdat[gpts2,3] )/np.sqrt(np.sum(gpts2))
    res1 = np.std( cdat[gpts1,1] )
    res2 = np.std( cdat[gpts2,3] )
    mean_list[i,:] = [ np.median(cdat[gpts1,1]), np.median(cdat[gpts2,3]), std1, std2, res1, res2 ]

    good_cts_in, good_cts_out = np.sum( cdat[:,1] >0), np.sum( cdat[:,3] >0)


    ## find trigger efficiency and random coincident rate (in loop)
    plt.figure(fig_in.number)
    hh, be = np.histogram( cdat[:,2], range=(-tthr,tthr), bins=150 )
    bc = be[:-1]+np.diff(be)/2
    blpts = np.logical_or(bc < -0.007, bc > 0.008)
    baseline = np.mean(hh[blpts])
    plt.errorbar(bc, hh-baseline, yerr=np.sqrt(hh), fmt='o', label=str(gev_list[i]), color=cols[i])
    ## now gauss + const fit
    errs = np.sqrt(hh)
    errs[errs==0]=0.1
    spars = [300, 0.0005, 0.0001, 0.5] 
    #bp, bcov = curve_fit(ffn, bc, hh, sigma=errs, p0=spars )
    bp = spars
    xx = np.linspace(-tthr,tthr, 1000)
    #plt.plot(xx, ffn(xx,*bp), color=cols[i])
    corr_fac = np.trapz( ffn(xx, *[bp[0],bp[1],bp[2],0]), xx )/np.trapz( ffn(xx, *bp ),xx)
    good_cts_in = np.sum( hh[np.logical_not(blpts)]-baseline )
    

    ## find trigger efficiency and random coincident rate (out loop)
    plt.figure(fig_out.number)
    hh, be = np.histogram( cdat[:,4], range=(-tthr,tthr), bins=150 )
    bc = be[:-1]+np.diff(be)/2
    blpts = np.logical_or(bc < -0.007, bc > 0.008)
    baseline = np.mean(hh[blpts])
    plt.errorbar(bc, hh-baseline, yerr=np.sqrt(hh), fmt='o', label=str(gev_list[i]), color=cols[i])
    ## now gauss + const fit
    errs = np.sqrt(hh)
    errs[errs==0]=0.1
    spars = [300, 0.0005, 0.0001, 0.5] 
    #bp, bcov = curve_fit(ffn, bc, hh, sigma=errs, p0=spars )
    bp = spars
    xx = np.linspace(-tthr,tthr, 1000)
    #plt.plot(xx, ffn(xx,*bp), color=cols[i])
    corr_fac2 = np.trapz( ffn(xx, *[bp[0],bp[1],bp[2],0]), xx )/np.trapz( ffn(xx, *bp ),xx)
    good_cts_out = np.sum( hh[np.logical_not(blpts)]-baseline )
    
    print("Efficiency correction is: ", corr_fac)
    corr_fac, corr_fac2 = 1, 1
    
    eff_in, eff_out = (good_cts_in*corr_fac)/tot_cts, (good_cts_out*corr_fac2)/tot_cts
    
    min_conf_in,max_conf_in = proportion_confint(int(round(good_cts_in*corr_fac)),tot_cts,alpha=0.32,method='beta')
    min_conf_out,max_conf_out = proportion_confint(int(round(good_cts_out*corr_fac2)),tot_cts,alpha=0.32,method='beta')

    eff_list[i,:] = [eff_in, eff_in-min_conf_in, max_conf_in-eff_in, eff_out, eff_out-min_conf_out, max_conf_out-eff_out]
    
plt.legend()
#plt.show()


## find the calibration to the pulse amplitude
plt.figure()
corr_fac_in = np.mean( mean_list[-3:,0]/gev_list[-3:] )
corr_fac_out = np.mean( mean_list[-3:,1]/gev_list[-3:] )

## now fit center and sigma of blob
def cfit(x,A,mu,sig):
    return x + A*(1.+erf((mu-x)/sig))

xx = np.linspace(0, gev_list[-1]*1.2, 1000)

plt.errorbar( gev_list, mean_list[:,0]/corr_fac_in, yerr=mean_list[:,2]/corr_fac_in, fmt='k.')
plt.errorbar( gev_list, mean_list[:,1]/corr_fac_out, yerr=mean_list[:,3]/corr_fac_out, fmt='r.')
plt.plot(gev_list, gev_list, 'k:')
plt.title("Corr fac inloop: %.2f, corr fac outloop: %.2f"%(corr_fac_in, corr_fac_out))
plt.xlabel("Calibration pulse amplitude [GeV]")
plt.ylabel("Recontructed amplitude")

bcaly, bcale = mean_list[:,0]/corr_fac_in, mean_list[:,2]/corr_fac_in

## fit calibraion to get search bias at low energy
spars = [0.299019, 0.56726896, 0.93185983]
ecbp, ecbc = curve_fit(cfit, gev_list, mean_list[:,0]/corr_fac_in,  p0=spars,  maxfev=10000)
#ecbp = [ 0.52721076, -0.04189387, 1.62850192] # this is what david finds
plt.plot( xx, cfit(xx, *ecbp), 'k')
#fern = [0.33882171, 0.19736197, 0.95576589] # this is what fernando finds and it is hard coded in the analysis code too. Both david and fernando number result in similar results.
#ecbp = fern
plt.plot( xx, cfit(xx, *ecbp), 'k:')

print("Energy cal params: ", ecbp)


## find the resolution at each kick value
plt.figure()
plt.errorbar( gev_list, mean_list[:,4]/corr_fac_in, yerr=0, fmt='k.')
plt.errorbar( gev_list, mean_list[:,5]/corr_fac_out, yerr=0, fmt='r.')
plt.xlabel("Calibration pulse amp [GeV]")
plt.ylabel("Resolution [GeV]")



cbp, cbc = curve_fit(cfit, joint_peaks[:,1]/corr_fac_in, joint_peaks[:,3]/corr_fac_out, p0=[1,1,1])
#cbp = [0.1,1,1]

## calculate the efficiency of the in/out loop amplitude matching criterion
plt.figure()
#xvals, yvals = joint_peaks[:,1]/corr_fac_in, joint_peaks[:,3]/corr_fac_out
# the for loop below is to remove negative values for when the code fails to find the peak at any of two detectors.
xvals = []
yvals = []
for i in range(len(joint_peaks[:,1])):
    if joint_peaks[:,1][i] > 0. and joint_peaks[:,3][i]  > 0.:
        xvals.append(joint_peaks[:,1][i] /corr_fac_in)
        yvals.append(joint_peaks[:,3][i] /corr_fac_out)
xvals = np.array(xvals)
yvals = np.array(yvals)

plt.plot( xvals, yvals, 'k.', ms=1)
#plt.plot( gev_list, gev_list, 'r:')
plt.plot( xx, cfit(xx, *cbp), 'r:')

sigval = np.std( yvals - cfit(xvals, *cbp) )
plt.fill_between( xx, cfit(xx, *cbp)-2*sigval, cfit(xx, *cbp)+2*sigval, facecolor='r', alpha=0.1)

cut_eff = np.sum( np.abs(yvals-cfit(xvals,*cbp))<2*sigval )/len(yvals)
cut_eff_err = np.sqrt(np.sum( np.abs(yvals-cfit(xvals,*cbp))<2*sigval ))/len(yvals)
plt.title("Cut efficiency = %.3f $\pm$ %.3f "%(cut_eff,cut_eff_err))
print("Sigma: ", sigval)


plt.xlabel("Reconstructed in loop amp [GeV]")
plt.ylabel("Reconstructed out of loop amp [GeV]")



def ffnerf(x, A1, mu1, sig1, A2, mu2, sig2):
    return A1*(1+erf((x-mu1)/(np.sqrt(2)*sig1)))/2 + A2*(1+erf((np.log(x)-mu2)/(np.sqrt(2)*sig2)))/2

def ffnerf2(x, A1, mu1, sig1, A2):
    return A1*(1.+erf((x-mu1)/(np.sqrt(2.)*sig1)))/2. + A2

#spars=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
#spars = [ 0.70557531, 0.51466432, 0.51597654, 0.29442469, 1.07858784, 0.3547105 ]
spars = [1.83226494, 0.19640354, 0.56620851, -0.83184023]
xo,yo,xeo = gev_list, eff_list[:,0], (eff_list[:,1]+eff_list[:,2])/2
bpi, bci = curve_fit(ffnerf2, gev_list, eff_list[:,0], p0=spars)
#bpi = spars
bpo, bco = curve_fit(ffnerf2, gev_list, eff_list[:,3], p0=spars)
#bpo = spars

print (bpi)
print (bpo)

spli = UnivariateSpline(gev_list, eff_list[:,0],w=2/(eff_list[:,1]+eff_list[:,2]))
splo = UnivariateSpline(gev_list, eff_list[:,3],w=2/(eff_list[:,4]+eff_list[:,5]))

plt.figure()
plt.errorbar( gev_list, eff_list[:,0], yerr=(eff_list[:,1],eff_list[:,2]), fmt='k.')
plt.plot(xx, ffnerf2(xx, *bpi), 'k', label='in loop')
#plt.plot(xx, spli(xx), 'k')
plt.errorbar( gev_list, eff_list[:,3], yerr=(eff_list[:,4],eff_list[:,5]), fmt='r.')
plt.plot(xx, ffnerf2(xx, *bpo), 'r', label='out of loop')
#plt.plot(xx, splo(xx), 'r')

plt.plot(xx, ffnerf2(xx, *bpi)*ffnerf2(xx, *bpo), 'b', label='combined')

plt.xlabel("Impulse amplitude [GeV]")
plt.ylabel("Reconstruction efficiency")
plt.legend()

print("In loop recon eff params: ", bpi)

#### done calculating in/out loop amplitude matching efficiency


#### new chi2 inloop
def func_x2(x,a,c):
    return np.abs(a) + np.abs(c)*x**2
gpts = np.logical_and( joint_peaks[:,2]>-0.001, joint_peaks[:,2]<0.0015 )

sigma2_5y = []
Meanx = []
for i in vlist:
    yyy = []
    xxx = []
    for j in range(len(joint_peaks[gpts,5])):
        if i == joint_peaks[gpts,5][j]:
            yyy.append(joint_peaks[gpts,6][j])
            xxx.append(joint_peaks[gpts,1][j]/corr_fac_in)
    yyy = np.array(yyy)
    xxx = np.array(xxx)
    Meanx.append(np.mean(xxx))
    # h, bc = histogram(yyy, 12)
    # sigma = []
    # for qq in h:
    #     if qq == 0:
    #         sigma.append(1.)
    #     else:
    #         sigma.append(qq ** 0.5)
    # if i > 0.2:
    #     ppo, cco = curve_fit(gauss, bc, h, p0 = [4e-17, 3e-18, 5], sigma = sigma)
    #     #sigma2_5y.append(np.abs(ppo[0]) + 2.5 * np.abs(ppo[1]))
    #     sigma2_5y.append(np.mean(yyy) + 2.5 * np.std(yyy))
    # else:
    #     sigma2_5y.append(np.mean(yyy)+2.5*np.std(yyy))
    for m in range(len(yyy)):
        if yyy[m] > np.mean(yyy) + 3.5 * np.std(yyy):
            yyy[m] = 0
    sigma2_5y.append(np.mean(yyy) + 2.5 * np.std(yyy))
    # plt.figure()
    # plt.plot(bc, h, ".", label = str(i))
    # rrr = np.linspace(0.9*np.min(bc), 1.1*np.max(bc), 100)
    # plt.plot(rrr, gauss(rrr, *ppo))
    # plt.legend()

print(Meanx, sigma2_5y)
poptchi2, pcovchi2 = curve_fit(func_x2, Meanx[2:], sigma2_5y[2:])
pin = [np.abs(poptchi2[1]),0,np.abs(poptchi2[0])]

## finally, make the chi2 cut:
plt.figure()
#plt.plot( joint_peaks[:,1], joint_peaks[:,6], 'k.', ms=1)
plt.plot( joint_peaks[gpts,1]/corr_fac_in, joint_peaks[gpts,6], 'k.', ms=1)
plt.plot( Meanx , sigma2_5y, 'ro')
plt.plot(xx, func_x2(xx, *poptchi2))
#plt.plot(xx, np.polyval(pin, xx))
#plt.plot( joint_peaks[gpts,1]/corr_fac_in, func_x2(joint_peaks[gpts,1]/corr_fac_in, *poptchi2)  )

#pin = [0.2e-18, 0, 2.5e-17]
#plt.plot(xx, np.polyval(pin, xx), 'r')
gptscut = np.logical_and( gpts, joint_peaks[:,5] > 0)
pass_cut = joint_peaks[gptscut,6] < func_x2(joint_peaks[gptscut,1]/corr_fac_in, *poptchi2)

##### inloop chi2 ended

#### new chi2 outloop

sigma2_5y = []
Meanx = []
for i in vlist:
    yyy = []
    xxx = []
    for j in range(len(joint_peaks[gpts,5])):
        if i == joint_peaks[gpts,5][j]:
            yyy.append(joint_peaks[gpts,7][j])
            xxx.append(joint_peaks[gpts,3][j]/corr_fac_out)
    yyy = np.array(yyy)
    xxx = np.array(xxx)
    Meanx.append(np.mean(xxx))
    # h, bc = histogram(yyy, 7)
    # sigma = []
    # for qq in h:
    #     if qq == 0:
    #         sigma.append(1.)
    #     else:
    #         sigma.append(qq ** 0.5)
    # if i > 0.2:
    #     ppo, cco = curve_fit(gauss, bc, h, p0 = [3e-17, 3e-18, 5], sigma = sigma)
    #     sigma2_5y.append(np.abs(ppo[0]) + 2.5 * np.abs(ppo[1]))
    # else:
    #     sigma2_5y.append(np.mean(yyy)+2.5*np.std(yyy))

    for m in range(len(yyy)):
        if yyy[m] > np.mean(yyy) + 3.5 * np.std(yyy):
            yyy[m] = 0

    sigma2_5y.append(np.mean(yyy) + 2.5 * np.std(yyy))
    # plt.figure()
    # plt.plot(bc, h, ".", label = str(i))
    # rrr = np.linspace(0.9*np.min(bc), 1.1*np.max(bc), 100)
    # plt.plot(rrr, gauss(rrr, *ppo))
    # plt.legend()
print(Meanx, sigma2_5y)
poptchi2, pcovchi2 = curve_fit(func_x2, Meanx[2:], sigma2_5y[2:])
pout = [np.abs(poptchi2[1]),0,np.abs(poptchi2[0])]

plt.figure()
plt.plot( joint_peaks[gpts,3]/corr_fac_out, joint_peaks[gpts,7], 'k.', ms=1)
#plt.plot(xx, np.polyval(pout, xx), 'r')
plt.plot( Meanx , sigma2_5y, 'ro')
plt.plot(xx, func_x2(xx, *poptchi2))
pass_cut = np.logical_and( pass_cut , joint_peaks[gptscut,7] < func_x2(joint_peaks[gptscut,3]/corr_fac_out, *poptchi2) )

##### outloop chi2 ended

print (pout, pin)

print("Cut efficiency, out+in: ", np.sum(pass_cut)/len(pass_cut))

chi2eff = np.sum(pass_cut)/len(pass_cut)

## now save all parameters to a calibration file
eng_cal_pars = [corr_fac_in, corr_fac_out]
np.savez("calibration_file_"+calibration_date+".npz", reconeff_params=bpi, chi2eff=chi2eff, bias_cal_params=ecbp, eng_cal_pars=eng_cal_pars, amp_match_eff=cut_eff, chi2cut_pars_in=pin, chi2cut_pars_out=pout, rx=xo,ry=yo,re=xeo, bias_cal_data=bcaly, bias_cal_err=bcale, sigval = sigval)

plt.show()
