import glob, os, h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp
import matplotlib.mlab as mlab
from scipy.optimize import curve_fit
import datetime as dt
import matplotlib.dates
from scipy.special import erf

Fs = 1e4
mass = 9.4e-13 # kg
flen = 524288  # file length, samples
SI_to_GeV = 1.87e18
tthr = 0.005 ## time threshold in s
repro = False
remake_coinc_cut = False

data_list = ["data/DM_20200615","data/DM_20200617","data/DM_20200619"]
eng_cal_in = [1.04, 1.04, 1.04]
eng_cal_out = [1.05, 1.05, 1.12]

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
vtom_in, vtom_out, f0 = get_v_to_m_and_fressonance("data/20200615_to/calibration1e_HiZ_20200615")
print("f0: ",f0)

## fix cal for now
vtom_in *= np.sqrt(2) ##/1.04
vtom_out *= np.sqrt(2) ##/1.05

gam = get_gammas("data")
temp = make_template(Fs, f0, gam, 1, mass)

#tempt = np.hstack((np.zeros(500), temp[0]))
tempt = temp[0]
b,a = sp.butter(3, np.array([65., 115.])/(Fs/2), btype='bandpass')
b2,a2 = sp.butter(3, (f0/2)/(Fs/2), btype='lowpass')
tempf = sp.filtfilt(b,a,tempt)

normf = np.sum( tempf**2 )
tempt #/= np.sum( tempt**2 )
tempf /= normf

bstop,astop = sp.butter(3, np.array([65., 115.])/(Fs/2), btype='bandstop')
bstop2,astop2 = sp.butter(3, 400./(Fs/2), btype='lowpass')

#tt = np.arange(-flen/(2*Fs), flen/(2*Fs), 1./Fs)
tt = np.arange(-1/gam, 1.5/gam, 1./Fs)
tempt = make_template(Fs, f0, gam, 11.6/SI_to_GeV, mass)[0]
#chi2pts = np.logical_and( tt >= 0, tt<1.5/gam) 
#chi2tempf = tempf[chi2pts]*normf/SI_to_GeV

# def get_chi2(a, d, idx):
#     ll = len(chi2tempf)

#     dec_fac = int(Fs/(4*115))

#     wfdat = np.roll(d, -idx)[:ll][::dec_fac]

#     chi_neg = np.sum( (wfdat + a*chi2tempf[::dec_fac])**2 )
#     chi_pos = np.sum( (wfdat - a*chi2tempf[::dec_fac])**2 )
    
#     if(False and a > 5):
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
ecilist = []
ecolist = []
for eci, eco, d in zip(eng_cal_in, eng_cal_out, data_list):
    f = sorted(glob.glob(d + "/*.h5"), key=get_num)
    for ff in f:
        flist.append(ff)
        ecilist.append(eci)
        ecolist.append(eco)
        
if(repro):
    ## now load each file and process the data for kicks
    joint_peaks = []
    file_offset = 0
    for fi, f in enumerate(flist[::1]):
        
        print(f)
        cd = getdata(f)

        dat = cd[0]

        ## get the real time of the file
        timestamp = cd[1]['Time']
        
        npts = len(dat[:,0])
        #print(npts)
        tvec = np.linspace(0, (npts-1)/Fs, npts)

        # if(fi == 1):
        #     oldi_psd, oldi_freqs = mlab.psd(dat[:,0], Fs=Fs, NFFT=int(npts/16))
        #     oldo_psd, oldo_freqs = mlab.psd(dat[:,4], Fs=Fs, NFFT=int(npts/16))
        
        indat = dat[:,0]*vtom_in/ecilist[fi]
        indat -= np.mean(indat)
        outdat = dat[:,4]*vtom_out/ecolist[fi]
        outdat -= np.mean(outdat)
        accdat = dat[:,5]*1e-7
        accdat -= np.mean(accdat)

        indatf = sp.filtfilt(b,a,indat)
        outdatf = sp.filtfilt(b,a,outdat)
        accdatf = sp.filtfilt(b,a,accdat)
        accdatf2 = sp.filtfilt(b2,a2,accdat)
        
        ## make a couple accelerometer related quantities
        accmax, accstd = np.max(np.abs(accdat)), np.std(accdat)
        accmaxf, accstdf = np.max(np.abs(accdatf)), np.std(accdatf)
        accmaxf2, accstdf2 = np.max(np.abs(accdatf2)), np.std(accdatf2)  
        
        indatf_outband = np.std(sp.filtfilt(bstop,astop,indat))
        outdatf_outband = np.std(sp.filtfilt(bstop,astop,outdat))
        indatf2_outband = np.std(sp.filtfilt(bstop2,astop2,sp.filtfilt(bstop,astop,indat)))
        outdatf2_outband = np.std(sp.filtfilt(bstop2,astop2,sp.filtfilt(bstop,astop,outdat)))
        
        incorr = sp.correlate(indatf, tempf, mode='same')
        outcorr = sp.correlate(outdatf, tempf, mode='same')

        ## now the local maxima
        incorrf = sp.filtfilt(b2,a2,np.sqrt(incorr**2))
        outcorrf = sp.filtfilt(b2,a2,np.sqrt(outcorr**2))

        # plt.figure()
        # plt.plot(tvec, indatf)
        # plt.plot(tvec, outdatf)
        # plt.plot(tvec, accdatf/10)

        incorrf *= 1.1*np.sqrt(2)*SI_to_GeV
        outcorrf *= 1.1*np.sqrt(2)*SI_to_GeV
        inpeaks_orig = sp.find_peaks(incorrf)[0]
        outpeaks_orig = sp.find_peaks(outcorrf)[0]

        ## now find the true peak in the correlation closest to the smoothed peak
        inpeaks = np.zeros_like(inpeaks_orig)
        outpeaks = np.zeros_like(outpeaks_orig)  
        naround = int(Fs * 0.002)
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
        

        for ip in inpeaks:
            if( ip<500 or ip>npts-500): continue

            time_diffs = np.abs(tvec[outpeaks] - tvec[ip])
            mop = np.argmin( time_diffs )
            min_diff = np.min(time_diffs)

            if min_diff < tthr:
                pdat = indatf[int(ip - len(tempf)/2):int(ip + len(tempf)/2)]
                ipp = np.abs(incorr[ip])
                ichi2a = np.sum( (pdat - tempf*ipp*normf)**2 )
                ichi2b = np.sum( (pdat + tempf*ipp*normf)**2 )
                chisq_in = np.min([ichi2a, ichi2b])

                if(False and ipp > 1.0/SI_to_GeV):
                    plt.close('all')

                    plt.figure()
                    plt.plot(pdat)
                    plt.plot(tempf*ipp*normf)
                    plt.title( "min chi2, %.2e, best idx, %d"%(chisq_in, 0) )
                    plt.show()
                
                op_s = outpeaks[mop]
                opp = np.abs(outcorr[op_s])
                pdat = outdatf[int(op_s - len(tempf)/2):int(op_s + len(tempf)/2)]
                ochi2a = np.sum( (pdat - tempf*opp*normf)**2 )
                ochi2b = np.sum( (pdat + tempf*opp*normf)**2 )
                chisq_out = np.min([ochi2a, ochi2b])

                if(False and opp > 1.0/SI_to_GeV):
                    plt.close('all')

                    plt.figure()
                    plt.plot(pdat)
                    plt.plot(tempf*opp*normf)
                    plt.title( "min chi2, %.2e, best idx, %d"%(chisq_out, 0) )
                    plt.show()                
                    
                joint_peaks.append( [file_offset+tvec[ip], ipp*SI_to_GeV, opp*SI_to_GeV, chisq_in, chisq_out, timestamp+tvec[ip], accmax, accstd, accmaxf, accstdf, accmaxf2, accstdf2])

        file_offset += npts/Fs
        
        if( False and np.any( incorrf[inpeaks] > 2.0 ) ):
            plt.figure()
            plt.plot(tvec, np.abs(incorr)*SI_to_GeV)
            plt.plot(tvec, np.abs(outcorr)*SI_to_GeV)
            plt.plot(tvec, outcorrf, 'g')
            plt.plot(tvec[outpeaks], outcorrf[outpeaks], 'go')
            plt.plot(tvec, incorrf, 'r')
            plt.plot(tvec[inpeaks], incorrf[inpeaks], 'ro') 
            
            plt.figure()
            plt.plot( tvec, outdatf)
            plt.plot( tvec, indatf)
            plt.plot( tvec, accdatf)
            plt.plot( tvec, accdat/10)
            
            plt.show()

    joint_peaks = np.array(joint_peaks)
    np.save("joint_peaks.npy", joint_peaks)
else:
    joint_peaks = np.load("joint_peaks.npy")

sig = 0.2818451718216727 ## from 20200616cal
gpts = np.abs( joint_peaks[:,1] - joint_peaks[:,2] ) < 2*sig

xe = np.linspace(0,8,1e2)
ye = np.linspace(0,4e-17,1e2)


def gauss_fit(x, A, mu, sig):
    return A*np.exp( -(x-mu)**2/(2*sig**2) )

def log_gauss_fit(x, A, mu, sig):
    return A/x*np.exp( -(np.log(x)-mu)**2/(2*sig**2) )

def total_fit(x,A1,mu1,sig1,A2,mu2,sig2):
    return gauss_fit(x,A1,mu1,sig1) + log_gauss_fit(x,A2,mu2,sig2)


## accelerometer cuts

accel_cols = [6,8,9]
accel_cut = np.zeros_like( joint_peaks[:,6] ) > 1
for acol in accel_cols:
    ha, ba = np.histogram( joint_peaks[:,acol], bins=50 )
    bac = ba[:-1] + np.diff(ba)/2
    plt.figure()
    plt.errorbar( bac, ha, yerr=np.sqrt(ha), fmt='k.')
    mu = bac[np.argmax(ha)]
    sig = 0.5*mu
    yerrs=np.sqrt(ha)
    yerrs[0] = 1
    fpts = ha > 0.2*np.max(ha)
    bap, bacov = curve_fit( gauss_fit, bac[fpts], ha[fpts], sigma=yerrs[fpts], p0=[5e6, mu, sig])
    #bap = [5e6, mu, sig]
    bb = np.linspace(bac[0], bac[-1], 1e3)
    plt.plot( bb, gauss_fit(bb, *bap), 'r')
    cut_val = bap[1] + 2.5*np.abs(bap[2])
    yy = plt.ylim()
    plt.plot([cut_val, cut_val], yy, 'k:')

    accel_cut = np.logical_or( accel_cut,  joint_peaks[:,acol]>cut_val )

    
#plt.figure()
#hh, xbe, ybe = np.histogram2d( joint_peaks[:,1], joint_peaks[:,3], bins=(xe,ye) )
#plt.pcolormesh(xbe, ybe, np.log10(hh.T), cmap='Greys', vmin=0, vmax=2)
#plt.colorbar()

#plt.figure()
#plt.plot( joint_peaks[:,1], joint_peaks[:,3], 'k.', ms=1)
#plt.plot( joint_peaks[:,1], joint_peaks[:,5], 'b.', ms=1)

#plt.figure()
#plt.plot( joint_peaks[:,1], joint_peaks[:,4], 'k.', ms=1)
#plt.plot( joint_peaks[:,1], joint_peaks[:,6], 'b.', ms=1)


# ## automated bad times cut -- make a list of counts per minute above 0.7
# nsecs = 1
# num_min = joint_peaks[-1,0]/nsecs
# ## throw out 1 min bins
# min_bins = np.linspace(0, num_min, (num_min+1) )
# hmin, bmin = np.histogram( joint_peaks[joint_peaks[:,1]>0.95,0]/nsecs, bins=min_bins )

# print("Mean cts/min: ", np.median(hmin))


# plt.figure()
# mc = bmin[:-1] + np.diff(bmin)/2
# plt.errorbar( mc, hmin, yerr=np.sqrt(hmin), fmt='k.' )
# bad_times = np.zeros_like(joint_peaks[:,0]) > 1

exposure = joint_peaks[-1,0]
exp_orig = 1.0*exposure

print("Exposure: ", exposure)
# exp_orig = exposure
# for j in range(len(bmin)-1):
#     if hmin[j] > 1.1:
#         bad_times = np.logical_or(bad_times, np.logical_and(bmin[j]*nsecs<joint_peaks[:,0],bmin[j+1]*nsecs>joint_peaks[:,0]))
#         exposure -= nsecs


# plt.figure()
# plt.plot( joint_peaks[:,1], joint_peaks[:,2], 'k.', label='all data', ms=2)
# #plt.plot( joint_peaks[bad_times,1], joint_peaks[bad_times,2], 'r.', label='high rate times', ms=2)
# plt.plot( joint_peaks[gpts,1], joint_peaks[gpts,2], 'b.', label='passing cuts', ms=4)
# plt.legend()
# plt.xlim([0,10])
# plt.ylim([0,10])
# plt.xlabel("In loop reconstructed momentum [GeV]")
# plt.ylabel("Out of loop reconstructed momentum [GeV]")
# plt.show()

#plt.close('all')

## I hate labview:
def labview_time_to_datetime(lt):
    ### Convert a labview timestamp (i.e. time since 1904) to a 
    ### more useful format (pytho datetime object)
    
    ## first get number of seconds between Unix time and Labview's
    ## arbitrary starting time
    lab_time = dt.datetime(1904, 1, 1, 0, 0, 0)
    nix_time = dt.datetime(1970, 1, 1, 0, 0, 0)
    delta_seconds = (nix_time-lab_time).total_seconds()

    lab_dt = dt.datetime.fromtimestamp( lt - delta_seconds)
    
    return lab_dt


plt.figure()
dates = []
for d in joint_peaks[:,5]:
    dates.append(labview_time_to_datetime(d))
npd = matplotlib.dates.date2num(dates)

lab_cut = np.zeros_like(npd) > 1
lab_entry = [ #[dt.datetime(2020,6,8,9,10,0), dt.datetime(2020,6,8,16,47,0), "Fernando"],
              #[dt.datetime(2020,6,9,8,0,0), dt.datetime(2020,6,9,10,0,0), "Gadi"],
              #[dt.datetime(2020,6,9,11,0,0), dt.datetime(2020,6,9,14,0,0), "Jiaxiang"],
              #[dt.datetime(2020,6,9,15,50,0), dt.datetime(2020,6,9,16,0,0), "Fernando"],
              #[dt.datetime(2020,6,9,16,30,0), dt.datetime(2020,6,9,19,45,0), "Jiaxiang"],
              #[dt.datetime(2020,6,11,11,20,0), dt.datetime(2020,6,11,11,35,0), "Fernando"],
              #[dt.datetime(2020,6,11,15,30,0), dt.datetime(2020,6,11,18,00,0), "Gadi"],
              #[dt.datetime(2020,6,12,10,15,0), dt.datetime(2020,6,12,10,30,0), "Gadi"],]
              [dt.datetime(2020,6,15,10,0,0), dt.datetime(2020,6,15,16,0,0), "Fernando"],
              [dt.datetime(2020,6,16,6,20,0), dt.datetime(2020,6,16,11,30,0), "Gadi"],
              [dt.datetime(2020,6,16,16,10,0), dt.datetime(2020,6,16,17,0,0), "Jiaxiang"],
              [dt.datetime(2020,6,16,21,30,0), dt.datetime(2020,6,16,23,50,0), "Gadi"],
              [dt.datetime(2020,6,17,10,0,0), dt.datetime(2020,6,17,11,20,0), "Fernando"],
              [dt.datetime(2020,6,17,15,45,0), dt.datetime(2020,6,17,17,0,0), "Gadi"],
              [dt.datetime(2020,6,18,14,40,0), dt.datetime(2020,6,18,19,30,0), "Gadi"],
              #[dt.datetime(2020,6,19,1,0,0), dt.datetime(2020,6,19,5,0,0), "??"],
              [dt.datetime(2020,6,19,10,0,0), dt.datetime(2020,6,19,15,5,0), "Gadi"],
              [dt.datetime(2020,6,19,15,20,0), dt.datetime(2020,6,19,16,20,0), "Fernando"],
              [dt.datetime(2020,6,21,16,0,0), dt.datetime(2020,6,21,17,6,0), "Fernando"],
              ]

plt.plot_date(npd[::10], joint_peaks[::10,1], 'k.', ms=2, label='all data')
#plt.plot_date(npd[bad_times], joint_peaks[bad_times,1], 'r.', ms=2, label='high rate')
yy = plt.ylim()
for ll in lab_entry:
    dsin = matplotlib.dates.date2num(ll[0])
    dsout = matplotlib.dates.date2num(ll[1])
    plt.fill_between( [dsin, dsout], [yy[0], yy[0]], [yy[1],yy[1]], facecolor='b', alpha=0.1 )
    plt.text( dsin, 0.9*yy[1], ll[2], color='b')
    lab_cut = np.logical_or( lab_cut, np.logical_and(npd>=dsin, npd<=dsout) )

    ## correct the exposure
    cut_pts = np.argwhere(np.logical_and(npd>=dsin, npd<=dsout)).flatten()
    if( len(cut_pts) > 0 ):
        exp_cut = (npd[cut_pts[-1]]-npd[cut_pts[0]]) * (24*3600)
        print("Cutting ", exp_cut, " s for ", ll[2])
    else:
        exp_cut = 0
    exposure -= exp_cut
plt.legend()

print("Exposure after lab entry cut: ", exposure)
print("Remaing fraction Exposure: ", exposure/exp_orig)

curr_exposure = exposure

#anti-coincidence cut -- require all events to be >1s from another above 1 GeV/c
high_peaks = np.argwhere( np.logical_and(joint_peaks[:,1] > 1.0, np.logical_not(lab_cut) )).flatten()
time_list = joint_peaks[:,0]
time_peaks = time_list[high_peaks]
bad_times = np.zeros_like(joint_peaks[:,0]) > 1

if(remake_coinc_cut):
    last_t2 = 0
    print("Working on %d peaks:"%len(time_peaks))
    for titer,tp in enumerate(time_peaks):

        if(titer % 5000 == 0): print(titer)

        if(tp < last_t2): continue

        ## as a double check, make sure we are past the last window
        if( tp-last_t2 < 1.0):
            t1 = last_t2
            t2 = tp + 0.1
            bad_times = np.logical_or( bad_times, np.logical_and( time_list >= t1, time_list<=t2 ) )
            last_t2 = t2
            exposure -= (t2-t1)
            continue
            
        time_diffs = np.abs(time_peaks - tp)
        coinc_peaks = np.argwhere( np.logical_and( time_diffs > 0, time_diffs<1.001 ) ).flatten()
        if( len(coinc_peaks) > 0 ):
            t1 = tp - 0.1
            t2 = time_peaks[coinc_peaks[-1]] + 0.1
            bad_times = np.logical_or( bad_times, np.logical_and( time_list >= t1, time_list<=t2 ) )
            last_t2 = t2
            exposure -= (t2-t1)

    gpts2 = np.logical_not(bad_times)
    np.savez("coinc_cut.npz", exposure=exposure, gpts2=gpts2)
else:
    ss = np.load("coinc_cut.npz")
    exposure = ss['exposure']
    gpts2 = ss['gpts2']
    bad_times = np.logical_not(gpts2)
    


print("Exposure after 1s cut: ", exposure)
print("1s cut exposure loss: ", exposure/curr_exposure)


# plt.figure()
# plt.plot(joint_peaks[:,1], joint_peaks[:,2], 'k.')
# plt.plot(joint_peaks[bad_times,1], joint_peaks[bad_times,2], 'b.')
# plt.plot(joint_peaks[gpts,1], joint_peaks[gpts,2], 'r.')
# max_val = 1.2*np.max( np.hstack((joint_peaks[:,1], joint_peaks[:,2])) )
# plt.fill_between( [0, max_val], [0-2*sig, max_val-2*sig], [0+2*sig, max_val+2*sig], alpha=0.1, facecolor='red')
# plt.plot([0, max_val], [0, max_val], 'r:')
# plt.xlim( (0, max_val) )
# plt.ylim( (0, max_val) )


gpts = np.logical_and(gpts, gpts2)

binlist = np.linspace(0,10,2e2)


## chi2 cut
xx = np.linspace(0,10, 1000)
pin = [0.2e-18, 0, 2.5e-17]
pout = [0.2e-18, 0, 5e-17]
plt.figure()
plt.plot( joint_peaks[gpts,1], joint_peaks[gpts,3], 'k.', ms=1)
plt.plot(xx, np.polyval(pin, xx), 'r')

plt.figure()
plt.plot( joint_peaks[gpts,2], joint_peaks[gpts,4], 'k.', ms=1)
plt.plot(xx, np.polyval(pout, xx), 'r')

chi2_cut = np.logical_and( joint_peaks[:,3] < np.polyval(pin, joint_peaks[:,1]), joint_peaks[:,4] < np.polyval(pout, joint_peaks[:,2]))


## apply energy calibration (accounts for search bias) -- fix hardcoded parameters
ebpc = [ 0.52721076, -0.04189387, 1.62850192] ## energy calibration from cal pulses
def cfit(x,A,mu,sig):
    return x + A*(1+erf((mu-x)/sig))

e_orig = joint_peaks[:,1]
e_xx = np.linspace(0, 10, 1000)
c_xx = cfit(e_xx, *ebpc)
# plt.close('all')
# plt.figure()
# plt.plot(c_xx, e_xx)
# plt.show()
e_cal = np.interp( e_orig, c_xx, e_xx)


## make plot of points passing all cuts
plt.figure()
plt.plot(joint_peaks[:,0], joint_peaks[:,1], 'k.', ms=3, label='all')
plt.plot(joint_peaks[gpts,0], joint_peaks[gpts,1], 'c.', ms=3, label='pass in/out coinc')
plt.plot(joint_peaks[bad_times,0], joint_peaks[bad_times,1], 'b.', ms=3, label='fail 1s coinc')
plt.plot(joint_peaks[accel_cut,0], joint_peaks[accel_cut,1], 'g.', ms=3, label='fail accel cut')



gpts = np.logical_and(gpts, np.logical_not(lab_cut))

orig_gpts = gpts

print("eff before chi2: ", np.sum(gpts)/len(gpts))
gpts = np.logical_and(gpts, chi2_cut)
print("eff after chi2: ", np.sum(gpts)/len(gpts))

accel_cut_comp_gpts = gpts
gpts = np.logical_and( gpts, np.logical_not(accel_cut) )
print("Accel cut efficiency: ", np.sum(gpts)/np.sum(accel_cut_comp_gpts))

plt.plot(joint_peaks[gpts,0], joint_peaks[gpts,1], 'r.', ms=3, label='pass all')
plt.legend()

final_eff = np.sum(gpts)/np.sum(orig_gpts)
print("Final cut efficiency: ", final_eff)

hh, be = np.histogram( joint_peaks[gpts,1], bins=binlist)
hh2, be2 = np.histogram( e_cal[gpts], bins=binlist)

bc = be[:-1] + np.diff(be)/2
bs = be[1]-be[0]


ysig = np.sqrt(hh2)
ysig[ysig==0] = 1
s_to_day = exposure/(24*3600)
#bp, bcov = curve_fit( total_fit, bc, hh2/s_to_day, sigma=ysig/s_to_day, p0=[10000/s_to_day, 0.3, 0.1, 10000/s_to_day, 0.3, 0.1] ) 

#plt.close('all')

xx = np.linspace(0,2,1e3)

plt.figure()
#plt.errorbar( bc, hh/s_to_day, yerr=np.sqrt(hh)/s_to_day, fmt='r.')
plt.errorbar( bc, hh2/s_to_day, yerr=np.sqrt(hh2)/s_to_day, fmt='k.')
#plt.plot(xx, total_fit(xx, *bp), 'k')
plt.yscale('log')
plt.xlim([0,10])
plt.ylim((0.1, 3e5))
plt.xlabel("dp [GeV]")
plt.ylabel("counts/(%.2f GeV day)"%bs)
#plt.step(be[:-1], hh, where='post', color='k')
#plt.step(be2[:-1], hh2, where='post', color='r')

np.savez("dm_data.npz", dp=bc, cts = hh2, expo=s_to_day)

plt.show()


