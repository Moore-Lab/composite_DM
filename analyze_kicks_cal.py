import glob, os, h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp
import matplotlib.mlab as mlab
from scipy.optimize import curve_fit
import datetime as dt
import matplotlib.dates

Fs = 1e4
mass = 9.4e-13 # kg
flen = 524288  # file length, samples
SI_to_GeV = 1.87e18
tthr = 0.005 ## time threshold in s
repro = True

#data_list = ["data/1","data/2","data/3","data/4"]
data_list = ["data/20200615_to/calibration1e_HiZ_20200615",] #["data/20200615_to/kick/0.1ms/6.4V",]

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
    time_template = np.arange(-1/gamma_total, 1.5/gamma_total, 1./Fs)
    w1 = np.sqrt(w0**2 - g**2)
    a = (dp / (Mass * w1)) * np.exp(-time_template * g) * np.sin(w1 * time_template)
    a[time_template<0] = 0
    return [a, time_template]

#vtom_in, vtom_out, f0 = get_v_to_m_and_fressonance("data")
vtom_in, vtom_out, f0 = get_v_to_m_and_fressonance("data/20200615_to/calibration1e_HiZ_20200615")
gam = get_gammas("data")
temp = make_template(Fs, f0, gam, 1, mass)

#tempt = np.hstack((np.zeros(500), temp[0]))
tempt = temp[0]
b,a = sp.butter(3, np.array([42., 58.])/(Fs/2), btype='bandpass')
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

def get_chi2(a, d, idx):
    ll = len(chi2tempf)

    dec_fac = int(Fs/(4*115))

    wfdat = np.roll(d, -idx)[:ll][::dec_fac]

    chi_neg = np.sum( (wfdat + a*chi2tempf[::dec_fac])**2 )
    chi_pos = np.sum( (wfdat - a*chi2tempf[::dec_fac])**2 )
    
    if(False and a > 5):
        plt.figure()
        npts = len(d)
        tvec = np.linspace(0, (npts-1)/Fs, npts)
        #plt.plot( tvec[::dec_fac], d[::dec_fac], 'k' )
        plt.plot( tvec[idx:(idx+ll)][::dec_fac], d[idx:(idx+ll)][::dec_fac], 'bo' )
        if( chi_neg < chi_pos ):
            plt.plot( tvec[idx:(idx+ll)], -chi2tempf*a, 'r' )
        else:
            plt.plot( tvec[idx:(idx+ll)], chi2tempf*a, 'r' )
        plt.show()

    return np.min( (chi_neg, chi_pos) )

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
    for fi, f in enumerate(flist[::1]):
        
        print(f)
        cd = getdata(f)

        dat = cd[0]

        ## get the real time of the file
        timestamp = cd[1]['Time']
        
        npts = len(dat[:,0])
        #print(npts)
        tvec = np.linspace(0, (npts-1)/Fs, npts)

        if(fi == 1):
            oldi_psd, oldi_freqs = mlab.psd(dat[:,0], Fs=Fs, NFFT=int(npts/16))
            oldo_psd, oldo_freqs = mlab.psd(dat[:,4], Fs=Fs, NFFT=int(npts/16))
        
        indat = dat[:,0]*vtom_in
        indat -= np.mean(indat)
        outdat = dat[:,4]*vtom_out
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
        inpeaks = sp.find_peaks(incorrf)[0]
        outpeaks = sp.find_peaks(outcorrf)[0]


        for ip in inpeaks:
            if( tvec[ip] < 0.05): continue
            time_diffs = np.abs(tvec[outpeaks] - tvec[ip])
            mop = np.argmin( time_diffs )
            min_diff = np.min(time_diffs)
            if min_diff < tthr:

                if( False and incorrf[ip] > 0.5):
                    chisq_in = get_chi2(incorrf[ip], indatf, ip)
                else:
                    chisq_in = indatf_outband

                if( False and outcorrf[outpeaks][mop] > 0.75):
                    chisq_out = get_chi2(outcorrf[outpeaks][mop], outdatf, outpeaks[mop])
                else:
                    chisq_out = outdatf_outband
                    
                    
                joint_peaks.append( [file_offset+tvec[ip], incorrf[ip], outcorrf[outpeaks][mop], chisq_in, chisq_out, indatf2_outband, outdatf2_outband, timestamp+tvec[ip]] )

        file_offset += npts/Fs
        
        if( True ): # and np.any( incorrf[inpeaks] > 1.5 ) ):
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
            mon = dat[:,3]
            pstart = np.argwhere( np.logical_and(mon<0.1, np.roll(mon,-1)>=0.1) ).flatten()
            mon *= np.max(indatf)/np.max(mon)

            #plt.figure()
            plt.plot( tvec, mon, 'k')            

            #stack up the kicks
            stackdati = np.zeros(500)
            stackdato = np.zeros(500)
            nstack = 0
            for p in pstart:
                stackdati += indat[(p-20):(p+480)]
                stackdato += outdat[(p-20):(p+480)]
                nstack += 1
            stackdati /= nstack
            stackdato /= nstack
                
            plt.figure()
            plt.plot( tvec[:480], stackdati[20:], label='inloop'  )
            plt.plot( tvec[:480], -stackdato[20:], label='outloop' )
            plt.plot( tt+0.5e-3, tempt, label="Fernando's template" )
            plt.plot( tt+0.5e-3, tempt/np.sqrt(2), label="Fernando's/sqrt(2)" )
            plt.legend(loc="upper right")
            
            newi_psd, newi_freqs = mlab.psd(dat[:,0], Fs=Fs, NFFT=int(npts/16))
            newo_psd, newo_freqs = mlab.psd(dat[:,4], Fs=Fs, NFFT=int(npts/16))
            
            plt.figure()
            # plt.loglog(oldi_freqs, oldi_psd)
            plt.loglog(newi_freqs, newi_psd)
            # plt.figure()
            # plt.loglog(oldo_freqs, oldo_psd)
            plt.loglog(newo_freqs, newo_psd)
            
            plt.show()

    joint_peaks = np.array(joint_peaks)
    np.save("joint_peaks.npy", joint_peaks)
else:
    joint_peaks = np.load("joint_peaks.npy")

sig = np.sqrt(0.11**2 + 0.17**2)
gpts = np.abs( joint_peaks[:,1] - joint_peaks[:,2] ) < 2*sig

xe = np.linspace(0,8,1e2)
ye = np.linspace(0,4e-17,1e2)
plt.figure()
hh, xbe, ybe = np.histogram2d( joint_peaks[:,1], joint_peaks[:,3], bins=(xe,ye) )
plt.pcolormesh(xbe, ybe, np.log10(hh.T), cmap='Greys', vmin=0, vmax=2)
plt.colorbar()

plt.figure()
#plt.plot( joint_peaks[:,1], joint_peaks[:,3], 'k.', ms=1)
plt.plot( joint_peaks[:,1], joint_peaks[:,5], 'b.', ms=1)

plt.figure()
#plt.plot( joint_peaks[:,1], joint_peaks[:,4], 'k.', ms=1)
plt.plot( joint_peaks[:,1], joint_peaks[:,6], 'b.', ms=1)

plt.show()

## automated bad times cut -- make a list of counts per minute above 0.7
num_min = joint_peaks[-1,0]/60
## throw out 1 min bins
min_bins = np.linspace(0, num_min, (num_min+1) )
hmin, bmin = np.histogram( joint_peaks[joint_peaks[:,1]>0.75,0]/60, bins=min_bins )

print("Mean cts/min: ", np.median(hmin))


plt.figure()
mc = bmin[:-1] + np.diff(bmin)/2
plt.errorbar( mc, hmin, yerr=np.sqrt(hmin), fmt='k.' )
bad_times = np.zeros_like(joint_peaks[:,0]) > 1

exposure = joint_peaks[-1,0]

print("Exposure: ", exposure)
exp_orig = exposure
for j in range(len(bmin)-1):
    if hmin[j] > 8:
        bad_times = np.logical_or(bad_times, np.logical_and(bmin[j]*60<joint_peaks[:,0],bmin[j+1]*60>joint_peaks[:,0]))
        exposure -= 60
gpts2 = np.logical_not(bad_times)
print("Exposure: ", exposure)
print("Remaing fraction Exposure: ", exposure/exp_orig)

plt.figure()
plt.plot(joint_peaks[:,0], joint_peaks[:,1], 'k.', ms=3)
plt.plot(joint_peaks[gpts,0], joint_peaks[gpts,1], 'r.', ms=3)
plt.plot(joint_peaks[bad_times,0], joint_peaks[bad_times,1], 'b.', ms=3)

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


# plt.figure()
# dates = []
# for d in joint_peaks[:,5]:
#     dates.append(labview_time_to_datetime(d))
# npd = matplotlib.dates.date2num(dates)

# lab_entry = [ [dt.datetime(2020,6,8,9,10,0), dt.datetime(2020,6,8,16,47,0), "Fernando"],
#               [dt.datetime(2020,6,9,8,0,0), dt.datetime(2020,6,9,10,0,0), "Gadi"],
#               [dt.datetime(2020,6,9,11,0,0), dt.datetime(2020,6,9,14,0,0), "Jiaxiang"],
#               [dt.datetime(2020,6,9,15,50,0), dt.datetime(2020,6,9,16,0,0), "Fernando"],
#               [dt.datetime(2020,6,9,16,30,0), dt.datetime(2020,6,9,19,45,0), "Jiaxiang"],
#               [dt.datetime(2020,6,11,11,20,0), dt.datetime(2020,6,11,11,35,0), "Fernando"],
#               #[dt.datetime(2020,6,11,15,30,0), dt.datetime(2020,6,11,18,00,0), "Gadi"],
#               #[dt.datetime(2020,6,12,10,15,0), dt.datetime(2020,6,12,10,30,0), "Gadi"],]
#               ]

# plt.plot_date(npd[:], joint_peaks[:,1], 'k.', ms=2, label='all data')
# plt.plot_date(npd[bad_times], joint_peaks[bad_times,1], 'r.', ms=2, label='high rate')
# yy = plt.ylim()
# # for ll in lab_entry:
# #     dsin = matplotlib.dates.date2num(ll[0])
# #     dsout = matplotlib.dates.date2num(ll[1])
# #     plt.fill_between( [dsin, dsout], [yy[0], yy[0]], [yy[1],yy[1]], facecolor='b', alpha=0.1 )
# #     plt.text( dsin, 0.9*yy[1], ll[2], color='b')
# plt.legend()
# plt.show()

# plt.figure()
# plt.plot(joint_peaks[:,1], joint_peaks[:,2], 'k.')
# plt.plot(joint_peaks[gpts,1], joint_peaks[gpts,2], 'r.')
# plt.plot(joint_peaks[bad_times,1], joint_peaks[bad_times,2], 'b.')
# max_val = 1.2*np.max( np.hstack((joint_peaks[:,1], joint_peaks[:,2])) )
# plt.fill_between( [0, max_val], [0-2*sig, max_val-2*sig], [0+2*sig, max_val+2*sig], alpha=0.1, facecolor='red')
# plt.plot([0, max_val], [0, max_val], 'r:')
# plt.xlim( (0, max_val) )
# plt.ylim( (0, max_val) )


gpts = np.logical_and(gpts, gpts2)

binlist = np.linspace(0,10,2e2)
hh, be = np.histogram( joint_peaks[:,1], bins=binlist)
hh2, be2 = np.histogram( joint_peaks[gpts,1], bins=binlist)

bc = be[:-1] + np.diff(be)/2
bs = be[1]-be[0]

def gauss_fit(x, A, mu, sig):
    return A*np.exp( -(x-mu)**2/(2*sig**2) )

def log_gauss_fit(x, A, mu, sig):
    return A/x*np.exp( -(np.log(x)-mu)**2/(2*sig**2) )

def total_fit(x,A1,mu1,sig1,A2,mu2,sig2):
    return gauss_fit(x,A1,mu1,sig1) + log_gauss_fit(x,A2,mu2,sig2)

ysig = np.sqrt(hh2)
ysig[ysig==0] = 1
s_to_day = exposure/(24*3600)
bp, bcov = curve_fit( total_fit, bc, hh2/s_to_day, sigma=ysig/s_to_day, p0=[10000/s_to_day, 0.3, 0.1, 10000/s_to_day, 0.3, 0.1] ) 

#plt.close('all')

xx = np.linspace(0,2,1e3)

plt.figure()
#plt.errorbar( bc, hh/s_to_day, yerr=np.sqrt(hh)/s_to_day, fmt='k.')
plt.errorbar( bc, hh2/s_to_day, yerr=np.sqrt(hh2)/s_to_day, fmt='k.')
plt.plot(xx, total_fit(xx, *bp), 'k')
plt.yscale('log')
plt.xlim([0,2.5])
plt.ylim((0.1, 3e5))
plt.xlabel("dp [GeV]")
plt.ylabel("counts/(%.2f GeV day)"%bs)
#plt.step(be[:-1], hh, where='post', color='k')
#plt.step(be2[:-1], hh2, where='post', color='r')

np.savez("dm_data.npz", dp=bc, cts = hh2, expo=s_to_day)

plt.show()


