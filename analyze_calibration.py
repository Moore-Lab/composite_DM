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

Fs = 1e4
mass = 9.4e-13 # kg
flen = 524288  # file length, samples
SI_to_GeV = 1.87e18
tthr = 0.050 ## time threshold in s for which to look for coincidences with calibration pulses (this is big to get random rate)
repro = False # Set true to reprocess data, false to read from file

data_list = ["data/20200615_to/kick/0.1ms/0.1V",
             "data/20200615_to/kick/0.1ms/0.2V",
             "data/20200615_to/kick/0.1ms/0.4V",
             "data/20200615_to/kick/0.1ms/0.8V",
             "data/20200615_to/kick/0.1ms/1.6V",
             "data/20200615_to/kick/0.1ms/3.2V",
             "data/20200615_to/kick/0.1ms/6.4V"]

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
vtom_in, vtom_out, f0 = get_v_to_m_and_fressonance("data/20200615_to/calibration1e_HiZ_20200615")

## fix cal for now
vtom_in *= np.sqrt(2)
vtom_out *= np.sqrt(2)

gam = get_gammas("data")
temp = make_template(Fs, f0, gam, 1, mass)

#tempt = np.hstack((np.zeros(500), temp[0]))
tempt = temp[0]
b,a = sp.butter(3, np.array([65., 115.])/(Fs/2), btype='bandpass')
b2,a2 = sp.butter(3, (f0/2)/(Fs/2), btype='lowpass')
tempf = sp.filtfilt(b,a,tempt)

normf = np.sum( tempf**2 )
#tempt #/= np.sum( tempt**2 )
tempf /= normf

bstop,astop = sp.butter(3, np.array([65., 115.])/(Fs/2), btype='bandstop')
bstop2,astop2 = sp.butter(3, 400./(Fs/2), btype='lowpass')

#tt = np.arange(-flen/(2*Fs), flen/(2*Fs), 1./Fs)
tt = np.arange(-1.5/gam, 1.5/gam, 1./Fs)
tempt = make_template(Fs, f0, gam, 11.6/SI_to_GeV, mass)[0]
tempt = sp.filtfilt(b,a,tempt)

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

        fparts = f.split("/")
        volts = float( fparts[4][:-1] )
        
        npts = len(dat[:,0])
        #print(npts)
        tvec = np.linspace(0, (npts-1)/Fs, npts)

        # if(fi == 1):
        #     oldi_psd, oldi_freqs = mlab.psd(dat[:,0], Fs=Fs, NFFT=int(npts/16))
        #     oldo_psd, oldo_freqs = mlab.psd(dat[:,4], Fs=Fs, NFFT=int(npts/16))
        
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

        ## fudge factor of 1.1 accounts for amplitude loss in filter frequency for finding envelope
        ## don't worry about the fudge factor, it's fine...
        incorrf *= 1.1*np.sqrt(2)*SI_to_GeV
        outcorrf *= 1.1*np.sqrt(2)*SI_to_GeV
        inpeaks_orig = sp.find_peaks(incorrf)[0]
        outpeaks_orig = sp.find_peaks(outcorrf)[0]

        ## now find the true peak in the correlation closest to the smoothed peak
        inpeaks = np.zeros_like(inpeaks_orig)
        outpeaks = np.zeros_like(outpeaks_orig)        

        ## 2 ms shouldn't be hardcoded -- depends on the period of the oscillation (sphere res frequency)
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
            if( ip<500 or ip>npts-500): continue
            time_diffs_ip = tvec[inpeaks] - tvec[ip]
            time_diffs_op = tvec[outpeaks] - tvec[ip]

            ## check for the closest outloop peak to the ipp
            #opp_closest = tvec[
            
            ##### in loop sensor only
            ## find the largest peak within the cal window for each
            if(np.any(np.abs(time_diffs_ip) < tthr)):
                ip_peakvals = np.abs(incorr[inpeaks][np.abs(time_diffs_ip) < tthr])*SI_to_GeV
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
                ipp, ipt, ichi2 = -1, -1, 1e20
            
            ### done with inloop

            ##outloop sensor only
            if(np.any(np.abs(time_diffs_op) < tthr)):
                op_peakvals = np.abs(outcorr[outpeaks][np.abs(time_diffs_op) < tthr])*SI_to_GeV
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
                opp, opt, ochi2 = -1, -1, 1e20
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
            plt.plot( tvec[:680]-0.02, -stackdato, label='out of loop' )
            #plt.plot( tt+0.5e-3, tempt, label="Fernando's template" )
            plt.plot( tt[120:]+0.5e-3, tempt[120:], 'k:', label="Template" )
            plt.legend(loc="upper right")
            
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
mean_list = np.zeros((len(vlist),4))

fig_in = plt.figure()
fig_out = plt.figure()
def ffn(x,A,mu,sig,C):
    return A*np.exp(-(x-mu)**2/(2*sig**2)) + C

cols = get_color_map(len(vlist))

## list of kicks in GeV/c (what we expect)
gev_list = 200*vlist*1e-4/(0.0033)*(1.6e-19) * SI_to_GeV

for i,v in enumerate(vlist):
    print("Working on volts: ", v)
    cdat = joint_peaks[joint_peaks[:,5] == v, :]

    ## get the mean value for each voltage of the kick amplitude (before final calibration)
    tot_cts = len(cdat[:,1])
    gpts1 = cdat[:,1] > 0
    gpts2 = cdat[:,3] > 0
    std1 = np.std( cdat[gpts1,1] )/np.sqrt(np.sum(gpts1))
    std2 = np.std( cdat[gpts2,3] )/np.sqrt(np.sum(gpts2))    
    mean_list[i,:] = [ np.median(cdat[gpts1,1]), np.median(cdat[gpts2,3]), std1, std2 ]

    good_cts_in, good_cts_out = np.sum( cdat[:,1] >0), np.sum( cdat[:,3] >0)


    ## find trigger efficiency and random coincident rate (in loop)
    plt.figure(fig_in.number)
    hh, be = np.histogram( cdat[:,2], range=(-tthr,tthr), bins=50 )
    bc = be[:-1]+np.diff(be)/2
    blpts = np.logical_or(bc < -0.02, bc > 0.02)
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
    hh, be = np.histogram( cdat[:,4], range=(-tthr,tthr), bins=50 )
    bc = be[:-1]+np.diff(be)/2
    blpts = np.logical_or(bc < -0.02, bc > 0.02)
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
    return x + A*(1+erf((mu-x)/sig))

xx = np.linspace(0, gev_list[-1]*1.2, 1000)

plt.errorbar( gev_list, mean_list[:,0]/corr_fac_in, yerr=mean_list[:,2]/corr_fac_in, fmt='k.')
plt.errorbar( gev_list, mean_list[:,1]/corr_fac_out, yerr=mean_list[:,3]/corr_fac_out, fmt='r.')
plt.plot(gev_list, gev_list, 'k:')
plt.title("Corr fac inloop: %.2f, corr fac outloop: %.2f"%(corr_fac_in, corr_fac_out))
plt.xlabel("Calibration pulse amplitude [GeV]")
plt.ylabel("Recontructed amplitude")


## fit calibraion to get search bias at low energy
ecbp, ecbc = curve_fit(cfit, gev_list, mean_list[:,0]/corr_fac_in,  p0=[0.1,1,1])
#ecbp = [0.1,1,1]
plt.plot( xx, cfit(xx, *ecbp), 'k')

print("Energy cal params: ", ecbp)


cbp, cbc = curve_fit(cfit, joint_peaks[:,1]/corr_fac_in, joint_peaks[:,3]/corr_fac_out, p0=[0.1,1,1])
#cbp = [0.1,1,1]

## calculate the efficiency of the in/out loop amplitude matching criterion
plt.figure()
xvals, yvals = joint_peaks[:,1]/corr_fac_in, joint_peaks[:,3]/corr_fac_out
plt.plot( xvals, yvals, 'k.', ms=1)
#plt.plot( gev_list, gev_list, 'r:')
plt.plot( xx, cfit(xx, *cbp), 'r:')

sigval = np.std( yvals - cfit(xvals, *cbp) )
plt.fill_between( xx, cfit(xx, *cbp)-2*sigval, cfit(xx, *cbp)+2*sigval, facecolor='r', alpha=0.1)

cut_eff = np.sum( np.abs(yvals-cfit(xvals,*cbp))<2*sigval )/len(yvals)
plt.title("Cut efficiency = %.2f"%cut_eff)
print("Sigma: ", sigval)


plt.xlabel("Reconstructed in loop amp [GeV]")
plt.ylabel("Reconstructed out of loop amp [GeV]")



def ffnerf(x, A1, mu1, sig1, A2, mu2, sig2):
    return A1*(1+erf((x-mu1)/(np.sqrt(2)*sig1)))/2 + A2*(1+erf((np.log(x)-mu2)/(np.sqrt(2)*sig2)))/2

#spars=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
spars = [ 0.70557531, 0.81466432, 0.31597654, 0.29442469, -1.07858784, 0.3547105 ]
bpi, bci = curve_fit(ffnerf, gev_list, eff_list[:,0], sigma=(eff_list[:,1]+eff_list[:,2])/2, p0=spars)
bpo, bco = curve_fit(ffnerf, gev_list, eff_list[:,3], sigma=(eff_list[:,4]+eff_list[:,5])/2, p0=spars)
#bpi = spars
#bpo = spars


spli = UnivariateSpline(gev_list, eff_list[:,0],w=2/(eff_list[:,1]+eff_list[:,2]))
splo = UnivariateSpline(gev_list, eff_list[:,3],w=2/(eff_list[:,4]+eff_list[:,5]))

plt.figure()
plt.errorbar( gev_list, eff_list[:,0], yerr=(eff_list[:,1],eff_list[:,2]), fmt='k.')
plt.plot(xx, ffnerf(xx, *bpi), 'k', label='in loop')
#plt.plot(xx, spli(xx), 'k')
plt.errorbar( gev_list, eff_list[:,3], yerr=(eff_list[:,4],eff_list[:,5]), fmt='r.')
plt.plot(xx, ffnerf(xx, *bpo), 'r', label='out of loop')
#plt.plot(xx, splo(xx), 'r')

plt.plot(xx, ffnerf(xx, *bpi)*ffnerf(xx, *bpo), 'b', label='combined')

plt.xlabel("Impulse amplitude [GeV]")
plt.ylabel("Reconstruction efficiency")
plt.legend()

print("In loop recon eff params: ", bpi)

#### done calculating in/out loop amplitude matching efficiency



## finally, make the chi2 cut:
plt.figure()
gpts = np.logical_and( joint_peaks[:,2]>-0.0005, joint_peaks[:,2]<0.0015 )
#plt.plot( joint_peaks[:,1], joint_peaks[:,6], 'k.', ms=1)
plt.plot( joint_peaks[gpts,1], joint_peaks[gpts,6], 'k.', ms=1)
pin = [0.2e-18, 0, 2.5e-17]
plt.plot(xx, np.polyval(pin, xx), 'r')
pass_cut = joint_peaks[gpts,6] < np.polyval(pin, joint_peaks[gpts,1])

plt.figure()
plt.plot( joint_peaks[gpts,3], joint_peaks[gpts,7], 'k.', ms=1)
pout = [0.2e-18, 0, 5e-17]
plt.plot(xx, np.polyval(pout, xx), 'r')
pass_cut = np.logical_and( joint_peaks[gpts,6] < np.polyval(pin, joint_peaks[gpts,1]), joint_peaks[gpts,7] < np.polyval(pout, joint_peaks[gpts,3]))

print("Cut efficiency, out+in: ", np.sum(pass_cut)/len(pass_cut))

plt.show()
