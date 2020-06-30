import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.signal as sp
import pickle
import matplotlib.cm as cmx
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.backends.backend_pdf import PdfPages
from scipy.special import erf

m_phi = 1e-1

o = open("drdq_interp_grace_%.2e.pkl"%m_phi, 'rb')
fdict = pickle.load(o)
o.close()


## make angular distribution for scattering in the x direction
if(False):
    nmc = 100000000
    theta_vec = np.random.rand(nmc)*np.pi
    phi_vec = np.random.rand(nmc)*2*np.pi
    hang, bang = np.histogram(theta_vec*phi_vec, range=(0,1), bins=100)
    bcang = bang[:-1] + np.diff(bang)/2
    hang = hang/np.sum(hang)
    plt.figure()
    plt.plot(bcang, hang)
    plt.show()
    np.savez("ang_distrib.npz", hang = hang, bcang = bcang )
else:
    ang_dat = np.load("ang_distrib.npz")
    bcang = ang_dat['bcang']
    hang = ang_dat['hang']
    #plt.figure()
    #plt.plot(bcang, hang)
    #plt.show()

    
Ns_to_GeV_over_c = 1.871e18
use_Gev = True

fold = True # uses single gaussian and only shows abs value of the kick

#https://colorfuldots.com/color/#e9531e

pdf = PdfPages('limit_plots_long/limit_plots_m_phi_%.2e.pdf'%m_phi)

def get_color_map( n ):
    jet = plt.get_cmap('jet') 
    cNorm  = colors.Normalize(vmin=0, vmax=n-1)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    outmap = []
    for i in range(n):
        outmap.append( scalarMap.to_rgba(i) )
    return outmap


dat = np.load("dm_data.npz") #, dp=bc, cts = hh2/s_to_day, err=np.sqrt(hh2)/s_to_day)
bc = dat['dp']
h = dat['cts']
sigma = np.sqrt(h)
Exposuretime = dat['expo']*24*3600.

cut_eff = 0.972 * 0.929 * 0.95  #efficiency of acceleromter cut, chi2 cut, amp cut
Exposuretime *= cut_eff

## energy threshold
analysis_thresh = 0.1 #min calibration point
sidx=np.argwhere( bc > analysis_thresh).flatten()[0]
print(sidx)

space = np.linspace(0, 2, 1500)

mx_list = sorted(fdict.keys())

## reduce number of points for speed in massless case
if( m_phi == 0):
    mx_list = np.array(mx_list)
    skip_pts = np.logical_or( mx_list<9.9e4, (mx_list%10000)==0 )
    mx_list = mx_list[skip_pts]
    mx_list = mx_list[ np.logical_not(mx_list == 1.38950e8)]
    mx_list = mx_list[ np.logical_not(mx_list == 1e4)]

limits = np.zeros_like(mx_list)

def gauss_fit(x, A, mu, sig):
    return np.abs(A)*np.exp( -(x-mu)**2/(2*sig**2) )

def log_gauss_fit(x, A, mu, sig):
    return np.abs(A)/x*np.exp( -(np.log(x)-mu)**2/(2*sig**2) )

def total_fit(x,A1,mu1,sig1,A2,mu2,sig2):
    return gauss_fit(x,A1,mu1,sig1) + log_gauss_fit(x,A2,mu2,sig2)

def ffnerf(x, A1, mu1, sig1, A2, mu2, sig2):
    return A1*(1+erf((x-mu1)/(np.sqrt(2)*sig1)))/2 + A2*(1+erf((np.log(x)-mu2)/(np.sqrt(2)*sig2)))/2

#in_loop_eff_pars = [0.70557531, 0.81466432, 0.31597654, 0.29442469, -1.07858784, 0.3547105]
#eff_corr_vec = ffnerf( bc, *in_loop_eff_pars)

ysig = np.sqrt(h)
ysig[ysig==0] = 1
cts_per_day = Exposuretime/(24*3600.)
spars = [40000/cts_per_day, 0, 0.35] # 80/cts_per_day, 1e-3, 0.2]
fpts = np.logical_and(bc>0.07,bc<1.2)
bp, bcov = opt.curve_fit( gauss_fit, bc[fpts], h[fpts]/cts_per_day, sigma=ysig[fpts]/cts_per_day, p0=spars, maxfev=100000 ) 
#bp = spars
print(bp)

xx = np.linspace(0,2,1000)
binsize = bc[1]-bc[0]

    
for mi_idx, mx in enumerate(mx_list): #mx = 1000
    print("Working on mass: ", mx)
    
    plt.figure()
    plt.errorbar(bc, h/cts_per_day, yerr = sigma/cts_per_day, fmt = "k.")#, color = "#7c1ee9")
    plt.plot(xx, gauss_fit(xx, *bp), 'k')

    plt.yscale('log')
    plt.xlim([0,5])
    plt.ylim((0.1, 3e5))
    #plt.show()
    
    alpha_vec = np.hstack((0,np.logspace(-11, -4, 80)))
    
    #qq = np.linspace(bc[0]-binsize/2, 10, 1e6)
    qq = np.linspace(0,20,int(1e6))
    bedges = bc-binsize/2
    bedges = np.hstack((bedges, bc[-1]+binsize/2))

    cols = get_color_map(len(alpha_vec))

    def logL( model, data, bg):
        
        ## fit over best gaussian amplitude
        galist = np.linspace(0.5,2,100)

        out_like = []
        for g in galist:
            cmod = model + g*bg + 1e-30 #tiny offset to keep log from overflowing
            out_like.append( -np.sum( data[sidx:] * np.log(cmod[sidx:]) - cmod[sidx:]) )

        best_like = np.min( out_like )
        best_ga = galist[np.argmin( out_like )]
        return best_like, best_ga
        
    logL_vec = np.zeros_like(alpha_vec)

    best_logL = 1e50
    for j,alpha in enumerate(alpha_vec):

        if(alpha == 0):
            dm_rate = np.zeros_like(qq)
        else:
            dm_rate = fdict[mx](np.log10(qq), np.log10(alpha))
            dm_rate[np.isnan(dm_rate)] = 0


            ## this is now done in plot_results.py instead
            # ## now correct by angular distribution
            # ang_dist_idx = np.argwhere(dm_rate > 0)
            # if(len(ang_dist_idx)>0):
            #     qmax = qq[ang_dist_idx[-1]]
            # else:
            #     continue
            # qcorr = qq[qq<=qmax]
            # corr_fac = np.interp( qcorr/qmax, bcang, hang, left = 0, right = 0 )
            # qcorr_tot = np.zeros_like(dm_rate)
            # qcorr_tot[qq<=qmax] = corr_fac
            # norm = np.trapz( qcorr_tot, qq )
            # qcorr_tot /= norm
            # # print( np.trapz( qcorr_tot, qq ) )
            
            # # plt.figure()
            # # plt.semilogy(qq, dm_rate)
            # # plt.plot(qq, qcorr_tot)
            # # plt.show()

            #dm_rate = dm_rate * qcorr_tot
            
        dm_vec = np.zeros_like(bc)
        bg_vec = np.zeros_like(bc)
        for i in range(len(bc)):
            gidx = np.logical_and(qq>=bedges[i], qq<bedges[i+1])
            dm_vec[i] = np.trapz( dm_rate[gidx], qq[gidx] ) * Exposuretime/3600. 
            bg_vec[i] = gauss_fit(bc[i], *bp)*cts_per_day
            #bg_vec[i] = gauss2(bc[i],*popt)

        ## this is now done in plot_results.py
        ### eff corr
        #dm_vec *= eff_corr_vec
        
        if(False):
            plt.close('all')
            plt.figure()
            plt.semilogy( bc, h, 'bo')
            plt.semilogy( bc, dm_vec, 'k.')
            plt.semilogy( bc, bg_vec, 'r.')
            plt.show()
            
        #bg_vec[bg_vec == 0] = 1e-20 ## slighly non-zero value so log doesn't choke
        logL_vec[j], best_ga = logL( dm_vec, h, bg_vec )
        #print(best_ga)
        
        if( j % 8 == 0):
            plt.plot( bc, (dm_vec + best_ga*bg_vec)/cts_per_day, color = cols[j], label=r"$\alpha_n$ = %.2e"%alpha)
        else:
            plt.plot( bc, (dm_vec + best_ga*bg_vec)/cts_per_day, color = cols[j])
        
        print("logL, alpha: ", logL_vec[j], alpha)
        if( np.isnan(logL_vec[j]) ):
            print("Bad value, skipping mass")
            break
        if( logL_vec[j]>best_logL+10):
            print("Found minimum")
            break
        if(logL_vec[j]<best_logL):
            best_logL = logL_vec[j]
        if(logL_vec[j]<=best_logL+2):
            upper_lim_curve = (dm_vec + best_ga*bg_vec)/cts_per_day

    plt.legend()

    #plt.savefig("/Users/fernandomonteiro/Desktop/Python/Impulse/tempx9/8/hist.pdf")
    plt.tight_layout(pad=0)
    plt.title(r"DM Mass = %.1e GeV, $m_\phi = %.1e$"%(mx, m_phi))
    plt.tight_layout(pad=0)
    plt.savefig("limit_plots_long/data_vs_dm_mx_%.1e_m_phi_%.2e.pdf"%(mx,m_phi))
    pdf.savefig()

    plt.figure()
    plt.errorbar(bc[sidx:], h[sidx:]/cts_per_day - bg_vec[sidx:]/cts_per_day, yerr = sigma[sidx:]/cts_per_day, fmt = "k.", label="no DM")#, color = "#7c1ee9")
    plt.errorbar(bc[sidx:], h[sidx:]/cts_per_day - upper_lim_curve[sidx:], yerr = sigma[sidx:]/cts_per_day, fmt = "r.")#, color = "#7c1ee9")
    plt.xlabel('dp [GeV]')
    plt.ylabel('Residual from best fit [cts/bin]')
    plt.xlim([0,3])
    plt.title(r"DM Mass = %.1e GeV, $m_\phi = %.1e$"%(mx, m_phi))
    plt.tight_layout(pad=0)
    plt.savefig("limit_plots_long/data_vs_dm_mx_resid_%.1e_m_phi_%.2e.pdf"%(mx,m_phi))
    pdf.savefig()
    
    #plt.show()
    
    #plt.figure()
    #plt.errorbar(bc, h-gauss2(bc, *popt), yerr = np.sqrt(h), fmt = "o")#, color = "#7c1ee9")

    logL_vec = 2*logL_vec

    # plt.figure()
    # plt.plot( alpha_vec, logL_vec )
    # plt.show()
    
    # lastval = np.min(((midx+3), len(alpha_vec)))
    # firstval = np.max((midx, midx-2))
    # pp = np.polyfit( alpha_vec[(firstval):lastval], logL_vec[(firstval):lastval], 2)
    
    # xx = np.linspace( alpha_vec[firstval], alpha_vec[lastval-1]*1.5, 1e2)
    # ff = np.polyval(pp,xx)

    # minval = np.min(ff)
    # ff -= minval

    # logL_vec = logL_vec - minval
    # uvec = np.logical_and(logL_vec > 4, alpha_vec > alpha_vec[midx])
    # if( np.sum(uvec) > 0 ):
    #     upper_idx = np.argwhere(uvec)[0][0]
    #     limval = np.interp( 4, logL_vec[midx:upper_idx], alpha_vec[midx:upper_idx])
    # else:
    #     limval = np.nan
    #     upper_idx = len(uvec)-1

    ## do hypothesis test relative to no DM:
    midx = 0 #np.argmin(logL_vec)
    minval = logL_vec[midx] #np.min(logL_vec)
    logL_vec = logL_vec - minval
    
    limval = np.interp(4, logL_vec[midx:], alpha_vec[midx:], left=np.nan, right=np.nan)
    xx = np.linspace(alpha_vec[midx], 1.1*limval, 1000)
    
    fig = plt.figure()
    plt.semilogx( alpha_vec, logL_vec)
    plt.xlabel(r"Neutron coupling, $\alpha_n$")
    plt.ylabel("Negative Log Likelihood, -2 ln( L )")
    #plt.plot(xx, ff , 'r')
    ax = plt.gca()
    axins = inset_axes(ax, width="35%", height="35%", loc="lower right", borderpad=3)
    plt.plot( alpha_vec, logL_vec)
    #plt.plot(xx, ff , 'r')
    if(not np.isnan(limval)):
        xlimval = np.max([0, midx-5])
        plt.xlim( alpha_vec[xlimval], 1.1*limval)
    plt.ylim( (0,5) )
    xxl = plt.xlim()
    plt.plot( xxl, [4,4], 'k:' )
    #limval = xx[ np.argmin( np.abs(ff-4) ) ]
    plt.plot( [limval, limval], [0,5], 'k:')
    plt.sca(ax)
    plt.title(r"DM Mass = %.1e GeV, $m_\phi = %.1e$ upper limit on $\alpha_n$ = %.1e"%(mx, m_phi, limval))
    plt.savefig("limit_plots_long/profile_mx_%.1e_mphi_%.2e.pdf"%(mx,m_phi))
    pdf.savefig()
    
    ##

    # # differential rate
    # binsize = np.diff(bc)[0]
    # h = 1.*h/binsize

    # ExposuretimeH = Exposuretime/3600.

    # plt.figure()
    # plt.errorbar(bc, h/ExposuretimeH, yerr = np.sqrt(h/ExposuretimeH), fmt = "o")#, color = "#7c1ee9")
    # plt.xlabel("dp [GeV/c]")
    # plt.ylabel("Counts/(h Gev/c)")
    # plt.yscale("log")
    # plt.xscale("log")

    print(limval, mi_idx)
    limits[mi_idx] = limval
    plt.close('all')

print(mx_list, limits)
    
plt.figure()
plt.loglog( mx_list, limits )
plt.xlabel("Dark matter mass [GeV]")
plt.ylabel(r"Upper limit on neutron coupling, $\alpha_n$")
plt.tight_layout(pad=0)
plt.savefig("limit_plots_long/limit_plot_%.2e.pdf"%m_phi)
pdf.savefig()

pdf.close()

#np.savez("limit_plots/limit_data_%.2e.npz"%m_phi, mx_list=mx_list, limits=limits)
np.savez("limit_plots_long/limit_data_%.2e.npz"%m_phi, mx_list=mx_list, limits=limits)

#plt.show()

