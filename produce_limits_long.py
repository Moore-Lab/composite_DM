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
from scipy.optimize import minimize, curve_fit
import sys
from scipy.interpolate import UnivariateSpline as us

m_phi = float(sys.argv[1]) #0
nugg_frac = float(sys.argv[2])

## fraction of dark matter that is nuggets of this type:


o = open("drdq_interp_grace_%.2e.pkl"%m_phi, 'rb')
fdict = pickle.load(o)
o.close()

## nuiscance paramters for systematics:
gscale1_mu, gscale1_sig = 1.0, 1.0  ## weak constraint for background
gscale2_mu, gscale2_sig = 1.0, 1.0  ## weak constraint for background
escale_mu, escale_sig = 1.0, 0.05/3.999 ## syst error on electrode spacing
nneut_mu, nneut_sig = 1.0, 0.33 ## syst error on number of neutrons

LLoffset = 10021889 ## loglikelihood offset to make numerical solution well-behaved


def get_color_map( n, mname = 'jet' ):
    jet = plt.get_cmap(mname) 
    cNorm  = colors.Normalize(vmin=0, vmax=n-1)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    outmap = []
    for i in range(n):
        outmap.append( scalarMap.to_rgba(i) )
    return outmap

## make paper spectrum plot
make_spec_plot = False
spec_plot_vals = [[1, 2.1e-8],
                  [0.1, 1.18e-8],
                  [0.01, 3.3e-9],
                  [0, 2.1e-9]]
spec_plot_mass = 5e3
lim_plot_cols = get_color_map(len(spec_plot_vals)+2, mname="Blues_r")

# ### code below now done in plot_results.py
# ## make angular distribution for scattering in the x direction
# if(False):
#     nmc = 100000000
#     theta_vec = np.random.rand(nmc)*np.pi
#     phi_vec = np.random.rand(nmc)*2*np.pi
#     hang, bang = np.histogram(theta_vec*phi_vec, range=(0,1), bins=100)
#     bcang = bang[:-1] + np.diff(bang)/2
#     hang = hang/np.sum(hang)
#     plt.figure()
#     plt.plot(bcang, hang)
#     plt.show()
#     np.savez("ang_distrib.npz", hang = hang, bcang = bcang )
# else:
#     ang_dat = np.load("ang_distrib.npz")
#     bcang = ang_dat['bcang']
#     hang = ang_dat['hang']
#     #plt.figure()
#     #plt.plot(bcang, hang)
#     #plt.show()
# ########################
    
Ns_to_GeV_over_c = 1.871e18
use_Gev = True

fold = True # uses single gaussian and only shows abs value of the kick

#https://colorfuldots.com/color/#e9531e

if(not make_spec_plot):
    pdf = PdfPages('limit_plots_long/limit_plots_m_phi_%.2e_nugg_frac_%.2e.pdf'%(m_phi,nugg_frac))


dat = np.load("dm_data.npz") #, dp=bc, cts = hh2/s_to_day, err=np.sqrt(hh2)/s_to_day)
bc = dat['dp']
h = dat['cts']
sigma = np.sqrt(h)
Exposuretime = dat['expo']*24*3600.

cut_eff = 0.974 * 0.959 * 0.95  #efficiency of acceleromter cut, chi2 cut, amp cut
Exposuretime *= cut_eff

## energy threshold
analysis_thresh = 0.15 #min calibration point
sidx=np.argwhere( bc >= analysis_thresh).flatten()[0]
print(sidx)

space = np.linspace(0, 2, 1500)

mx_list = sorted(fdict.keys())

## reduce number of points for speed in massless case
#if( m_phi == 0):
#    mx_list = np.array(mx_list)
    #skip_pts = np.logical_or( mx_list<9.9e4, (mx_list%10000)==0 )
    #mx_list = mx_list[skip_pts]
    #mx_list = mx_list[ np.logical_not(mx_list == 1.38950e8)]
    #mx_list = mx_list[ np.logical_not(mx_list == 1e4)]

limits = np.zeros_like(mx_list)

def gauss_fit(x, A, sig):
    return np.abs(A)*np.exp( -x**2/(2*sig**2) )

def log_gauss_fit(x, A, mu, sig):
    return np.abs(A)/x*np.exp( -(np.log(x)-mu)**2/(2*sig**2) )

def total_fit(x,A1,sig1): #,A2,mu2,sig2):
    return gauss_fit(x,A1,sig1) #+ 0*log_gauss_fit(x,A2,mu2,sig2)

def cball_fit(x, Amp, sig, alpha, n):
    A = (n*sig/alpha)**n * np.exp(-alpha**2/(2*sig**2))
    B = n*sig/alpha - alpha/sig

    p1 = np.exp( -x**2/(2*sig**2))
    p2 = A*(B + x/sig)**(-n)

    p1[ x > alpha] = p2[x > alpha]
    return Amp*p1

def ffnerf(x, A1, mu1, sig1, A2, mu2, sig2):
    return A1*(1+erf((x-mu1)/(np.sqrt(2)*sig1)))/2 + A2*(1+erf((np.log(x)-mu2)/(np.sqrt(2)*sig2)))/2

#in_loop_eff_pars = [0.70557531, 0.81466432, 0.31597654, 0.29442469, -1.07858784, 0.3547105]
#eff_corr_vec = ffnerf( bc, *in_loop_eff_pars)

ysig = np.sqrt(h)
ysig[ysig==0] = 1
cts_per_day = Exposuretime/(24*3600.)
#spars = [40000/cts_per_day, 0, 0.35] # 80/cts_per_day, 1e-3, 0.2]
#spars = [40000/cts_per_day, 0.35, 80/cts_per_day, 1e-3, 0.2]
spars = [ 1.81436062e+05,  2.32744182e-01] #  1.34924320e+03, -8.25155098e-01, 1.90743776e-01]
fpts = np.logical_and(bc>analysis_thresh,bc<1.2)
bp, bcov = opt.curve_fit( total_fit, bc[fpts], h[fpts]/cts_per_day, sigma=ysig[fpts]/cts_per_day, p0=spars, maxfev=100000 ) 
#bp = spars
print(bp)

xx = np.linspace(0.05,2,1000)
binsize = bc[1]-bc[0]

def rescaled_model( xv, model, bg1, data, qvals ):
    escale = xv[0]
    gscale1 = xv[1]
    nneut = xv[2]
    ## rescale DM model by escale and error on number of neutrons
    new_model = np.interp( qvals, escale*qvals, model/escale, left=0, right=0 ) * nneut**2
    new_bg1 = gscale1 * bg1  ## gauss term
    #new_bg2 = gscale2 * bg2  ## log gauss term
    tot_mod = new_model + new_bg1 #+ new_bg2

    ## remove any bad values from the model
    tot_mod[ tot_mod < 1e-30 ] = 1e-30 ## minimum value so the log doesn't overflow

    ## for nneut term, ensure the fit is not driven to impossible values (i.e., those that would not be
    ## consistent with res frequency) are truncated at central 1 sig (30%)
    ## based on Slack conversation between Dave/Fernando, 2020/07/08
    if( np.abs(nneut - nneut_mu) > 0.5*nneut_sig ):
        ## smoother way of providing "hard edge"
        nneut_term = (nneut - nneut_mu)**2/(2*nneut_sig**2) + ((nneut - nneut_mu)**2 - 0.5**2)/(1e-2*nneut_sig)**2
    else:
        nneut_term = (nneut - nneut_mu)**2/(2*nneut_sig**2)

    gauss_term = (escale - escale_mu)**2/(2*escale_sig**2) + (gscale1 - gscale1_mu)**2/(2*gscale1_sig**2) + nneut_term
    return -np.sum( data[sidx:] * np.log(tot_mod[sidx:]) - tot_mod[sidx:] ) + gauss_term + LLoffset

    
def logL( model, data, bg1, qvals):
    
    ## profile over best gaussian amplitude, escale
    res = minimize( rescaled_model, (1.0, 1.0, 1.0), args=(model, bg1, data, qvals), tol=0.1 )

    if( not res.success and not res.status==2):
        #print( "optimization failed" )
        return np.nan, 1.0, 1.0, 1.0, []
                    
    xvals = res.x

    best_like = res.fun
    best_ea = xvals[0]
    best_ga1 = xvals[1]
    #best_ga2 = xvals[2]
    best_nn = xvals[2]
    new_model = np.interp( qvals, best_ea*qvals, model/best_ea, left=0, right=0 )
    new_bg = best_ga1 * bg1 #+ best_ga2 * bg2
    best_model = new_model + new_bg
    
    return best_like, best_ea, best_ga1, best_nn, best_model



for mi_idx, mx in enumerate(mx_list): #mx = 1000
    print("Working on mass: ", mx)

    #if( mx < 612 ): continue
    
    fig=plt.figure()
    if(make_spec_plot):
        ## bin size correction

        binlist = np.linspace(0,10,200)
        cut_dat = np.load("cut_data.npz")
        bt_cut = np.logical_not(cut_dat['bad_times'])
        lab_cut = np.logical_not(cut_dat['lab_cut'])
        all_lt_cuts = np.logical_not(cut_dat['accel_cut'] + cut_dat['lab_cut'] + cut_dat['bad_times'])  # + cut_dat['chi2_cut'] + cut_dat['amp_match_cut'] + cut_dat['bad_times']
        all_cuts = np.logical_not(cut_dat['accel_cut'] + cut_dat['lab_cut'] + cut_dat['bad_times']) * cut_dat['chi2_cut'] * cut_dat['amp_match_cut']
        print("accel cut:", np.sum(all_cuts)/len(all_cuts))
        h_orig, be = np.histogram( cut_dat['ec'], bins=binlist)
        h_bt, be = np.histogram( cut_dat['ec'][bt_cut], bins=binlist)
        h_lab, be = np.histogram( cut_dat['ec'][lab_cut], bins=binlist)
        h_mid, be = np.histogram( cut_dat['ec'][all_lt_cuts], bins=binlist)
        h, be = np.histogram( cut_dat['ec'][all_cuts], bins=binlist)

        sigma = np.sqrt(h_orig)
        plt.step(bc, h_orig/cts_per_day * 1/binsize, linestyle = ':', color='gray', where='mid')
        #plt.errorbar(bc, h_orig/cts_per_day * 1/binsize, yerr = sigma/cts_per_day * 1/binsize, fmt = ".", color='b')#, color = "#7c1ee9")
        sigma = np.sqrt(h_bt)
        h_bt[h_bt==0] = 1e-10
        plt.step(bc, h_bt/cts_per_day * 1/binsize, linestyle = '-', color='gray', where='mid')
        sigma = np.sqrt(h_lab)
        h_lab[h_lab==0] = 1e-10
        plt.step(bc, h_lab/cts_per_day * 1/binsize, linestyle = '-', color='gray', where='mid')
        sigma = np.sqrt(h_mid)
        h_mid[h_mid==0] = 1e-10
        plt.step(bc, h_mid/cts_per_day * 1/binsize, linestyle = '-', color='gray', where='mid')
        #plt.errorbar(bc, h_mid/cts_per_day * 1/binsize, yerr = sigma/cts_per_day * 1/binsize, fmt = "r.")#, color = "#7c1ee9")
        sigma = np.sqrt(h)
        plt.errorbar(bc, h/cts_per_day * 1/binsize, yerr = sigma/cts_per_day * 1/binsize, fmt = "k.")#, color = "#7c1ee9")
        plt.plot(xx, total_fit(xx, *bp) * 1/binsize, 'k:')
    else:
        plt.errorbar(bc, h/cts_per_day, yerr = sigma/cts_per_day, fmt = "k.")#, color = "#7c1ee9")
        #plt.plot(xx, gauss_fit(xx, *bp[0:2]), 'k:')
        #plt.plot(xx, log_gauss_fit(xx, *bp[2:]), 'k--')
        plt.plot(xx, total_fit(xx, *bp), 'k')
        
    plt.yscale('log')
    plt.xlim([0,5])
    plt.ylim((0.1, 3e5))
    #plt.show()
    
    alpha_vec = np.hstack((0,np.logspace(-11, -4, 80)))
    
    #qq = np.linspace(bc[0]-binsize/2, 10, 1e6)
    qq = np.linspace(0.001,20,int(1e6))
    bedges = bc-binsize/2
    bedges = np.hstack((bedges, bc[-1]+binsize/2))

    if(make_spec_plot):
        for csidx, spec_vals in enumerate(spec_plot_vals):
            mx = spec_plot_mass
            alpha = spec_vals[1]
            mp = spec_vals[0]
            if(not mp==0.1): continue
            o = open("drdq_interp_grace_%.2e.pkl"%mp, 'rb')
            fdict = pickle.load(o)
            o.close()

            mx_list = sorted(fdict.keys())
            
            closest_mx = mx_list[np.argmin( np.abs(np.array(mx_list) - mx) )]
            dm_rate = fdict[closest_mx](np.log10(qq), np.log10(alpha))
            dm_rate[np.isnan(dm_rate)] = 0

            dm_vec = np.zeros_like(bc)
            bg_vec1 = np.zeros_like(bc)
            for i in range(len(bc)):
                gidx = np.logical_and(qq>=bedges[i], qq<bedges[i+1])
                dm_vec[i] = np.trapz( dm_rate[gidx], qq[gidx] ) * Exposuretime/3600. * nugg_frac
                bg_vec1[i] = gauss_fit(bc[i], *bp)*cts_per_day
                #bg_vec2[i] = log_gauss_fit(bc[i], *bp[2:])*cts_per_day
            clogL, best_ea, best_ga, best_nn, best_model = logL( dm_vec, h, bg_vec1, bc )
            bidx = bc > 0.05
            plt.plot( bc[bidx], best_model[bidx]/cts_per_day * 1/binsize, color = lim_plot_cols[csidx], label=r"$\alpha_n$ = %.2e"%alpha)

        plt.xlim([0,5])
        plt.ylim((1, 5e6))
        plt.xlabel("Reconstructed impulse amplitude [GeV]")
        plt.ylabel("Counts [GeV$^{-1}$ day$^{-1}$]")

        fig.set_size_inches(5,2.75)
        plt.tight_layout(pad=0)

        ax = plt.axes([0.5, 0.6, 0.45, 0.35])
        
        ## now the recon efficiency in the inset
        qvals = np.linspace(0.05, 4, 1000)

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

        ## interpolate the scatter between runs as the error
        stack = np.vstack( (ry1, ry2, ry3) )
        print(np.shape(stack))
        eff_err = np.std( stack, axis=0 )/3
        #plt.figure()
        #plt.plot(rx2, eff_err)
        #eei = np.interp(qvals, rx2, eff_err)
        uss = us( rx2, eff_err, s = 1e-4, k=1)
        eei = uss( qvals )
        #plt.plot( qvals, eei )
        #plt.show()
        
        #print(rp1)
        #bpi, bci = curve_fit(ffnerf, rx1, cdat, p0 = [1.46, 0.35, 0.46, -0.46, 4.7, -0.05])
        bpi, bci = curve_fit(ffnerf2, rx1, cdat, p0 = [1.46, 0.35, 0.46, 0.3], sigma = cerr)
        
        #plt.errorbar( rx1, ry1, yerr=re1, fmt='.')
        #plt.errorbar( rx2, ry2, yerr=re2, fmt='.')
        #plt.errorbar( rx3, ry3, yerr=re3, fmt='.')
        ieff = 0.959 * 0.95
        ieff_err = np.sqrt(0.012**2 + 0.018**2) ## relative error on cut efficiencies from analyze_calibration
        yvals = ffnerf2(qvals, *bpi)*ieff
        yvals[qvals < analysis_thresh] = 0
        #plt.errorbar( rx2, cdat*ieff, yerr=cerr*ieff, fmt='k.')
        yvals_u = yvals + np.sqrt(eei**2 + ieff_err**2)
        yvals_u[qvals < analysis_thresh] = 0
        yvals_l = yvals - np.sqrt(eei**2 + ieff_err**2)
        yvals_l[qvals < analysis_thresh] = 0
        plt.fill_between( qvals, yvals_l, yvals_u, color='k', alpha = 0.2, edgecolor='none')
        plt.plot(qvals, yvals, 'k', label='com')

        plt.plot([analysis_thresh, analysis_thresh], [0,1], 'k:')
        
        plt.xlim([0, 2.5])
        plt.ylim([0, 1])

        plt.grid(True)
        
        plt.xlabel("Recon. impulse amplitude [GeV]")        
        plt.ylabel("Signal efficiency")

        plt.savefig("dm_spectrum.pdf")
        
        plt.show()
        sys.exit(0)
            
    cols = get_color_map(len(alpha_vec))
        
    logL_vec = np.zeros_like(alpha_vec)

    best_logL = 1e50
    for j,alpha in enumerate(alpha_vec):

        if(alpha == 0):
            dm_rate = np.zeros_like(qq)
        else:
            dm_rate = fdict[mx](np.log10(qq), np.log10(alpha))
            dm_rate[np.isnan(dm_rate)] = 0
            
        dm_vec = np.zeros_like(bc)
        bg_vec1 = np.zeros_like(bc)
        bg_vec2 = np.zeros_like(bc)
        for i in range(len(bc)):
            gidx = np.logical_and(qq>=bedges[i], qq<bedges[i+1])
            dm_vec[i] = np.trapz( dm_rate[gidx], qq[gidx] ) * Exposuretime/3600. 
            bg_vec1[i] = gauss_fit(bc[i], *bp)*cts_per_day
            #bg_vec2[i] = log_gauss_fit(bc[i], *bp[2:])*cts_per_day

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
            
        logL_vec[j], best_ea, best_ga, best_nn, best_model = logL( dm_vec, h, bg_vec1, bc )

        #print("best nn: ", best_nn)
        print("logL, alpha: %.5f\t%.2e"%(logL_vec[j], alpha))

        if( np.isnan(logL_vec[j]) ):
            continue
        
        if( j % 1 == 0):
            plt.plot( bc, best_model/cts_per_day, color = cols[j], label=r"$\alpha_n$ = %.2e"%alpha)
        else:
            plt.plot( bc, best_model/cts_per_day, color = cols[j])
        
        if( logL_vec[j]>best_logL+20):
            print("Found minimum")
            break
        if( j == 0 ):
            best_logL = logL_vec[j]
        if(logL_vec[j]<=best_logL+2):
            upper_lim_curve = (dm_vec + best_ga*bg_vec1 )/cts_per_day

    plt.legend()

    #plt.savefig("/Users/fernandomonteiro/Desktop/Python/Impulse/tempx9/8/hist.pdf")
    plt.tight_layout(pad=0)
    plt.title(r"DM Mass = %.1e GeV, $m_\phi = %.1e$"%(mx, m_phi))
    plt.tight_layout(pad=0)
    if( not make_spec_plot):
        plt.savefig("limit_plots_long/data_vs_dm_mx_%.1e_m_phi_%.2e.pdf"%(mx,m_phi))
        pdf.savefig()

    plt.figure()
    best_mod = best_ga*bg_vec1
    plt.errorbar(bc[sidx:], h[sidx:]/cts_per_day - best_mod[sidx:]/cts_per_day, yerr = sigma[sidx:]/cts_per_day, fmt = "k.", label="no DM")#, color = "#7c1ee9")
    plt.errorbar(bc[sidx:], h[sidx:]/cts_per_day - upper_lim_curve[sidx:], yerr = sigma[sidx:]/cts_per_day, fmt = "r.")#, color = "#7c1ee9")
    plt.xlabel('dp [GeV]')
    plt.ylabel('Residual from best fit [cts/bin]')
    plt.xlim([0,3])
    plt.title(r"DM Mass = %.1e GeV, $m_\phi = %.1e$"%(mx, m_phi))
    plt.tight_layout(pad=0)
    if( not make_spec_plot):
        plt.savefig("limit_plots_long/data_vs_dm_mx_resid_%.1e_m_phi_%.2e.pdf"%(mx,m_phi))
        pdf.savefig()
    
    #plt.show()
    
    #plt.figure()
    #plt.errorbar(bc, h-gauss2(bc, *popt), yerr = np.sqrt(h), fmt = "o")#, color = "#7c1ee9")

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
    
    pts_for_lim = np.logical_not(np.isnan(logL_vec))
    logL_vec = logL_vec[pts_for_lim]
    alpha_vec = alpha_vec[pts_for_lim]

    
    logL_vec = 2*logL_vec
    ## throw out any spikes or non-increasing values
    for i in range(1,len(logL_vec)):

        if( i < len(logL_vec)-1 ):
            cmean = (logL_vec[i-1] + logL_vec[i+1])/2
            cdiff = np.abs(logL_vec[i-1] - logL_vec[i+1])
            if( np.abs(logL_vec[i]-cmean) > 5*cdiff ):
                logL_vec[i] = cmean

        ## fix error values from failed fits
        if ( np.abs(logL_vec[i] + 6508645.30457) < 0.3):
            logL_vec[i] = -6508649.64618
                
        if logL_vec[i] <= logL_vec[i-1]:
            logL_vec[i] = logL_vec[i-1] ## fill last largest val

            
    ## do hypothesis test relative to no DM:
    midx = 0 #np.argmin(logL_vec)
    minval = logL_vec[midx] #np.min(logL_vec)
    logL_vec = logL_vec - minval

    #pts_for_lim = logL_vec > 0
    #logL_vec = logL_vec[pts_for_lim]
    #alpha_vec = alpha_vec[pts_for_lim]    

    if(len(logL_vec) > 0):
        limval = np.interp(4, logL_vec[midx:], alpha_vec[midx:], left=np.nan, right=np.nan)
        xx = np.linspace(alpha_vec[midx], 1.1*limval, 1000)
    else:
        limval = np.nan
        xx = np.linspace(0, 1.1*limval, 1000)

    
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
    if( not make_spec_plot):
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
if( not make_spec_plot):
    plt.savefig("limit_plots_long/limit_plot_%.2e_nugg_frac_%.2e.pdf"%(m_phi, nugg_frac))
    pdf.savefig()

    pdf.close()
    np.savez("limit_plots_long/limit_data_%.2e_nugg_frac_%.2e.npz"%(m_phi, nugg_frac), mx_list=mx_list, limits=limits)

#plt.show()

