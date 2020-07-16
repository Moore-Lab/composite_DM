import sys, os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.integrate import quad
import matplotlib.cm as cmx
import matplotlib.colors as colors
import scipy.signal as sig
from scipy.special import erf
from itersmooth import itersmooth

def run_nugget_calc(M_X_in, alpha_n_in, m_phi=0.05, debug=False, show_plots=False, smooth=True):

    hbarc = 0.2 # eV um

    M_X = M_X_in * 1e9 ## Dark matter nugget mass, eV (assumes mass in GeV given on command line)
    m_chi = 0.01 * 1e9 ## eV
    N_chi = M_X/m_chi ## number of dark matter particles in the nugget

    #m_phi = 0.05 ## mediator mass, eV

    y_chi = 1 ## dark matter dimensionless coupling
    y_T = 2e-15 ## normal matter dimensionless coupling

    ## this is the single neutron coupling, y_T*N_chi*y_chi/4*pi
    ## in the Zurek group model it is derived from the parameters above,
    ## however, here we are agnostic and just take a value from teh command line

    alpha_n = alpha_n_in #y_T*N_chi*y_chi/(4*np.pi) 

    R_um = 5 ## sphere radius, um
    R = R_um/hbarc ## radius in natural units, eV^-1

    rho_T = 2.0e3 ## kg/m^3
    mAMU = 1.66e-27
    N_T = 0.5*(4/3*np.pi*(R_um*1e-6)**3)*rho_T/mAMU

    res = 170e6 ## detector resolution in eV

    rhoDM = 0.3e9 # dark matter mass density, eV/cm^3
    q_thr = 0.05e9 # momentum threshold, eV



    ############################################
    print("Starting parameters:  ", M_X, alpha_n)
    print("N_T is: %.3e"%N_T)

    outdir = "data/mphi_%.0e"%m_phi
    if(not os.path.isdir(outdir)):
        os.mkdir(outdir)
        os.mkdir(outdir + "/plots")
    
    #alpha = N_T * N_chi * y_T * y_chi/(4*np.pi)
    alpha = alpha_n * N_T

    mR = m_phi*R

    ## function giving the exact potential for a uniform density sphere with Yukawa force
    def vtot(u):
        if(u <= 0):
            return np.inf
        elif(u < 1/R):
            if(mR > 0):
                return 3*alpha/mR**3 * (mR * np.cosh(mR) - np.sinh(mR)) * np.exp(-m_phi/u) * u
            else:
                return alpha * u
        else:
            if(mR > 0):
                return 3*alpha/mR**3 * (m_phi - u*(1 + mR)/(1+1./np.tanh(mR)) * (np.sinh(m_phi/u)/np.sinh(mR)))
            else:
                return alpha/2 * (3/R - 1./(R**3 * u**2))

    ## integrand needed for finding the scattering angle as a function of impact parameter
    def integ(u, b, E):

        sval = 1 - vtot(u)/E - (b*u)**2
        #print(u,b,E,vtot(u)/E, b*u, sval)
        if(sval >= 0):
            integ = b/np.sqrt(sval)
        else:
            integ = 0
        return integ

    def integ_out(u, b, E):
        if(u <= 0):
            sval = 0
        else:
            if(mR > 0):
                sval = 3*alpha/mR**3 * (mR * np.cosh(mR) - np.sinh(mR)) * np.exp(-m_phi/u) * u
            else:
                sval = alpha * u
        return b/np.sqrt(1 - sval/E - (b*u)**2)

    def integ_in(u, b, E):
        if(u <= 0):
            sval = 0
        else:
            if(mR > 0):
                sval = 3*alpha/mR**3 * (m_phi - u*(1 + mR)/(1+1./np.tanh(mR)) * (np.sinh(m_phi/u)/np.sinh(mR)))
            else:
                sval = alpha/2 * (3/R - 1./(R**3 * u**2))
        return b/np.sqrt(1 - sval/E - (b*u)**2)

        
    ## function to numerically find the maximum value of u = 1/r, not used currently
    def minr_func(u,b,E):
        return np.abs( 1 - (b*u)**2 - vtot(u)/E)

    # make a list of impact parameters over which to calculate the scattering angle
    if(mR > 0):
        b_vec_um = np.logspace(-3,3,2000)
    else:
        b_vec_um = np.logspace(-3,5,2000)

    #  analyitical formula for finding the maximum value of u = 1/r
    def min_u_exact(b, E):
        if b < R:
            outval = b*np.sqrt(E)/np.sqrt(E - ((3*alpha)/(mR**3)*(m_phi - ((1 + mR)*np.sinh(m_phi*b)/np.sinh(mR))/(b*(1+1/np.tanh(mR))))))
        else:
            outval = b*np.sqrt(E)/np.sqrt(E - (3*alpha*np.exp(-m_phi*b))/(b * mR**3) * (mR*np.cosh(mR)-np.sinh(mR)))
        return 1./outval

    def get_color_map( n ):
        jet = plt.get_cmap('jet') 
        cNorm  = colors.Normalize(vmin=0, vmax=n-1)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
        outmap = []
        for i in range(n):
            outmap.append( scalarMap.to_rgba(i) )
        return outmap

    ## function used to smear the spectrum with gaussian resolution
    def res_fun( x, sig ):
        return 1.0/(sig * np.sqrt(2*np.pi)) * np.exp(-0.5*(x/sig)**2)

    #####################################################

    nvels = 100  ## number of velocities to include in integration
    vmin = 5e-5  ## minimum velocity to consider, natural units
    vesc = 1.815e-3 ## galactic escape velocity
    v0 = 7.34e-4 ## v0 parameter from Zurek group paper
    ve = 8.172e-4 ## ve parameter from Zurek group paper

    pmax = np.max((vesc * M_X, 10e9))
    
    cmap = get_color_map(nvels)

    vel_list = np.linspace(vmin, vesc, nvels)
    
    cxfig = plt.figure()  ## cross section figure
    cx_check = [] ## values of cross section, integrated either over b or q
    dsigdq_mat = [] ## differential cross section for each velocity
    for c,v in zip(cmap, vel_list):

        print("Working on velocity: %f"%v)
        p = M_X * v  ## initial momentum

        theta_vec = []
        E = 1./2 * M_X * v**2  ## initial kinetic energy of incoming particle
        for j,b_um in enumerate(b_vec_um):

            b = b_um/hbarc

            ## numerically solve for the max value of u (not currently used)
            min_un = minimize_scalar(minr_func, args=(b,E),bounds=(hbarc/1e3,hbarc/1e-3),method='bounded',options={'disp': 0, 'xatol': 1e-15})
            min_u = min_un.x

            ## use the exact formula to get max value of U
            #min_u = min_u_exact(b,E)


            #print(min_u, min_u2, np.abs(min_u-min_u2)/min_u)
            #out_us.append([b,min_u, min_u2])

            if(False):
                rvec_um = np.linspace(0.1,10,1e5)
                uvec = hbarc/rvec_um
                ff = np.zeros_like(uvec)
                for i in range(len(uvec)):
                    ff[i] = minr_func(uvec[i], b, E)
                plt.figure()
                plt.plot( uvec, ff)
                ii = np.argmin(np.abs(uvec-min_un))
                plt.plot( uvec[ii], ff[ii], 'ro')
                #plt.ylim([-0.1, 100])
                plt.show()

            ## split the integral into two pieces, to handle the diverging piece only over a small range
            ## this substantially improves the performance of quad
            int_res1 = quad(integ, min_u/1e7, min_u*0.9, args=(b,E)) #, limit=1000, full_output=1)
            int_res2 = quad(integ, min_u*0.9, min_u, args=(b,E)) #, limit=1000, full_output=1)
            int_res = int_res1[0] + int_res2[0]

            ## now try to be smarter:
            if( min_u < 1./R ):
                int_res1_check = quad(integ_out, 0, min_u, args=(b,E))
                int_res_check = int_res1_check[0]
            else:
                int_res1_check = quad(integ_out, 0, 1./R, args=(b,E))
                int_res2_check = quad(integ_in, 1./R, min_u, args=(b,E))
                int_res_check = int_res1_check[0] + int_res2_check[0]

            if(False):
                plt.figure()
                uvec = np.linspace(min_u*0.999, min_u,10000)
                ff = np.zeros_like(uvec)
                for i in range(len(uvec)):
                    ff[i] = integ(uvec[i], b, E)
                plt.plot( uvec,  1/ff, label=str(np.pi - 2*int_res))

                plt.figure()
                print("Max U: ", min_u, 1/R)
                if(min_u < 1./R):
                    t1vec = np.linspace(0, min_u,100000)
                    ff = np.zeros_like(t1vec)
                    for i in range(len(t1vec)):
                        ff[i] = integ_out(t1vec[i], b, E)
                    plt.plot(t1vec, ff )
                else:
                    t1vec = np.linspace(0, 1./R,1e5)
                    t2vec = np.linspace(1./R, min_u, 100000)
                    ff = np.zeros_like(t1vec)
                    for i in range(len(t1vec)):
                        ff[i] = integ_out(t1vec[i], b, E)
                    plt.plot(t1vec, ff )
                    ff = np.zeros_like(t2vec)
                    for i in range(len(t2vec)):
                        ff[i] = integ_out(t2vec[i], b, E)
                    plt.plot(t2vec, ff )
                
                plt.show()

            #print("int check: ", int_res, int_res_check, np.abs(int_res-int_res_check)/int_res)
            if( np.abs(int_res-int_res_check)/int_res > 1e-5 ): int_res = np.nan
            theta_vec.append(np.pi - 2*int_res) ## this is the scattering angle at this b

        theta_vec = np.array(theta_vec)
        ## remove any nans
        good_pts = np.logical_not( np.logical_or(np.isnan(theta_vec), np.isinf(theta_vec)) )
        theta_vec = theta_vec[good_pts]
        b_vec_um = b_vec_um[good_pts]
        
        if(smooth == True):
            smooth_pts = itersmooth(theta_vec, show_plots=True)
            theta_vec = theta_vec[smooth_pts]
            b_vec_um = b_vec_um[smooth_pts]           
        
        
        if(debug):
            plt.figure()
            plt.plot(b_vec_um, theta_vec)
            plt.xlabel("Impact parameter, b [um]")
            plt.ylabel("Scattering angle [rad]")
            #plt.show()
            
        ## for most values of v, there is a maximum in the theta vs b plot
        ## find the b value that corresponds to that maximum (bcrit) so that
        ## we can split up the contributions to the cross section below and above
        b_vec = b_vec_um/hbarc
        bcidx = np.argmax( theta_vec )
        bcrit = b_vec[bcidx]
        bcrit_um = b_vec_um[bcidx]

        ## now need the cross section above and below bcrit
        b1, t1 = b_vec[:bcidx], theta_vec[:bcidx]
        b2, t2 = b_vec[bcidx:], theta_vec[bcidx:]

        ## momentum transfer as a function of scattering angle
        q1 = p * np.sqrt( 2*(1-np.cos(t1)) )
        q2 = p * np.sqrt( 2*(1-np.cos(t2)) )
        q = p * np.sqrt( 2*(1-np.cos(theta_vec)) )

        ## find only the points above the desired threshold
        gidx1 = q1 > q_thr
        gidx2 = q2 > q_thr

        if(len(b1) > 1 ):
            db1 = np.abs(np.gradient(b1, q1))
            gidx1 = np.logical_and( gidx1, np.logical_not(np.isnan(db1)) )
            gidx1 = np.logical_and( gidx1, np.logical_not(np.isinf(db1)) )
        else:
            db1 = 0
        db2 = np.abs(np.gradient(b2, q2))
        gidx2 = np.logical_and( gidx2, np.logical_not(np.isnan(db2)) )
        gidx2 = np.logical_and( gidx2, np.logical_not(np.isinf(db2)) )
            
        ## make sure we limit the second part to only the first time we fall below threshold
        first_fall = np.argwhere(np.logical_and( q2>q_thr, np.roll(q2, -1)<q_thr))
        if(len(first_fall > 0)):
            gidx2 = np.logical_and(gidx2, b2<b2[first_fall[0]])

        ## if there's a peak, fit the region around the peak to find the max value allowed.
        ## this ensures the numerical derivative is limited to the appropriate bin size and
        ## the cross section doesn't get too large due to numerical artifacts

        qq = np.linspace(q_thr, 2*pmax*1.1, 10000)

        if( np.sum( gidx2 ) == 0 ):
            dsigdq_mat.append( np.zeros_like(qq) )
            continue

        ## find the local sigma and throw out points much above it
        bad_vec = np.ones_like( b2 )
        for jj in range(len(b2)):
            min_idx = np.max( (0, jj-3) )
            max_idx = np.min( (len(b2), jj+3) )
            cneigh = np.hstack( (q2[min_idx:jj],q2[(jj+1):max_idx]) )
            cneigh = cneigh[np.logical_not(np.isnan(cneigh))]
            cneigh = cneigh[np.logical_not(np.isinf(cneigh))]
            curr_std = np.std(cneigh)
            curr_mean = np.median(cneigh)
            if( q2[jj]-curr_mean > 10*curr_std ):
                bad_vec[jj] = 0

        gidx2 = np.logical_and(gidx2, bad_vec)        

        ## resolution function for smearing with same spacing
        bsize = qq[1]-qq[0]
        gauss_x = np.arange(-3*res, 3*res, bsize)
        gauss_blur = res_fun(gauss_x, res)*bsize
        #print("guass_norm: ", np.sum(gauss_blur))

        ## in order to make the divergence in dsig/dq correct at the last bin, fit the peak
        ## to a parabola, and estimate the max deriviate in one q bin
        if( len(b1) > 6 ):

            nsides = 2
            pfit = np.polyfit(b_vec[(bcidx-nsides):(bcidx+nsides+1)], q[(bcidx-nsides):(bcidx+nsides+1)], 2 )

            max_loc = -pfit[1]/(2*pfit[0])
            delt_x = np.sqrt( np.abs(bsize/pfit[0]) )
            max_deriv = delt_x/bsize

            if(debug):
                plt.figure()
                plt.plot( b_vec[(bcidx-nsides):(bcidx+nsides+1)], q[(bcidx-nsides):(bcidx+nsides+1)],  'ko')
                xx = np.linspace( b_vec[(bcidx-nsides)], b_vec[(bcidx+nsides+1)], 100)
                plt.plot(xx, np.polyval(pfit, xx))
                plt.plot( [max_loc, max_loc+delt_x], [np.polyval( pfit, max_loc), np.polyval( pfit, max_loc+delt_x)], 'bo')


            #print(len(b1), np.max(db1), np.max(db2), max_deriv)
            ## Limit the divergence to the max per bin size
            db1[ db1 > max_deriv] = max_deriv
            db2[ db2 > max_deriv] = max_deriv

        if(debug):
            plt.figure()
            plt.plot(b1[gidx1],q1[gidx1])
            plt.plot(b2[gidx2], q2[gidx2])
            plt.xlabel("Impact parameter")
            plt.ylabel("Momentum")
            plt.title("q vs b, split across peak")

        ## now resample the momentum to a uniform spacing
        if( len(b1[gidx1]) > 1):
            dsigdq1 = np.interp(qq, q1[gidx1], 2*np.pi*b1[gidx1]*db1[gidx1], left=0, right=0)
        else:
            dsigdq1 = 0
        q2, b2, db2 = q2[gidx2], b2[gidx2], db2[gidx2]
        dsigdq2 = np.interp(qq, q2[::-1], 2*np.pi*b2[::-1]*db2[::-1], left=0, right=0)

        if(debug):
            plt.figure()
            plt.plot(q1, b1)
            plt.plot(q2, b2)
            plt.ylabel("Impact parameter")
            plt.xlabel("Momentum")
            plt.title("b vs q, split across peak")
            
        ## integrate the differential cross section to get the total cross sectino
        st = np.trapz((dsigdq1+dsigdq2) * hbarc**2, qq)

        ## sum the pieces above and below the peak
        dsigdq_tot = dsigdq1+dsigdq2

        ## make sure that your q vector has a high enough limit at this velocity -- if you get this warning
        ## probably want to increase the upper limit on the qq vector above
        if(dsigdq_tot[-1] != 0):
            print("Warning -- momentum vector upper limit to low, v=%f"%v)

        ## plot dsig/dq for each velocity considered
        plt.figure(cxfig.number)
        plt.semilogy(qq/1e9, dsigdq_tot * hbarc**2 * 1e9, color=c, label="Vel: %.1e$c$, sig_tot=%.1f$\mu$m$^2$"%(v,st))
        plt.xlabel("Momentum [GeV]")
        plt.ylabel("d$\sigma$/dq [$\mu$m$^2$/GeV]")

        ## now convolve with gaussian of proper width
        #dsigdq_tot_smear = np.convolve(dsigdq_tot, gauss_blur, mode='same')
        dsigdq_tot_smear = dsigdq_tot
        plt.plot(qq/1e9, dsigdq_tot_smear * hbarc**2 * 1e9, ':', color=c)
        plt.title(r"$\alpha_n$ = %.2e per neutron, M$_X$ = %.2e GeV"%(alpha_n, M_X/1e9))
        plt.xlim([0,100])
        plt.ylim([1e-2,1e6])


        ## now compare the total cross section integrated with respect to b or to q -- these should
        ## agree with each other
        thr_idx = q >= q_thr
        st_b = 2*np.pi*np.trapz( b_vec_um[thr_idx], b_vec_um[thr_idx])
        print(" Total cross section: ", st, st_b, np.abs(st-st_b)/st)
        cx_check.append([v, st, st_b])

        dsigdq_mat.append( dsigdq_tot_smear )

        if(debug):
            plt.show()

    if( len(cx_check) == 0):
        return
    dsigdq_mat = np.array(dsigdq_mat)

    ## color bar for the dsig/dq plot
    norm = colors.Normalize(vmin=vel_list[0],vmax=vel_list[-1])
    sm = cmx.ScalarMappable(cmap=plt.get_cmap('jet'), norm=norm)
    sm.set_array([])
    cb = plt.colorbar(sm)
    cb.set_label("Velocity")
    plt.tight_layout()
    cxfig.set_size_inches(20,6)

    if(not debug):
        plt.savefig(outdir + "/plots/cross_sec_vs_v_alpha_%.2e_MX_%.2e.pdf"%(alpha_n, M_X/1e9))

    ## check whether the two integrated cross sections agree
    cx_check = np.array(cx_check)
    plt.figure()
    plt.plot(cx_check[:,0], cx_check[:,1], 'k.', label='dq')
    plt.plot(cx_check[:,0], cx_check[:,2], 'r.', label='db')
    plt.xlabel("Velocity")
    plt.ylabel("Integrated cross section")
    plt.legend()
    #plt.show()

    ## now calculate the velocity distribution, following https://journals.aps.org/prd/pdf/10.1103/PhysRevD.100.035025
    ## note there is a typo in the sign of the exponential in the above paper
    N0 = np.pi**1.5 * v0**3 * ( erf(vesc/v0) - 2/np.sqrt(np.pi) * (vesc/v0) * np.exp(-(vesc/v0)**2))
    def f_halo(v):
        f1 = np.exp( -(v+ve)**2/v0**2 )*(np.exp(4*v*ve/v0**2)-1)
        f2 = np.exp( -(v-ve)**2/v0**2 ) - np.exp(-vesc**2/v0**2)

        f = np.zeros_like(v)

        g1 = v < (vesc - ve)
        g2 = np.logical_and( vesc-ve < v, v < vesc + ve)
        print(v)
        print(vesc-ve)
        f[g1] = f1[g1]
        f[g2] = f2[g2]

        return f * np.pi * v * v0**2/(N0 * ve)

    ## plot the DM velocity distribution
    plt.figure()
    plt.plot( vel_list, f_halo(vel_list) )
    plt.title("Velocity distribution, integral = %.2f"%(np.trapz(f_halo(vel_list), vel_list)))
    plt.xlabel("Velocity")
    plt.ylabel("f(v)")

    conv_fac = hbarc**2 * 1e9 * 3e10 * 1e-8 * 3600  # natural units -> um^2/GeV, c [cm/s], um^2/cm^2, s/hr

    ## integrand for differential rate
    int_vec = rhoDM/M_X * vel_list * f_halo(vel_list)

    ## total rate found by integrating over the cross section
    tot_xsec = np.zeros_like(qq)
    for j in range(len(tot_xsec)):
        tot_xsec[j] = np.trapz( int_vec * dsigdq_mat[:,j], vel_list )

    plt.figure()
    plt.plot( qq/1e9, tot_xsec*conv_fac)
    plt.xlabel("Momentum [GeV]")
    plt.ylabel("dR/dq [counts/hr/GeV]")
    plt.title(r"$\alpha_n$ = %.2e per neutron, M$_X$ = %.2e GeV"%(alpha_n, M_X/1e9))
    plt.xlim([0,10])
    plt.tight_layout()
    
    plt.close('all')
    if(not debug):
        #plt.savefig(outdir + "/plots/differential_rate_alpha_%.2e_MX_%.2e.pdf"%(alpha_n, M_X/1e9))

        np.savez(outdir + "/differential_rate_alpha_%.5e_MX_%.5e.npz"%(alpha_n, M_X/1e9), q=qq/1e9, dsigdq = tot_xsec*conv_fac)

    if(show_plots):
        plt.show()

if __name__ == "__main__":

    M_X_in = float(sys.argv[1])
    alpha_n_in = float(sys.argv[2])
    m_phi = float(sys.argv[3])
    print(M_X_in, alpha_n_in)
    run_nugget_calc(M_X_in, alpha_n_in, m_phi=m_phi, debug=False, show_plots=False)
