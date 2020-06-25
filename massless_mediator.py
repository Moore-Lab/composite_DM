import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
import matplotlib.cm as cmx
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp2d
from scipy.integrate import quad
import pickle

## calculate dR/dq for a massless mediator

mxlist = [1e6,] #np.hstack((np.logspace(np.log10(25), 3, 20), np.logspace(3.1,9,20)))

vesc = 1.835e-3 ## galactic escape velocity
v0 = 7.34e-4 ## v0 parameter from Zurek group paper
ve = 8.014e-4 ## ve parameter from Zurek group paper
vmin = 1e-5
nvels = 10

rhoDM = 0.4 # dark matter mass density, GeV/cm^3

N_T = 4.5e15/2 ## number neutrons
alpha_n = 1e-10

hbarc = 0.2e-13 # GeV cm
c = 3e10 # cm/s
conv_fac = hbarc**2 * c * 3600  ## to give rate in cts/(GeV hr)

sig = 0.14 ## sigma of inloop

def get_color_map( n ):
    jet = plt.get_cmap('jet') 
    cNorm  = colors.Normalize(vmin=0, vmax=n-1)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    outmap = []
    for i in range(n):
        outmap.append( scalarMap.to_rgba(i) )
    return outmap

def eta(vmin):

    norm = v0**2 * np.pi * (np.sqrt(np.pi) * erf(vesc/v0) - 2*vesc/v0*np.exp(-(vesc/v0)**2))
    eta1 = np.pi*v0**2/(2*ve*norm) * (-4*np.exp(-(vesc/v0)**2) * ve + np.sqrt(np.pi)*v0*(erf((vmin+ve)/v0) - erf((vmin-ve)/v0)))
    eta2 = np.pi*v0**2/(2*ve*norm) * (-2*np.exp(-(vesc/v0)**2) * (vesc - vmin +ve) + np.sqrt(np.pi)*v0*(erf(vesc/v0) - erf((vmin-ve)/v0)))
    eta_out = np.zeros_like(vmin)
    gpts1 = vmin<vesc-ve
    gpts2 = np.logical_and(vmin>=vesc-ve,vmin<vesc+ve)
    eta_out[gpts1] = eta1[gpts1]
    eta_out[gpts2] = eta1[gpts2]

    return eta_out

def integrand(b, k, R, E):
    return np.sqrt(k + (b**2 * E/R) - E*R) * np.log(4*E*b**2 + R*(3*k-2*E*R) + 4*b*np.sqrt(E*R*(k + (b**2 * E/R) - E*R)))/(2*np.sqrt(-E*R*(1 - b**2/R**2 - k/(E*R))))
    

def theta( beta, gamma, sphedge ):
    int_bnd = -integrand(sphedge, beta, gamma)
    return np.pi/2 - int_bnd

def umax1( b, k, E, R):
    return np.sqrt( 1/(2*b**2) - 3*k/(4*b**2 * E * R) + np.sqrt(8*b**2 * E * k * R**3 + (3*k*R**2 - 2*E*R**3)**2)/(4*b**2 * E * R**3) )

def umax2( b, k, E, R):
    return 0.5*np.sqrt(2/b**2 - 3*k/(b**2 * E * R) - np.sqrt(R**3 * (8*b**2 * E*k + 9*k**2 * R - 12*E*k*R**2 + 4*E**2 * R**3))/(b**2 * E * R**3))

def nint( u, b, k, R, E):
    return b/np.sqrt(1 - b**2 * u**2 - k/(2*R*E)*(3-1/(u*R)**2))

prefac = alpha_n**2 * N_T**2

qq = np.linspace(0.05, 1e2, 1000)

plt.figure()

aa = np.logspace(-11, -4, 30)

qqmat, aamat = np.meshgrid(qq,aa)

clist = get_color_map(len(aa))

qsp = qq[1]-qq[0]
qkern = np.arange(-3*sig, 3*sig, qsp)
gkern = 1./(np.sqrt(2*np.pi)*sig) * np.exp( -qkern**2/(2*sig**2) ) * qsp


## now calculate the velocity distribution, following https://journals.aps.org/prd/pdf/10.1103/PhysRevD.100.035025
## note there is a typo in the sign of the exponential in the above paper
N0 = np.pi**1.5 * v0**3 * ( erf(vesc/v0) - 2/np.sqrt(np.pi) * (vesc/v0) * np.exp(-(vesc/v0)**2))
def f_halo(v):
    f1 = np.exp( -(v+ve)**2/v0**2 )*(np.exp(4*v*ve/v0**2)-1)
    f2 = np.exp( -(v-ve)**2/v0**2 ) - np.exp(-vesc**2/v0**2)

    f = np.zeros_like(v)

    g1 = v < (vesc - ve)
    g2 = np.logical_and( vesc-ve < v, v < vesc + ve)
    f[g1] = f1[g1]
    f[g2] = f2[g2]

    return f * np.pi * v * v0**2/(N0 * ve)

vel_list = np.linspace(vmin, vesc, nvels)

# ## plot the DM velocity distribution
# plt.figure()
# plt.plot( vel_list, f_halo(vel_list) )
# plt.title("Velocity distribution, integral = %.2f"%(np.trapz(f_halo(vel_list), vel_list)))
# plt.xlabel("Velocity")
# plt.ylabel("f(v)")
# plt.show()

bvec_cm = np.linspace(0,5e-4,2000)
bvec = bvec_cm/hbarc

bvec_cm_out = np.linspace(5e-4,20e-4,2000)
bvec_out = bvec_cm_out/hbarc

out_dict = {}
for mx in mxlist:
    out_dat = np.zeros((len(aa),len(qq)))

    # ## plot q vs b
    # plt.figure()
    # v1 = v0
    # bvec_cm = np.logspace(np.log10(5e-4), 0, 1e3)
    # bvec = bvec_cm/hbarc
    # p = mx*v1
    # E = 0.4*mx*v1**2
    # alpha = 1e-8
    # k = alpha * N_T
    # q = 2*p/np.sqrt(4*E**2 * bvec**2/k**2 + 1)
    # plt.figure()
    # plt.plot(bvec_cm*1e4, q)
    # plt.xlabel("impact param [um]")
    # plt.ylabel("momentum [gev]")

    # anum = k/(E*bvec)
    # plt.figure()
    # plt.plot(bvec_cm*1e4, 2*np.arcsin(anum/np.sqrt(4+anum**2)))
    # plt.xlabel("impact param [um]")
    # plt.ylabel("scatt angle [rad]")      
    # plt.show()

    #plt.figure()
    q_orig = {}
    q_out_orig = {}
    for aidx, alpha in enumerate(aa):
        print("Working on mass, alpha: ", mx, alpha)
        nX = rhoDM/mx
        #vmins = qq/(2*mx)
        #dRdq = (N_T*alpha_n)**2/(2*np.pi)/qq**3 * nX * eta(vmins) * conv_fac

        dRdq = np.zeros(len(qq))
        dRdq_in = np.zeros(len(qq))
        dRdq_mat = np.zeros((len(qq), nvels))
        dRdq_in_mat = np.zeros((len(qq), nvels))
        for vidx, vel in enumerate(vel_list):

            p = mx * vel
            
            ## cross section outside the sphere
            sigvals = (N_T*alpha)**2 * (2*np.pi) * 1/qq**3 * 1/vel * f_halo(vel)
            Ecm = 0.5*mx*vel**2
            k = alpha * N_T  ##/(4*np.pi) 4pi is already in the def of alpha
            bmin = 5e-4/hbarc ## 
            qmax = 2*mx*vel/np.sqrt(4*Ecm**2 * bmin**2/k**2 + 1)
            sigvals[qq > qmax] = 0
            dRdq_mat[:,vidx] = sigvals * nX * conv_fac

            if(aidx == 0):
                ##now inside
                cint = integrand(bvec, k, bmin, Ecm)
                numint = np.zeros_like(bvec)
                for jj,cb in enumerate(bvec):

                    umaxval = umax1( cb, k, Ecm, bmin)
                    ffn = lambda u: nint(u,cb,k,bmin,Ecm)
                    ival = quad(ffn, 1/bmin, umaxval)
                    numint[jj] = ival[0]

                alp = k/(Ecm * bvec)
                R = 1.0*bmin
                anint = np.arctan( (alp + 2*bvec/R)/(2*np.sqrt(1-alp*bvec/R-(bvec/R)**2)) ) - np.arctan(alp/2)
                theta = np.pi - 2*( numint + anint )

                alp = k/(Ecm * bvec_out)
                theta_out = 2*np.arcsin(alp/np.sqrt(4+alp**2))
                alp = k/(Ecm * bvec)
                theta_in_pt = 2*np.arcsin(alp/np.sqrt(4+alp**2))

                q = p * np.sqrt(2 * (1-np.cos(theta)))
                q_out = p * np.sqrt(2 * (1-np.cos(theta_out)))
                q_in_pt = p * np.sqrt(2 * (1-np.cos(theta_in_pt)))

                q_orig[vidx] = q
                q_out_orig[vidx] = q_out

                if(True):
                    afac = (alpha/aa[0])
                    gpts = np.logical_not(np.isnan(theta))
                    plt.figure()
                    plt.plot( bvec*hbarc*1e4, q, 'b')
                    plt.plot( bvec_out*hbarc*1e4, q_out, 'b')
                    plt.plot( bvec*hbarc*1e4, q_in_pt, "b:")
                    plt.plot( bvec*hbarc*1e4, q_orig[vidx]*afac, 'r:')
                    plt.plot( bvec_out*hbarc*1e4, q_out_orig[vidx]*afac, 'r:')
                    plt.ylim([0, np.max(q[gpts])*2])
                    plt.xlabel("Impact param, b [um]")
                    plt.ylabel(r"Momentum transfer, q [GeV]")
                    plt.title("Mass %.2e, alpha %.2e"%(mx,alpha))
                    plt.savefig("massless_plots/m%.2e_a%.2e_v%.4e.pdf"%(mx,alpha,vel))
                    plt.show()
                
            else:
                afac = (alpha/aa[0])
                q = q_orig[vidx] * afac
                
            gpts = np.logical_not(np.isnan(theta))
            dbdq = np.abs(np.gradient( bvec[gpts], q[gpts] ))
            
            maxpos = np.argmax(dbdq)
            dsigdq1 = np.interp(qq, q[gpts][:maxpos], 2*np.pi*bvec[gpts][:maxpos]*dbdq[:maxpos], left=0, right=0)
            dsigdq2 = np.interp(qq, q[gpts][maxpos:], 2*np.pi*bvec[gpts][maxpos:]*dbdq[maxpos:], left=0, right=0)
            dsigdq_tot = dsigdq1 + dsigdq2

            dRdq_in_mat[:,vidx] = dsigdq_tot * nX * conv_fac
            
            #if(True):
            #    plt.figure()
            #    plt.plot( qq, dsigdq_tot*hbarc**2)
            #    plt.show()
            
        for j in range( len(qq) ):
            dRdq[j] = np.trapz(dRdq_mat[j,:], vel_list)
            dRdq_in[j] = np.trapz(dRdq_in_mat[j,:], vel_list)

        if(True):
            plt.figure()
            plt.plot( qq, dRdq)
            plt.plot( qq, dRdq_in)
            plt.show()
        
        dRdq = np.convolve(dRdq+dRdq_in, gkern, mode='same')
        out_dat[aidx,:] = dRdq

        #fpts = qq > 2
        #print(alpha, np.trapz(dRdq[fpts], qq[fpts])) ##(alpha * N_T)**2/(8*np.pi) * 1/(2)**2 * 1/vel * nX * conv_fac * 5*24)
        
        #plt.semilogy( qq, dRdq, color=clist[aidx] )
        
    #plt.show()

    fig=plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for j,a in enumerate(aa):
        ax.plot( np.log10(qq), np.log10(np.ones_like(qq)*a), np.log10(out_dat[j,:]), 'o', color=clist[j])
        
    plt.title("Mass = %d"%mx)
    
    interp_fun = interp2d(np.log10(qq), np.log10(aa), out_dat)

    x = qq
    y = aa
    zz=interp_fun(np.log10(x), np.log10(y))
    xx,yy = np.meshgrid(x,y)
    
    ax.plot_wireframe(np.log10(xx),np.log10(yy),np.log10(zz))
        
    out_dict[mx] = interp_fun
    
    
    #plt.show()
    plt.close('all')

    # plt.figure()
    # plt.semilogy( qq, dRdq )
    # plt.xlabel('dp [GeV]')
    # plt.ylabel("Counts/(GeV hr)")
    # plt.xlim([0,10])
    # plt.show()
    
o = open("drdq_interp_grace_%.2e.pkl"%0, 'wb')
pickle.dump(out_dict, o)
o.close()
