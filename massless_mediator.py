import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
import matplotlib.cm as cmx
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp2d
import pickle

## calculate dR/dq for a massless mediator

mxlist = np.hstack((np.logspace(np.log10(25), 3, 20), np.logspace(3.1,9,20)))

vesc = 1.835e-3 ## galactic escape velocity
v0 = 7.34e-4 ## v0 parameter from Zurek group paper
ve = 8.014e-4 ## ve parameter from Zurek group paper
vmin = 1e-5
nvels = 1000

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

def integrand(x, beta, gamma):
    print( (x**2 * (x**2 + beta - 1) - gamma)/(1 - beta + gamma/x**2 - x**2) )
    input()
    print( 1 - beta - 2*x**2 - 2*np.sqrt( x**4 + x**2 * (beta-1) - gamma ))
    input()
    return 1/(2*x) * np.sqrt((x**2 * (x**2 + beta - 1) - gamma)/(1 - beta + gamma/x**2 - x**2)) * np.log( 1 - beta - 2*x**2 - 2*np.sqrt( x**4 + x**2 * (beta-1) - gamma) )

def theta( beta, gamma, sphedge ):
    int_bnd = -integrand(sphedge, beta, gamma)
    return np.pi/2 - int_bnd

def xmax( beta, gamma ):
    return np.sqrt( 0.5*(1 - beta + np.sqrt(1 - 2*beta + beta**2 + 4*gamma)) )

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

bvec_cm = np.linspace(0,5e-4,1e3)
bvec = bvec_cm/hbarc

out_dict = {}
for mx in mxlist:
    print("Working on mass: ", mx)
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
    for aidx, alpha in enumerate(aa):

        nX = rhoDM/mx
        #vmins = qq/(2*mx)
        #dRdq = (N_T*alpha_n)**2/(2*np.pi)/qq**3 * nX * eta(vmins) * conv_fac

        dRdq = np.zeros(len(qq))
        dRdq_mat = np.zeros((len(qq), nvels))
        for vidx, vel in enumerate(vel_list):
            ## cross section outside the sphere
            sigvals = (N_T*alpha)**2 * (2*np.pi) * 1/qq**3 * 1/vel * f_halo(vel)
            Ecm = 0.5*mx*vel**2
            k = alpha * N_T  ##/(4*np.pi) 4pi is already in the def of alpha
            bmin = 5e-4/hbarc ## 
            qmax = 2*mx*vel/np.sqrt(4*Ecm**2 * bmin**2/k**2 + 1)
            sigvals[qq > qmax] = 0
            dRdq_mat[:,vidx] = sigvals * nX * conv_fac

            ##now inside
            beta = 3*k/(2*Ecm*bmin)
            gamma = k*bvec**2/(2*Ecm*bmin**3)

            print(beta)
            input()
            print(gamma)
            input()
            
            xmax_vals = xmax(beta, gamma)
            theta_vec = theta( beta, gamma, bvec/bmin )
            
            plt.figure()
            plt.plot( bvec, theta_vec)
            plt.show()
            
        for j in range( len(qq) ):
            dRdq[j] = np.trapz(dRdq_mat[j,:], vel_list)

        dRdq = np.convolve(dRdq, gkern, mode='same')
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
