import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline as us
import matplotlib.cm as cmx
import matplotlib.colors as colors

m_phi_list = [5e-1, 5e-2, 5e-3, 5e-4, 0]

mchi = 1e-6 ## componenent mass, GeV

msi = 28.0
v0 = 1e-3
mn = 1
Eth = 500e-9
hbarc = 2e-14 ## GeV cm

def get_color_map( n ):
    jet = plt.get_cmap('jet') 
    cNorm  = colors.Normalize(vmin=0, vmax=n-1)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    outmap = []
    for i in range(n):
        outmap.append( scalarMap.to_rgba(i) )
    return outmap

cs = get_color_map(len(m_phi_list))

fig=plt.figure()

nugg_fig = plt.figure()

ff = np.load("fifth_force_limits.npz")

for c,m in zip(cs, m_phi_list):

    cdat = np.load("limit_plots_long/limit_data_%.2e.npz"%m)

    gpts = np.logical_not(np.isnan(cdat['limits']))
    
    if( m == 0):
        skip_pts = []
        for sp in skip_pts:
            gpts = np.logical_and( gpts, np.logical_not(np.abs(cdat['mx_list']-sp)<1) )
        xx = np.logspace(np.log10(cdat['mx_list'][gpts][0]),9,100)
        sfac = 0.1
    elif( m == 5e-3):
        gpts = np.logical_and( gpts, np.logical_not(cdat['limits']<1e-8) )
        skip_pts = [2512,]
        for sp in skip_pts:
            gpts = np.logical_and( gpts, np.logical_not(np.abs(cdat['mx_list']-sp)<1) )
        sfac = 0.5
    elif( m == 5e-1):
        gpts = np.logical_and( gpts, np.logical_not(cdat['limits']<1e-8) )
        skip_pts = [3981,]
        for sp in skip_pts:
            gpts = np.logical_and( gpts, np.logical_not(np.abs(cdat['mx_list']-sp)<1) )
        sfac = 0.5
    else:
        xx = np.logspace(np.log10(cdat['mx_list'][gpts][0]),9,100)
        sfac = 0.5
        
    spl = us( np.log10(cdat['mx_list'][gpts]), np.log10(cdat['limits'][gpts]), s=sfac )
    #p = np.polyfit( np.log10(cdat['mx_list'][gpts]), np.log10(cdat['limits'][gpts]), 3)

    plt.figure(fig.number)

    if(m == 0):
        ## finely sampled, so only spline smooth above the step
        d1 = 10**spl(np.log10(xx))
        spl2 = us( np.log10(cdat['mx_list'][gpts]), np.log10(cdat['limits'][gpts]), s=0 )
        spl3 = us( np.log10(cdat['mx_list'][gpts]), np.log10(cdat['limits'][gpts]), s=0, k=1 )
        d2 = 10**spl2(np.log10(xx))
        d3 = 10**spl3(np.log10(xx))
        dd = np.logical_and( xx>560, xx<1e4 )
        dd3 = xx>1e4
        d2[dd] = d1[dd]
        d2[dd3] = d3[dd3]
        #plt.loglog(xx, d2, label="$m_\phi$ = %.0e eV"%m, color=c)
        plt.loglog(cdat['mx_list'][gpts], cdat['limits'][gpts]/1.4, '-', color=c, mfc='none')
        plt.loglog(cdat['mx_list'][gpts], cdat['limits'][gpts]/1.4, 'o', color=c, label="$m_\phi$ = %.0e eV"%m)
    else:
        print("")
        #plt.loglog(xx, 10**spl(np.log10(xx)), label="$m_\phi$ = %.0e eV"%m, color=c)
        
        #plt.loglog(xx, 10**np.polyval(p,np.log10(xx)), label="$m_\phi$ = %.0e eV"%m)
    
        plt.loglog(cdat['mx_list'][gpts], cdat['limits'][gpts], '-', color=c, mfc='none')

        plt.loglog(cdat['mx_list'][gpts], cdat['limits'][gpts], 'o', color=c, label="$m_\phi$ = %.0e eV"%m)
        
    plt.figure(nugg_fig.number)
    ff_limit = np.interp(m, ff['x'][::-1], ff['y'][::-1]) * 1/(4*np.pi)
    plt.loglog(cdat['mx_list'][gpts], 1e3*ff_limit*cdat['mx_list'][gpts]/(4*np.pi*cdat['limits'][gpts]), '-', color=c)
    plt.xlabel("DM Mass")
    plt.ylabel("Constituent mass [MeV]")
    
    # if( True and m == 5e-2 ):
    #    np.savez("proj_mphi_%.0e.npz"%m, x=xx, y=10**spl(np.log10(xx)))
    # else:
    #    np.savez("limits_mphi_%.0e.npz"%m, x=xx, y=10**spl(np.log10(xx)))
    # np.savetxt("limits_mphi_%.0e.csv"%m, np.vstack(( cdat['mx_list'][gpts], cdat['limits'][gpts])).T, delimiter=",")
    

    if( False and m == 5e-2 ):
        plt.figure(nugg_fig.number)
        yvals = 10**spl(np.log10(xx)) / (xx/mchi)

        plt.loglog(xx, yvals, color=c)
        ff_limit = np.interp(m, ff['x'][::-1], ff['y'][::-1]) * np.ones_like(xx) * 1/(4*np.pi)

        plt.plot(xx, ff_limit, '--', color=c)

        ## now the nuclear recoil cross section vs mass
        mu = msi * xx/(msi + xx)
        sig0 = 4*np.pi * (10**spl(np.log10(xx)))**2 * mu**2
        Rx = (9*np.pi* xx/(4*mchi**4))**(1.0/3)
        print(Rx)
        signr = sig0 * 3/(64* msi * Rx**4 * v0**2 * mn**5 * Eth**3 ) #* hbarc**2

        plt.figure()
        plt.loglog(xx, signr)
        
plt.figure(fig.number)
plt.legend(loc="lower right", fontsize=8)
plt.xlabel("Dark matter mass, $m_X$ [GeV]")
plt.ylabel(r"Upper limit on neutron coupling, $\alpha_n$")
#plt.title("Limits from 1 ng sphere, exposure = 20.6 min")
plt.tight_layout(pad=0)
plt.xlim([1e1, 1e9])
plt.ylim([1e-12, 1e-5])
#fig.set_size_inches(5,4)
plt.tight_layout()

plt.savefig("combined_limit_plot.pdf") #, transparent=True)

## now nugget model


plt.show()

#plt.savefig("limit_plots/limit_plot_%.2e.pdf"%m_phi)
