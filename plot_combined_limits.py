import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline as us
import matplotlib.cm as cmx
import matplotlib.colors as colors

m_phi_list = [1, 1e-1, 1e-2, 0]
clist = ['k', 'b', 'r', 'g']

nugg_frac = 1

mchi = 1e-6 ## componenent mass, GeV

msi = 28.0
v0 = 1e-3
mn = 1
Eth = 500e-9
hbarc = 2e-14 ## GeV cm

def get_color_map( n ):
    jet = plt.get_cmap('Blues_r') 
    cNorm  = colors.Normalize(vmin=0, vmax=n-1)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    outmap = []
    for i in range(n):
        outmap.append( scalarMap.to_rgba(i) )
    return outmap

cs = get_color_map(len(m_phi_list)+2)

fig=plt.figure()

nugg_fig = plt.figure()

ff = np.load("fifth_force_limits.npz")

lw = 2
for c,m in zip(cs[3::-1], m_phi_list[::-1]):

    if(nugg_frac == 1):
        cdat = np.load("limit_plots_long/limit_data_%.2e.npz"%m)
    else:
        cdat = np.load("limit_plots_long/limit_data_%.2e_nugg_frac_%.2e.npz"%(m,nugg_frac))
        
    gpts = np.logical_not(np.isnan(cdat['limits']))

    ## correction factor for DAve's dumb mistake in mass:
    cdatl = np.array(cdat['limits']) * 7.5
    
    if( m == 0):
        ## bad points from spectra prodcued by plot_results.py
        skip_pts = [294, 582, 1987, 1e5, 1e6, 2.1545e6] #[43.5, 90.7, 582, 1311, 1987]
        for sp in skip_pts:
            gpts = np.logical_and( gpts, np.logical_not((np.abs(cdat['mx_list']-sp)/sp)<0.01) )
        gpts = np.logical_and( gpts, np.logical_or( cdat['mx_list']<0, cdat['mx_list']>260) )
        for sp in [67, 105]:
            gpts = np.logical_or( gpts, (np.abs(cdat['mx_list']-sp)/sp)<0.01)
        xx = np.logspace(np.log10(40),9,100)
        sfac = 0.35
        cdatm = np.array(cdat['mx_list'])
    elif( m == 1e-2):
        ## bad points from spectra prodcued by plot_results.py
        skip_pts = [189,254,294] #[1278,]
        if(len(skip_pts)>0):
            for sp in skip_pts:
                gpts = np.logical_and( gpts, np.logical_not((np.abs(cdat['mx_list']-sp)/sp)<0.01) )
        gpts = np.logical_and( gpts, np.logical_or( cdat['mx_list']<5e3, cdat['mx_list']>12e3) )
        gpts = np.logical_and( gpts, np.logical_or( cdat['mx_list']<1e3, cdat['mx_list']>1.6e3) )
        cdatm = np.hstack((55, np.array(cdat['mx_list'])))
        cdatl = np.hstack((7.5e-5, cdatl))
        gpts = np.hstack((True, gpts))
        xx = np.logspace(np.log10(40),9,100)
        sfac = 0.25  
    elif( m == 1e-1):
        ## bad points from spectra prodcued by plot_results.py
        skip_pts = [400, 380,]  #400,] #[80,] #[113,]
        if(len(skip_pts)>0):
            for sp in skip_pts:
                gpts = np.logical_and( gpts, np.logical_not((np.abs(cdat['mx_list']-sp)/sp)<0.01) )
        xx = np.logspace(np.log10(110),9,100)
        sfac = 0.25
        cdatm = np.array(cdat['mx_list'])
    else:
        #gpts = np.logical_and( gpts, np.logical_and( cdat['mx_list']>1300, cdat['mx_list']<1e4) )
        skip_pts = [5555,1e4]
        if(len(skip_pts)>0):
            for sp in skip_pts:
                gpts = np.logical_and( gpts, np.logical_not((np.abs(cdat['mx_list']-sp)/sp)<0.01) )
        gpts = np.logical_and( gpts, cdat['mx_list']>1.3e3 )
        xx = np.logspace(np.log10(800),9,100)
        index = np.argmin( np.abs(xx-4900) )
        xx = np.delete(xx, index)
        sfac = 0.65
        cdatm = np.array(cdat['mx_list'])


    

    spl = us( np.log10(cdatm[gpts]), np.log10(cdatl[gpts]), s=sfac )
    #spl = None
    #p = np.polyfit( np.log10(cdat['mx_list'][gpts]), np.log10(cdat['limits'][gpts]), 3)

    plt.figure(fig.number)

    if(False and m == 0):
        ## finely sampled, so only spline smooth above the step
        d1 = 10**spl(np.log10(xx))
        gpts = np.logical_and( gpts, np.logical_or( cdat['mx_list']<6e3, cdat['mx_list']>12e3) )
        spl3 = us( np.log10(cdat['mx_list'][gpts]), np.log10(cdat['limits'][gpts]), s=1, k=2 )
        d3 = 10**spl3(np.log10(xx))
        jidx = np.argmin( np.abs( xx - 3000) )
        #d1[jidx:] = d3[jidx:]
        #skidx = np.argmin( np.abs(xx - 2500) )
        #d1[skidx] = np.nan
        ppts = np.logical_not(np.isnan(d1))
        plt.loglog(xx[ppts], d1[ppts], label="$m_\phi$ = %g eV"%m, color=c, lw=lw)
    elif(False and  m == 1e-2):
        ## finely sampled, so only spline smooth above the step
        d1 = 10**spl(np.log10(xx))
        gpts = np.logical_and( gpts, np.logical_or( cdat['mx_list']<0, cdat['mx_list']>2.2e3) )
        spl3 = us( np.log10(cdat['mx_list'][gpts]), np.log10(cdat['limits'][gpts]), s=0.1, k=3 )
        d3 = 10**spl3(np.log10(xx))
        jidx = np.argmin( np.abs( xx - 2000) )
        #d1[jidx:] = d3[jidx:]
        skidx = np.argmin( np.abs(xx - 1770) )
        #d1[skidx] = np.nan
        skidx = np.argmin( np.abs(xx - 2096) )
        #d1[skidx] = np.nan
        ppts = np.logical_not(np.isnan(d1))
        plt.loglog(xx[ppts], d1[ppts], label="$m_\phi$ = %g eV"%m, color=c, lw=lw)
        #plt.loglog(xx[ppts], d1[ppts], '.', color=c)       
    else:
        if(not spl == None):
            plt.fill_between(xx, 10**spl(np.log10(xx)), 1e-4*np.ones_like(xx), color=c, alpha=0.4)
            plt.loglog(xx, 10**spl(np.log10(xx)), label="$m_\phi$ = %g eV"%m, color=c, lw=lw)
            print(m,c)
        #plt.loglog(xx, 10**np.polyval(p,np.log10(xx)), label="$m_\phi$ = %.0e eV"%m)
    
        #plt.loglog(cdat['mx_list'], cdat['limits'], 'o', color=c, mfc='none')

        #plt.loglog(cdat['mx_list'][gpts], cdat['limits'][gpts], 'o', color=c, label="$m_\phi$ = %.0e eV"%m)
    #plt.loglog(cdatm[gpts], cdatl[gpts], 'o', color=c, mfc='none')
    
    plt.figure(nugg_fig.number)
    ff_limit = np.interp(m, ff['x'][::-1], ff['y'][::-1] )
    plt.loglog(cdatm[gpts], 1e3*ff_limit*cdatm[gpts]/(4*np.pi*cdatl[gpts]), '-', color=c)

    # print("ff limit: ", m, ff_limit) 
    
    # plt.xlabel("DM Mass")
    # plt.ylabel("Constituent mass [MeV]")

    if(True and (m == 1e-1 or m==1) and not spl==None):
        np.savez("limits_mphi_%.0e_nugg_frac_%.2e.npz"%(m,nugg_frac), x=xx, y=10**spl(np.log10(xx)))
        print("saving new files")
    
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
plt.legend(loc="lower right", fontsize=9)
plt.xlabel("DM mass, $M_X$ [GeV]")
plt.ylabel(r"Limit on neutron coupling, $\alpha_n$")
#plt.title("Limits from 1 ng sphere, exposure = 20.6 min")
plt.xlim([1e1, 1e8])
plt.ylim([3e-9, 1e-4])
fig.set_size_inches(5,3.1)
plt.tight_layout(pad=0)

if(nugg_frac == 1):
    plt.savefig("combined_limit_plot.pdf") #, transparent=True)

## now nugget model


plt.show()

#plt.savefig("limit_plots/limit_plot_%.2e.pdf"%m_phi)
