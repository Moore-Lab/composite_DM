import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp2d
import pickle

m_phi = 5e-4 ##0

def sortfun(s):
    a = float(s[46:57])
    m = float(s[61:72])
    return m + a
    
flist = sorted(glob.glob('grace/data/mphi_%.0e/*.npz'%m_phi),key=sortfun)

qvec = []
sigvec = []
amvec = []
for f in flist:

    cdat = np.load(f)
    print(f)

    alpha = float(f[46:57])
    mx = float(f[61:72])

    amvec.append([mx, alpha])
    qvec.append(cdat['q'])
    sigvec.append(cdat['dsigdq'])

print(amvec)

amvec = np.array(amvec)
qvec = np.array(qvec)
sigvec = np.array(sigvec)

def get_color_map( n ):
    jet = plt.get_cmap('jet') 
    cNorm  = colors.Normalize(vmin=0, vmax=n-1)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    outmap = []
    for i in range(n):
        outmap.append( scalarMap.to_rgba(i) )
    return outmap

#mxlist = np.hstack( ((50, 100, 150, 200), (2.5e2, 5e2),  np.logspace(3, 6, 10) ) )
#mxlist = sorted(np.hstack((25, np.logspace(2,5,18),np.logspace(1,6,30))))
#mxlist = np.logspace(1,6,30)
#mxlist = np.logspace(2,5,18)
mxlist = np.logspace(1,8,42)

out_dict = {}
for mx in [137382.0,]: #mxlist:
    print(mx)

    if( np.abs(mx - 137382.0) > 1 ):
        print("Skipping ", mx)
        continue
    

    fig=plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    gidx = np.abs(amvec[:,0] - mx) < 1
    nm = np.sum(gidx)
    if( nm < 2): continue
    cmm = get_color_map(nm)
    qvals = np.logspace(-1, 1, 50)
    avals = amvec[gidx,1]
    out_dat = np.zeros((len(avals),len(qvals)))
    j=0
    for i, am in enumerate(amvec):
        if(not gidx[i]): continue

        print( am[1] )
        if( am[1] != 1e-10 ): continue
        plt.close('all')
        plt.figure()
        cdat = np.interp(np.log10(qvals), np.log10(qvec[i,:]), sigvec[i,:])
        plt.semilogy(qvals, cdat)
        plt.show()
        
        print(am[0])
        cdat = np.interp(np.log10(qvals), np.log10(qvec[i,:]), sigvec[i,:])
        ax.plot( np.log10(qvals), np.log10(np.ones_like(qvals)*am[1]), cdat, 'o', color=cmm[j])
        out_dat[j,:] = cdat
        j+=1
        
    interp_fun = interp2d(np.log10(qvals), np.log10(avals), out_dat) #, kind='cubic')

    x = np.logspace(-1,1,1e3)
    y = np.logspace( np.log10(avals[0]), np.log10(avals[-1]), 1e2)
    zz=interp_fun(np.log10(x), np.log10(y))
    xx,yy = np.meshgrid(x,y)
    
    ax.plot_wireframe(np.log10(xx),np.log10(yy),zz)
    #ax.set_zlim([0,1000])
        
    out_dict[mx] = interp_fun
    
    plt.title(str(mx))
    #plt.xlim([0.5,2])
    #plt.ylim([0,3])

if(False):
    o = open("drdq_interp_grace_%.2e.pkl"%m_phi, 'wb')
    pickle.dump(out_dict, o)
    o.close()
else:
    plt.show()
