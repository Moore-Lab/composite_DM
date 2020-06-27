import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp2d
import pickle

m_phi = 0 #5e-4

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

    print(np.shape(cdat['dsigdq']))
    
    amvec.append([mx, alpha])
    qvec.append(cdat['q'])
    #qvec = np.vstack( (qvec, cdat['q']) )
    #sigvec = np.vstack( (sigvec, cdat['dsigdq']) )
    sigvec.append(cdat['dsigdq'])

amvec = np.array(amvec)
#qvec = np.array(qvec)
#sigvec = np.array(sigvec)


def get_color_map( n ):
    jet = plt.get_cmap('jet') 
    cNorm  = colors.Normalize(vmin=0, vmax=n-1)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    outmap = []
    for i in range(n):
        outmap.append( scalarMap.to_rgba(i) )
    return outmap

mxlist = np.unique( amvec[:,0] )
amlist = np.unique( amvec[:,1] )

print(mxlist)

#mxlist = np.logspace(1,8,42)

print(mxlist)
print(amlist)

bad_pairs = [[56.8482, 0, 1e-4],
             [80.4749, 0, 1e-4],
             [113.921, 0, 1e-10],
             [29627.7, 0, 1e-8],
             [59372.4, 0, 1e-8],
             [84048.2, 0, 1e-8],
             [118979.0, 0, 1e-8],
             [168428.0, 0, 1e-8],
             [238429.0, 0, 1e-8],
             [337522.0, 0, 1e-8],
             [676377.0, 0, 1e-5],
             [957485.0, 0, 1e-5],
             [1918750.0, 0, -1],
             [2716200.0, 0, -1],
             [3845080.0, 0, -1],
             [5443130.0, 0, -1],
             [7705350.0, 0, -1],
             [10907800.0, 0, -1],
             [15441100.0, 0, -1],
             [21858600.0, 0, -1],
             [30943300.0, 0, -1],
             [43803600.0, 0, -1],
             [62008700.0, 0, -1],
             [87780100.0, 0, -1],
             [124262000.0, 0, -1],
             [175907000.0, 0, -1],]
             

## cleaning step
if(False):
    for mx in mxlist:

        plt.figure()
        for am in amlist:

            for i in range(len(amvec)):

                if( not (amvec[i][0]==mx and amvec[i][1]==am) ): continue

                plt.loglog( qvec[i], sigvec[i], label=am )

        plt.legend()
        plt.title(mx)

        plt.show()

def bad_check(m, a, mp):
    is_bad = False
    for b in bad_pairs:
        if( (np.abs(m-b[0])<1.0 or b[0]==-1) and (mp == b[1] or b[1]==-1) and (a==b[2] or b[2]==-1) ):
            is_bad = True
            break
    return is_bad
            
out_dict = {}
for mx in mxlist:
    print(mx)

    fig=plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    gidx = np.abs(amvec[:,0] - mx) < 1
    nm = np.sum(gidx)
    if( nm < 2): continue
    cmm = get_color_map(nm)
    qvals = np.logspace(np.log10(0.25), 1, 50)
    #avals = amvec[gidx,1]
    out_dat = np.array([]) #np.zeros((len(avals),len(qvals)))
    j=0
    avals = []
    for i, am in enumerate(amvec):
        if(not gidx[i]): continue

        if(bad_check( mx, am[1], m_phi ) ):
            print("Bad value: ", mx, am[1], m_phi )
            continue
        
        print( am[1] )

        gpts = np.logical_not( np.isnan(sigvec[i]) )
        gpts = np.logical_and( sigvec[i] > 0, gpts )
        if(np.sum(gpts)==0): continue
        cdat = np.interp(np.log10(qvals), np.log10(qvec[i][gpts]), np.log10(sigvec[i][gpts]), right=-10, left=-10)
        ax.plot( np.log10(qvals), np.log10(np.ones_like(qvals)*am[1]), cdat, 'o', color=cmm[j])
        if(len(out_dat)==0):
            out_dat = cdat
        else:
            out_dat = np.vstack((out_dat, cdat))
        avals.append( am[1] )
        j+=1

    if(j==0): continue
        
    interp_fun = interp2d(np.log10(qvals), np.log10(avals), out_dat, fill_value=-10) #, kind='cubic')

    x = np.logspace(np.log10(0.25),1,1e3)
    y = np.logspace( np.log10(avals[0]), np.log10(avals[-1]), 1e2)
    zz=interp_fun(np.log10(x), np.log10(y))
    xx,yy = np.meshgrid(x,y)

    print(zz)
    
    ax.plot_wireframe(np.log10(xx),np.log10(yy),zz)
    #ax.set_zlim([0,1000])
        
    out_dict[mx] = interp_fun
    
    plt.title(str(mx))
    #plt.xlim([0.5,2])
    #plt.zlim([-2,6])
    if( True ):
        plt.close('all')
    else:
        plt.show()
    
if(True):
    o = open("drdq_interp_grace_%.2e.pkl"%m_phi, 'wb')
    pickle.dump(out_dict, o)
    o.close()
else:
    plt.show()
