import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

def mad(arr, mu):
    return np.mean( np.abs( arr - mu ) )

def itersmooth(curve, show_plots=False):
    ## iteratively smooth and then throw out points far above the smooth curve

    xvec = np.arange(0, len(curve))

    ## walk along the curve and get neighboring points std

    nn = 5 ## num neighbors
    nel = len(curve)
    nsig = 10
    
    gpts = np.zeros_like( curve ) == 0
    for i in range(nel):

        if( i < 2*nn ):
            cmu = np.mean( np.hstack( (curve[0:nn], curve[(i+1):(2*nn+1)]) ))
            #cstd = np.std( np.hstack( (curve[0:nn], curve[(i+1):(2*nn+1)]) ))
            cstd = mad( np.hstack( (curve[0:nn], curve[(i+1):(2*nn+1)]) ), cmu)
        elif( i < len(curve)-2*nn):
            cmu = np.mean( np.hstack( (curve[(i-nn):i], curve[(i+1):(i+nn+1)]) ) )
            #cstd = np.std( np.hstack( (curve[(i-nn):i], curve[(i+1):(i+nn+1)]) ) )
            cstd = mad( np.hstack( (curve[(i-nn):i], curve[(i+1):(i+nn+1)]) ), cmu )
        else:
            cmu = np.std( np.hstack( (curve[(nel-2*nn):i], curve[(i+1):]) ) )
            #cstd = np.std( np.hstack( (curve[(nel-2*nn):i], curve[(i+1):]) ) )
            cstd = mad( np.hstack( (curve[(nel-2*nn):i], curve[(i+1):]) ), cmu )

        if( np.abs( curve[i]-cmu ) > nsig * cstd ):
            gpts[i] = False

    if(show_plots):
        plt.figure()
        plt.plot( xvec, curve )
        plt.plot( xvec[np.logical_not(gpts)], curve[np.logical_not(gpts)], 'rx' )
        plt.show()

    return gpts
                        
