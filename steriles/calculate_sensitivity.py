import sys, pickle
import numpy as np
import usphere_utils as uu

## using pdfs calculated from calculate_pdfs.py, calculate the sensitivity
## for a given isotope as a function of mass

def calc_limit(Q, t12, A, loading_frac, num_spheres, livetime, bkg_pdf, sig_pdf, eta_xyz=[0.6,0.6,0.6], f0=1e5, ang_error = 0.01, nbins=100):

    m_sph = 4/3 * np.pi * uu.sphere_radius**3 * uu.rho
    n_nuclei = m_sph * uu.N_A/A * loading_frac * num_spheres
    n_decays = int(n_nuclei * (1 - 0.5**(livetime/t12) ))

    m_sterile = np.linspace(0, Q, int(50))

    bkg_pdf_x = bkg_pdf[:,0] ## require bkg and sig pdf to have same x values (by construction in calculate pdfs)

    bkg_only_cts = uu.draw_from_pdf( n_decays, bkg_pdf_x, bkg_pdf[:,1] )

    p_res = np.sqrt(uu.hbar * m_sph * 2*np.pi*f0)/uu.kg_m_per_s_to_keV
    hh, be = np.histogram(bkg_only_cts, bins = np.arange(bkg_pdf_x[0], bkg_pdf_x[-1], p_res/4))
    bc = be[:-1] + np.diff(be)/2

    ue4, prof = uu.profile_sig_counts(bc, hh, bkg_pdf_x, bkg_pdf[:,1], sig_pdf[:,1])
        
    ## find the min of the profile
    midx = np.argmin(prof)
    ue4 = ue4[midx:]
    prof = prof[midx:]

    if(len(prof) == 0): ## min not found early enough
      ulim = np.nan
    else:
      ulim = np.interp(uu.conf_lev, prof, ue4, left=np.nan, right=np.nan)
        
    return m_sterile, ulim


if(len(sys.argv)==1):
    iso = 'ar_37'
else:
    iso = sys.argv[1]

of = open('pdfs/')
pdfs = 