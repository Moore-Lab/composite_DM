import pickle
import numpy as np
import matplotlib.pyplot as plt
import usphere_utils as uu

## using pdfs calculated from calculate_pdfs.py, calculate the sensitivity
## for a given isotope as a function of mass

def calc_limit(t12, A, loading_frac, num_spheres, livetime, bkg_pdf, sig_pdf, trig_prob = 0.5, eta_xyz=[0.6,0.6,0.6], f0=1e5, ang_error = 0.01, nbins=100):

    do_asimov = True ## use asimov dataset for sensitivity

    m_sph = 4/3 * np.pi * uu.sphere_radius**3 * uu.rho
    n_nuclei = m_sph * uu.N_A/A * loading_frac * num_spheres
    n_decays = int(trig_prob * n_nuclei * (1 - 0.5**(livetime/t12) ))

    bkg_pdf_x = bkg_pdf[:,0] ## require bkg and sig pdf to have same x values (by construction in calculate pdfs)

    if(do_asimov):
      bc, hh = uu.draw_asimov_from_pdf( n_decays, bkg_pdf_x, bkg_pdf[:,1] )

    else:
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
        
    return ulim



iso_list = ['v_49','cr_51',"fe_55", 'ge_68', 'se_72']
## list of parameters to use (loading frac, num spheres, livetime)
params_list = [[1e-2, 1, 10], 
               [1e-2, 1000, 365], ]

for iso in iso_list:

  for p in params_list:

    loading_frac, num_spheres, livetime = p

    of = open('pdfs/%s_pdfs.pkl'%iso, 'rb')
    pdfs = pickle.load(of)
    of.close()

    iso_dat = np.loadtxt("/home/dcm42/impulse/steriles/data_files/%s.txt"%iso, delimiter=',', skiprows=3)
    Q, t12, A = iso_dat[0, :]

    mass_list_str = pdfs.keys()
    mass_list = []
    for m in mass_list_str:
      cmass = float(m)
      if(cmass > 0):
        mass_list.append(cmass)

    mass_list = sorted(mass_list)

    ulim = np.ones_like(mass_list)*1e6
    bkg_pdf = pdfs['0.0']
    for i,m in enumerate(mass_list):

      #if( livetime < 300 or m < 340): continue

      sig_pdf = pdfs['%.1f'%m]
      if(np.max(np.abs(bkg_pdf[:,0]-sig_pdf[:,0]))>1e-10):
        print("mismatched x vectors")
      
      ulim[i] = calc_limit(t12, A, loading_frac, num_spheres, livetime, bkg_pdf, sig_pdf, **uu.params_dict)
      print(m, ulim[i])

    params = [loading_frac, num_spheres, livetime]
    np.savez("/home/dcm42/impulse/steriles/limits/%s_limit_%.1e_%d_%.1f.npz"%(iso,loading_frac,num_spheres,livetime), m=mass_list, lim=ulim, params=params)