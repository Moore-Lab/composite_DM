import pickle, sys, glob
import numpy as np
import matplotlib.pyplot as plt
import usphere_utils as uu

## using pdfs calculated from calculate_pdfs.py, calculate the sensitivity
## for a given isotope as a function of mass

def calc_limit(t12, A, loading_frac, num_spheres, livetime, bkg_pdf, sig_pdf, m, Q, isEC = True, trig_prob = 0.5, eta_xyz=[0.6,0.6,0.6], f0=1e5, ang_error = 0.01, nbins=100, sphere_rad=50e-9):

    do_asimov = True ## use asimov dataset for sensitivity

    m_sph = 4/3 * np.pi * sphere_rad**3 * uu.rho
    n_nuclei = m_sph * uu.N_A/A * loading_frac * num_spheres

    if(livetime < t12):
      n_decays = int(trig_prob * n_nuclei * (1 - 0.5**(livetime/t12) ))
    else:
      ## assume the sphere is reloaded once per half-life
      niters = np.floor(livetime/t12)
      livetime_remain = livetime % t12
      #print("Working on %d halflives and %f days remaining"%(niters, livetime_remain))
      n_decays = niters * 0.5*n_nuclei + int(trig_prob * n_nuclei * (1 - 0.5**(livetime_remain/t12) ))

    if(isEC):
      ## require bkg and sig pdf to have same x values (by construction in calculate pdfs)
      bkg_pdf_p = bkg_pdf[:,1]
      sig_pdf_p = sig_pdf[:,1]

      if(do_asimov):
        hh = uu.draw_asimov_from_pdf( n_decays, bkg_pdf_p)

      else:
        bkg_only_cts = uu.draw_from_pdf( n_decays, bkg_pdf_x, bkg_pdf_p )
        bkg_pdf_x = bkg_pdf[:,0] 
        p_res = np.sqrt(uu.hbar * m_sph * 2*np.pi*f0)/uu.kg_m_per_s_to_keV
        hh, be = np.histogram(bkg_only_cts, bins = np.arange(bkg_pdf_x[0], bkg_pdf_x[-1], p_res/4))
        bc = be[:-1] + np.diff(be)/2

    else:
      ## for a beta we need to handle the 2D PDFs
      ## only allow asimov for now
      bkg_pdf_p = bkg_pdf[:,2:]
      sig_pdf_p = sig_pdf[:,2:]

      hh = uu.draw_asimov_from_pdf( n_decays,  bkg_pdf_p)

    #plt.figure()
    #plt.pcolormesh(hh)
    #plt.savefig('dummy.png')
    #plt.close()
    #input('plotted')

    ue4, prof = uu.profile_sig_counts(hh, bkg_pdf_p, sig_pdf_p)

    ## find the min of the profile
    midx = np.argmin(prof)
    ue4 = ue4[midx:]
    prof = prof[midx:]


    if(len(prof) == 0): ## min not found early enough
      ulim = np.nan
    else:
      ulim = np.interp(uu.conf_lev, prof, ue4, left=np.nan, right=np.nan)
        
    return ulim


if(len(sys.argv)==1):
  iso_list = ['p_32',] #,'s_35','y_90','be_7','ar_37', 'v_49','cr_51',"fe_55", 'ge_68', 'se_72']
  params_list = [[0.2, 10000, 365*5],]
else:
  iso_list = [sys.argv[1],]
  l = float(sys.argv[2])
  n = int(sys.argv[3])
  d = int(sys.argv[4])
  params_list = [[l,n,d],]

## list of parameters to use (loading frac, num spheres, livetime, radius, f0)

## 20% loading by mass for h_3
# params_list = [[0.2, 1, 10],
#                [0.2, 1, 30],
#                [0.2, 1, 100],
#                [0.2, 10, 10],
#                [0.2, 10, 365], 
#                [0.2, 1000, 365]]

for iso in iso_list:

  isEC = not iso in uu.beta_list

  for p in params_list:

    loading_frac, num_spheres, livetime = p

    flist = glob.glob('pdfs/%s*_pdfs.pkl'%iso)

    for file in flist:

      fparts = file.split('_')
      rad, f0 = float(fparts[2]), float(fparts[3])

      of = open(file, 'rb')
      pdfs = pickle.load(of)
      of.close()

      iso_dat = np.loadtxt("/home/dcm42/impulse/steriles/data_files/%s.txt"%iso, delimiter=',', skiprows=3)
      Q, t12, A, Z = iso_dat[0, :]

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

        #if( m < 1210): continue

        sig_pdf = pdfs['%.1f'%m]
        if(np.max(np.abs(bkg_pdf[:,0]-sig_pdf[:,0]))>1e-10):
          print("mismatched x vectors")

        curr_params = uu.params_dict
        curr_params['f0'] = f0
        curr_params['sphere_rad'] = rad/uu.m_to_nm 

        ulim[i] = calc_limit(t12, A, loading_frac, num_spheres, livetime, bkg_pdf, sig_pdf, m, Q, isEC=isEC, **curr_params)
        print(m, ulim[i])

      params = [loading_frac, num_spheres, livetime, rad, f0]
      np.savez("/home/dcm42/impulse/steriles/limits/%s_limit_%.1e_%d_%.1f_%.1f_%.1e.npz"%(iso,loading_frac,num_spheres,livetime,rad,f0), m=mass_list, lim=ulim, params=params)