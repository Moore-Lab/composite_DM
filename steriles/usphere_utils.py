import numpy as np
from scipy.optimize import minimize

## various config parameters for impulse calculation

m_light = 50e-3 # light neutrino mass [eV]
days_to_s = 24*3600
sphere_radius = 50e-9 # m
rho = 2e3 # kg/m^3
N_A = 6.02e23 * 1e3 # amu per kg
sphere_mass_amu = 4/3*np.pi*sphere_radius**3 * rho * N_A
hbar = 1.05e-34 ## SI units
kg_m_per_s_to_keV = 5.34e-25
conf_lev = 3.84/2 ## ln(L) for 95% confidence
me = 511 #electron mass in keV
e_res = 10 # keV, energy resolution of beta detection

beta_list = ['p_32',]

params_dict = { 'eta_xyz': [0.6,0.6,0.6], ## detection efficiency in each coord
                'f0': 1e5, ## trap resonant frequency
                'ang_error': 0.01, ## angular error for secondary detection [rad], 
                'nbins': 250, ## number of bins for PDF
                }


def draw_from_pdf(n, pdf_x, pdf_p):
  ## function to draw n values from a PDF
  cdf = np.cumsum(pdf_p)
  cdf /= np.max(cdf)
  rv = np.random.rand(n)
  return np.interp(rv, cdf, pdf_x)

def draw_asimov_from_pdf(n, pdf_x, pdf_p):
  ## function to draw n values from a PDF
  n_counts = np.round( n * pdf_p/np.sum(pdf_p) )
  return pdf_x, n_counts

def fit_fun(N, sig, pdf_sig, pdf_bkg, data):
  model = N*(sig*pdf_sig + pdf_bkg)/(1+sig)
  gpts = model > 0
  return np.sum( model[gpts] - data[gpts]*np.log(model[gpts]) ) 

def simple_beta(E, Q, ms):
  #return a simple spectrum of counts vs electron KE (to be updated eventually)
  ## assumes E in keV
  ## Q is Q value in keV
  ## ms is nu mass in keV
  N = np.sqrt(E**2 + 2*E*me)*(E + me)*np.sqrt((Q-E)**2 - ms**2)*(Q-E)
  N[E > Q-ms] = 0
  return N

def profile_sig_counts(toy_data_x, toy_data_cts, pdf_x, pdf_bkg, pdf_sig):
  ## function to profile over values of the signal counts

  pb = np.interp(toy_data_x, pdf_x, pdf_bkg/np.sum(pdf_bkg))
  ps = np.interp(toy_data_x, pdf_x, pdf_sig/np.sum(pdf_sig))

  ## make a guess for the upper limit of the profile
  xr = np.where( ps>0.05*np.max(ps) )[0]
  tcts = np.sum(toy_data_cts[xr])
  if(tcts == 0): tcts = 10
  ue4_max = 100*np.sqrt(tcts)/np.sum(toy_data_cts)

  sig_range = np.linspace(0, ue4_max, 500)
  profile = np.zeros_like(sig_range)
  best_nll = 1e20
  for i,sig in enumerate(sig_range):
      fit = minimize(fit_fun, np.sum(toy_data_cts), args=(sig, ps, pb, toy_data_cts), method='Nelder-Mead')
      profile[i] = fit.fun

      if(profile[i] < best_nll):
        best_nll = profile[i]
      
      if(profile[i] > best_nll + 500):
        break

  ## if we didn't get enough points, then retry with more points
  if(i < 4):
    sig_range = np.linspace(0, ue4_max, 5000)
    profile = np.zeros_like(sig_range)
    best_nll = 1e20
    for i,sig in enumerate(sig_range):
      fit = minimize(fit_fun, np.sum(toy_data_cts), args=(sig, ps, pb, toy_data_cts), method='Nelder-Mead')
      profile[i] = fit.fun

      if(profile[i] < best_nll):
        best_nll = profile[i]
      
      if(profile[i] > best_nll + 500):
        break

  sig_range = sig_range[:i]
  profile = profile[:i]
  profile -= np.min(profile)
  
  return sig_range, profile