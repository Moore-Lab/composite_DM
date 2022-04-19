import numpy as np

## various config parameters for impulse calculation

m_light = 50e-3 # light neutrino mass [eV]
days_to_s = 24*3600
sphere_radius = 50e-9 # m
rho = 2e3 # kg/m^3
N_A = 6.02e23 * 1e3 # amu per kg
sphere_mass_amu = 4/3*np.pi*sphere_radius**3 * rho * N_A
hbar = 1.05e-34 ## SI units
kg_m_per_s_to_keV = 5.34e-25

params_dict = { 'eta_xyz': [0.6,0.6,0.6], ## detection efficiency in each coord
                'f0': 1e5, ## trap resonant frequency
                'ang_error': 0.01, ## angular error for secondary detection [rad], 
                'nbins': 250, ## number of bins for PDF
                }