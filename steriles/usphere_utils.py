import numpy as np

m_light = 50e-3 # light neutrino mass [eV]
days_to_s = 24*3600
t12_Be7 = 53.3 * days_to_s
sphere_radius = 50e-9 # m
rho = 2e3 # kg/m^3
N_A = 6.02e23 * 1e3 # amu per kg
sphere_mass_amu = 4/3*np.pi*sphere_radius**3 * rho * N_A
hbar = 1.05e-34 ## SI units
kg_m_per_s_to_keV = 5.34e-25
