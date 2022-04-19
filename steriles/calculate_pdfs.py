import sys
import numpy as np
import usphere_utils as uu

def plot_recon_mass_secondaries(Q, t12, A, secondaries, loading_frac=1e-2, livetime=10, num_spheres=1, sphere_radius = 50e-9, eta_xyz=[0.6,0.6,0.6], f0=1e5, Ue4_2 = 1e-4, msterile=750, trigger_prob=0.5, ang_error = 0.01, nbins=100):
    
    ## secondaries is a list of other correlated particles (augers, xrays, gammas, with probabilities)
    ## first column is the probability of that path
    ## second column is the kinetic energy 
    ## third column is the rest mass of the particle
    ## last row is always the probability to be undetected
    m_sph = 4/3*np.pi*sphere_radius**3 * uu.rho
    n_nuclei = m_sph * uu.N_A/A * loading_frac * num_spheres
    n_events = n_nuclei * (1 - 0.5**(livetime/t12) ) 
    nmc = int(n_events)

    nsecondaries = np.shape(secondaries)[0] #number of secondaries

    second_list = np.random.choice(nsecondaries, nmc, p=secondaries[:,0])

    ## keep only the points for which there was a trigger particle
    gpts = second_list < nsecondaries-1
    gpts = gpts & (np.random.rand(nmc)<trigger_prob) ## only events that the trigger is detected
    second_list = second_list[gpts]

    nmc_detect = np.sum(gpts) ## number of detected events

    ## first get the truth quantities ######################
    ## calculate the neutrino mass
    mnu = 50e-3*np.ones(nmc_detect)
    sterile_decays = np.random.rand(nmc_detect) < Ue4_2
    mnu[sterile_decays] = msterile

    ## random direction for the nu
    phi_nu = np.random.rand(nmc_detect)*2*np.pi
    theta_nu = np.arccos(2*np.random.rand(nmc_detect) - 1)

    ## random direction for the secondary
    phi_second = np.random.rand(nmc_detect)*2*np.pi
    theta_second = np.arccos(2*np.random.rand(nmc_detect) - 1)

    ## kinetic energy of the secondary
    T_sec = secondaries[second_list,1]
    m_sec = secondaries[second_list,2]
    p_sec = np.sqrt( (T_sec + m_sec)**2 - m_sec**2 ) ## momentum of the secondary

    ## neutrino momentum
    p_nu_true = np.sqrt((Q-T_sec)**2 - mnu**2)

    p_sph_x = -( p_nu_true*np.cos(phi_nu)*np.sin(theta_nu) + p_sec*np.cos(phi_second)*np.sin(theta_second) )
    p_sph_y = -( p_nu_true*np.sin(phi_nu)*np.sin(theta_nu) + p_sec*np.sin(phi_second)*np.sin(theta_second) )
    p_sph_z = -( p_nu_true*np.cos(theta_nu) + p_sec*np.cos(theta_second) )

    ### end of the truth quantitites ######################

    p_res = np.sqrt(uu.hbar * m_sph * 2*np.pi*f0)/uu.kg_m_per_s_to_keV

    ### now the reconstructed quantities (noise for each direction -- eventually update with detection effficiencies)
    p_sph_x_recon = p_sph_x + eta_xyz[0]**-0.25 * p_res*np.random.randn( nmc_detect )
    p_sph_y_recon = p_sph_y + eta_xyz[1]**-0.25 * p_res*np.random.randn( nmc_detect )
    p_sph_z_recon = p_sph_z + eta_xyz[2]**-0.25 * p_res*np.random.randn( nmc_detect )

    phi_second_recon = phi_second + ang_error*np.random.randn( nmc_detect )
    theta_second_recon = theta_second + ang_error*np.random.randn( nmc_detect )
    energy_second_recon = T_sec  ## assume we know the energy better than we can reconstruct it
    p_second_recon = np.sqrt( (energy_second_recon + m_sec)**2 - m_sec**2 )

    p_second_x_recon = p_second_recon*np.cos(phi_second_recon)*np.sin(theta_second_recon)
    p_second_y_recon = p_second_recon*np.sin(phi_second_recon)*np.sin(theta_second_recon)
    p_second_z_recon = p_second_recon*np.cos(theta_second_recon)

    p_nu_recon = np.sqrt( (p_sph_x_recon + p_second_x_recon)**2 + (p_sph_y_recon + p_second_y_recon)**2 + (p_sph_z_recon + p_second_z_recon)**2 )


    nbins1 = int(nbins)
    bins = np.linspace(-5*p_res, Q+5*p_res, nbins1)
    hh, be = np.histogram(p_nu_recon, bins=bins)
    bc = be[:-1] + np.diff(be)/2

    return bc, hh    

Q_Ar37 = 813
t12_Ar37 = 35
Ar37_seconds = [[0.81, 2.4, 511],
                [0.082, 2.6, 0],
                [0.005, 2.8, 0],
                [0.103, 0, 0]]
Ar37_seconds = np.array(Ar37_seconds)

b, h = plot_recon_mass_secondaries(Q_Ar37, t12_Ar37, 37, Ar37_seconds)
