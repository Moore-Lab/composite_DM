import sys
import numpy as np
import usphere_utils as uu

def plot_recon_mass_secondaries(Q, t12, A, secondaries, mnu, n_events=1e6, eta_xyz=[0.6,0.6,0.6], f0=1e5, ang_error = 0.01, nbins=100, isEC=True):
    
    ## secondaries is a list of other correlated particles (augers, xrays, gammas, with probabilities)
    ## first column is the probability of that path
    ## second column is the kinetic energy 
    ## third column is the rest mass of the particle
    ## last row is always the probability to be undetected

    nmc = int(n_events)

    nsecondaries = np.shape(secondaries)[0] #number of secondaries

    second_list = np.random.choice(nsecondaries, nmc, p=secondaries[:,0])

    ## keep only the points for which there was a trigger particle
    gpts = second_list < nsecondaries-1
    second_list = second_list[gpts]

    nmc_detect = np.sum(gpts) ## number of detected events

    ## first get the truth quantities ######################

    ## random direction for the nu
    phi_nu = np.random.rand(nmc_detect)*2*np.pi
    theta_nu = np.arccos(2*np.random.rand(nmc_detect) - 1)

    ## random direction for the secondary
    phi_second = np.random.rand(nmc_detect)*2*np.pi
    theta_second = np.arccos(2*np.random.rand(nmc_detect) - 1)

    ## kinetic energy of the secondary
    if(isEC):
        T_sec = secondaries[second_list,1]
    else:
        ## beta spectrum
        elec_e_vals = np.linspace(0, Q, int(1e3)) # electron kinetic energies to evaluate beta spectrum at
        beta_spec_e = uu.simple_beta(elec_e_vals, Q, mnu)
        T_sec = uu.draw_from_pdf(nmc_detect, elec_e_vals, beta_spec_e)
    
    m_sec = secondaries[second_list,2]
    p_sec = np.sqrt( (T_sec + m_sec)**2 - m_sec**2 ) ## momentum of the secondary

    ## neutrino momentum
    p_nu_true = np.sqrt((Q-T_sec)**2 - mnu**2)

    p_sph_x = -( p_nu_true*np.cos(phi_nu)*np.sin(theta_nu) + p_sec*np.cos(phi_second)*np.sin(theta_second) )
    p_sph_y = -( p_nu_true*np.sin(phi_nu)*np.sin(theta_nu) + p_sec*np.sin(phi_second)*np.sin(theta_second) )
    p_sph_z = -( p_nu_true*np.cos(theta_nu) + p_sec*np.cos(theta_second) )

    ### end of the truth quantitites ######################

    m_sph = 4/3*np.pi*uu.sphere_radius**3 * uu.rho
    p_res = np.sqrt(uu.hbar * m_sph * 2*np.pi*f0)/uu.kg_m_per_s_to_keV

    ### now the reconstructed quantities (noise for each direction -- eventually update with detection effficiencies)
    p_sph_x_recon = p_sph_x + eta_xyz[0]**-0.25 * p_res*np.random.randn( nmc_detect )
    p_sph_y_recon = p_sph_y + eta_xyz[1]**-0.25 * p_res*np.random.randn( nmc_detect )
    p_sph_z_recon = p_sph_z + eta_xyz[2]**-0.25 * p_res*np.random.randn( nmc_detect )

    phi_second_recon = phi_second + ang_error*np.random.randn( nmc_detect )
    theta_second_recon = theta_second + ang_error*np.random.randn( nmc_detect )
    
    energy_second_recon = T_sec  ## assume we know the energy better than we can reconstruct it

    ## if it's a beta, need to include error on kinetic energy
    if(not isEC):
        energy_second_recon += uu.e_res*np.random.randn( nmc_detect )
        energy_second_recon[energy_second_recon < 0 ] = 0 ## throw out unphysical smearings below zero

    p_second_recon = np.sqrt( (energy_second_recon + m_sec)**2 - m_sec**2 )

    p_second_x_recon = p_second_recon*np.cos(phi_second_recon)*np.sin(theta_second_recon)
    p_second_y_recon = p_second_recon*np.sin(phi_second_recon)*np.sin(theta_second_recon)
    p_second_z_recon = p_second_recon*np.cos(theta_second_recon)

    p_nu_recon = np.sqrt( (p_sph_x_recon + p_second_x_recon)**2 + (p_sph_y_recon + p_second_y_recon)**2 + (p_sph_z_recon + p_second_z_recon)**2 )

    if(isEC):
        ## 1D histo for ECs
        nbins1 = int(nbins)
        bins = np.linspace(-10*p_res, Q+10*p_res, nbins1)
        hh, be = np.histogram(p_nu_recon, bins=bins)
        bc = be[:-1] + np.diff(be)/2
    else:
        ## 2D histo for betas
        nbins1 = int(nbins)
        bins_x = np.linspace(-10*uu.e_res, Q+10*uu.e_res, nbins1)
        bins_y = np.linspace(-10*p_res, Q+10*p_res, nbins1)
        hh, bex, bey = np.histogram2d(energy_second_recon, p_nu_recon, bins=[bins_x, bins_y])
        bcx = bex[:-1] + np.diff(bex)/2
        bcy = bey[:-1] + np.diff(bey)/2
        bc = np.vstack((bcx, bcy)).T
    
    return bc, hh    

if(len(sys.argv)==1):
    iso = 'p_32'
    num_reps = 1
    idx = 0
    mnu_list = "1000"
else:
    iso = sys.argv[1]
    mnu_list = sys.argv[2]
    num_reps = int(sys.argv[3])
    idx = int(sys.argv[4])

mnu_list = mnu_list.split(",")
print(mnu_list)

isEC = True
if(iso in uu.beta_list):
    isEC = False
    print("Assuming %s is a beta"%iso)

iso_dat = np.loadtxt("/home/dcm42/impulse/steriles/data_files/%s.txt"%iso, delimiter=',', skiprows=3)

Q, t12, A = iso_dat[0, :]

seconds = iso_dat[1:,:]
tot_prob = np.sum(seconds[:,0])

seconds = np.vstack( (seconds, [1-tot_prob, 0, 0]) ) ## add any missing prob as last row

for cmnu in mnu_list:
    mnu = float(cmnu)
    h_tot = 0
    for i in range(num_reps):
        print("working on iteration %d for mnu %f"%(i,mnu))
        b, h = plot_recon_mass_secondaries(Q, t12, A, seconds, mnu, n_events=1e7, isEC=isEC, **uu.params_dict)

        if(i==0):
            h_tot = 1.0*h
        else:   
            h_tot += h

    c = np.cumsum(h_tot)/np.sum(h_tot)

    np.savez("/home/dcm42/impulse/steriles/data_files/%s_mnu_%.1f_pdf_%d.npz"%(iso, mnu, idx), x=b, pdf=h_tot, cdf=c, mnu=mnu)
