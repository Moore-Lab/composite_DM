import numpy as np
import pickle, glob, os
import usphere_utils as uu

## collect and plot files from a given directory
data_dir = "/home/dcm42/impulse/steriles/data_files/"
save_dir = "/home/dcm42/impulse/steriles/pdfs/"
save_files = True

file_list = glob.glob(data_dir + "*.npz")

iso_list = []
## assumes file name starts with isotope in the form "el_A"
for f in file_list:
    cfile = os.path.split(f)[-1]
    fp = cfile.split('_')
    ciso = fp[0] + "_" + fp[1]
    if not ciso in iso_list:
        iso_list.append(ciso)

for iso in iso_list:

    if(iso != 'y_90'): continue

    curr_dict = {} ## dictionary to hold pdfs

    iso_files = glob.glob(data_dir + iso + "*.npz")

    ## make a list of the masses for that iso
    mnu_list = []
    for f in iso_files:
        cfile = os.path.split(f)[-1]
        fp = cfile.split('_')
        cmnu = fp[3]
        if not cmnu in mnu_list:
            mnu_list.append(cmnu)

    for mnu in mnu_list:

        curr_iso_files = glob.glob(data_dir + iso + "_mnu_" + mnu + "*.npz")
        nfiles = len(curr_iso_files)
        print("working on %d files for %s with mnu = %s: "%(nfiles, iso, mnu))

        for i in range(nfiles):
            print(i)
            pdf = np.load(curr_iso_files[i])
            
            if(i==0):
                p = pdf['pdf']
            else:
                p += pdf['pdf']

        x = pdf['x']

        if(not iso in uu.beta_list):
            cdf = np.cumsum(p)/np.sum(p)
            curr_dict[mnu] = np.vstack((x,p,cdf)).T
        else:
            ## first column is x
            ## second column is y
            ## remaining columns are data
            curr_dict[mnu] = np.hstack((x,p)) 


    if(save_files):
        of = open(save_dir + "%s_pdfs.pkl"%iso, "wb")
        pickle.dump(curr_dict, of)
        of.close()