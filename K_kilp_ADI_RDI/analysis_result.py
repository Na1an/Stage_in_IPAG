import os
import sys
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import vip_hci as vip
import matplotlib.pyplot as plt

from astropy.io import fits
from hciplot import plot_frames, plot_cubes
from astropy.visualization import simple_norm
from photutils.aperture import CircularAperture, aperture_photometry, CircularAnnulus

# if show the aperture and the annulus of companion's position
SHOW_POSITION = False

# get one fwhm of a frame
def get_fwhm_from_psf(psf):
    '''
    Args:
        psf : a 2D np.ndarray. The PSF in one wavelength.
    Return:
        res : a float. The one fwhm of the psf.
    '''
    fwhm = vip.var.fit_2dgaussian(psf, crop=True, cropsize=15, debug=False)

    return np.mean([fwhm.loc[0,'fwhm_y'], fwhm.loc[0,'fwhm_x']])

# get S/N ratio
def get_SN(path, positions, fwhm):
    '''
    Args:
        path : a string. The path of repository where the files are.
    Rrturn:
        res : a np.array, 1 dimension. Store the list of each companion's Signal to Noise ratio.
    '''
    files = os.listdir(path)
    files.sort()
    l = len(files)
    res = np.zeros(l)
    for i in range(l):
        file = path+'/'+files[i]
        print("file",i,"=", file)
        data = vip.fits.open_fits(file)
        lets_plot = False
        if i==2:
            lets_plot = True
        res[i] = vip.metrics.snr(data, source_xy=positions[0], fwhm=fwhm, plot=lets_plot)
    
    return res

if __name__ == "__main__":
    print("###### Start to process the data ######")
    start_time = datetime.datetime.now()
    positions = [(125.05284, 248.11)]
    aperture = CircularAperture(positions, r=2)

    psf = vip.fits.open_fits(str(sys.argv[1]))
    fwhm = get_fwhm_from_psf(psf[0])
    
    best_3 = get_SN("./RDI_res_3", positions, fwhm)
    best_4 = get_SN("./RDI_res_4", positions, fwhm)
    best_5 = get_SN("./RDI_res_5", positions, fwhm)    

    nb_data = 3
    data = np.zeros((200, nb_data))

    for i in range(200):
        if(i>=len(best_3)):
            data[i][0] = best_3[-1]
        else:
            data[i][0] = best_3[i]
        
        if(i>=len(best_4)):
            data[i][1] = best_4[-1]
        else:
            data[i][1] = best_4[i]
        
        if(i>=len(best_5)):
            data[i][2] = best_5[-1]
        else:
            data[i][2] = best_5[i]

    data_total_SN = pd.DataFrame(data[:,:], columns=['RDI_3_best','RDI_4_best','RDI_5_best'])
    #data_total_SN = pd.DataFrame(data_SN[:,:], columns=['S/N_ADI_Outer_Region_3_best','S/N_RDI_Outer_Region_3_best'])
    data_total_SN.index = data_total_SN.index + 1
    print("######### S/N ########")
    print(data_total_SN)
    #data_total_SN.to_csv("SN_companions.csv")
    #data_total.plot(kind='line', style='--o', title='comparation')
    ax = sns.relplot(kind='line',data=data_total_SN)
    plt.legend(fontsize = '16')
    #plt.title("Target:GJ667c Ref: 9 other stars")
    plt.xlabel("K_kilp", fontsize= "16")
    plt.ylabel("S/N - diameter 4 px", fontsize = "16")
    plt.show()
    end_time = datetime.datetime.now()
    print("cost :", end_time - start_time)
    print("###### End of program ######")


