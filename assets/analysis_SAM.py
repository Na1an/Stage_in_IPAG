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
    fwhm = vip.var.fit_2dgaussian(psf, crop=True, cropsize=17, debug=False)

    return np.mean([fwhm.loc[0,'fwhm_y'], fwhm.loc[0,'fwhm_x']])

# get flux
def get_flux(path, positions):
    '''
    Args:
        path : a string. The path of repository where the files are.
        positions : a list of tuple (x,y). The coordinates of companions.
    Return:
        res : a np.array, 1 dimension. Store the list of each companion's Signal to Noise ratio.
    '''
    files = os.listdir(path)
    files.sort()
    l = len(files)
    res = np.zeros(l)
    for i in range(len(res)):
        file = path + '/' + files[i]
        #print("file"

# get S/N ratio
def get_SN(path, positions, fwhm):
    '''
    Args:
        path : a string. The path of repository where the files are.
        positions : a list of tuple (x,y). The coordinates of companions.
    Return:
        flux : a np.array, 1 dimension. Store the list of each companion's flux.
        SN : a np.array, 1 dimension. Store the list of each companion's Signal to Noise ratio.
    '''
    files = os.listdir(path)
    files.sort()
    l = len(files)
    
    # flux
    flux = np.zeros(l)
    aperture = CircularAperture(positions, r=2)
    annulus = CircularAnnulus(positions, r_in=4, r_out=6)
 
    # SN
    SN = np.zeros(l)

    for i in range(l):
        file = path+'/'+files[i]
        print("file",i,"=", file)
        data = vip.fits.open_fits(file)
        
        # flux
        flux_companion = aperture_photometry(data, [aperture, annulus])
        flux_companion['aperture_sum_0','aperture_sum_1'].info.format = '%.8g'
        bkg_mean = flux_companion['aperture_sum_1']/annulus.area
        bkg_sum_in_companion = bkg_mean * aperture.area
        flux[i] = flux_companion['aperture_sum_0'] - bkg_sum_in_companion 

        # SN
        lets_plot = False
        if i==2:
            lets_plot = True
            #ds9.display(data)
        SN[i] = vip.metrics.snr(data, source_xy=positions[0], fwhm=fwhm, plot=lets_plot)

    return flux, SN
# process data
def process_data(positions, path):
    '''
    Args:
        position : a array of tuple (x,y). The position of companion.
        path : a string. The path of repository where the files are.
    Return:
        Many data.
    '''
    # companion 
    start_time = datetime.datetime.now()
    # psf
    psf = vip.fits.open_fits(str(sys.argv[1]))
    fwhm = get_fwhm_from_psf(psf[0])
    print("fwhm =", fwhm)

    # 4 is 4 pxls
    fwhm_for_snr = 4
    # 65.20625576328908 
    
    flux_3_best, sn_3_best = get_SN(path+"/3_best", positions, fwhm_for_snr)
    flux_5_best, sn_5_best = get_SN(path+"/5_best", positions, fwhm_for_snr)
    flux_7_best, sn_7_best = get_SN(path+"/7_best", positions, fwhm_for_snr)
    
    return flux_3_best, flux_5_best, flux_7_best, sn_3_best, sn_5_best, sn_7_best
    
if __name__ == "__main__":
    print("###### Start to process the data ######")
    
    # companion 
    start_time = datetime.datetime.now()
    positions = [(125.05284, 248.11)]
    positions_fake = [(142.06025, 165.01939)]
    # psf
    psf = vip.fits.open_fits(str(sys.argv[1]))
    fwhm = get_fwhm_from_psf(psf[0])
    print("fwhm =", fwhm)

    # 4 is 4 pxls
    fwhm_for_snr = 4
    # 34.1118 real 
    # 225.9158469659664 fake - injection with 40
    # 38.11965163928164
    '''
    # real
    flux_3_best_real, flux_5_best_real, flux_7_best_real, sn_3_best_real, sn_5_best_real, sn_7_best_real = process_data(positions, "./spat-mean/")
    flux_3_best_real_sam, flux_5_best_real_sam, flux_7_best_real_sam, sn_3_best_real_sam, sn_5_best_real_sam, sn_7_best_real_sam = process_data(positions, "./spat-annular/")
    '''

    # fake
    flux_3_best_fake, flux_5_best_fake, flux_7_best_fake, sn_3_best_fake, sn_5_best_fake, sn_7_best_fake = process_data(positions_fake, "./spat-mean/")
    flux_3_best_fake_sam, flux_5_best_fake_sam, flux_7_best_fake_sam, sn_3_best_fake_sam, sn_5_best_fake_sam, sn_7_best_fake_sam = process_data(positions_fake, "./spat-annular/")

    nb_data = 7
    l=len(flux_3_best_fake)
    data = np.zeros((l, nb_data))

    for i in range(l):
        data[i][0] = flux_3_best_fake[i]
        data[i][1] = flux_5_best_fake[i]
        data[i][2] = flux_7_best_fake[i]
        data[i][3] = flux_3_best_fake_sam[i]
        data[i][4] = flux_5_best_fake_sam[i]
        data[i][5] = flux_7_best_fake_sam[i]
        data[i][6] = 38.11965163928164 # origin
   
    data_SN = np.zeros((l,nb_data))
    
    for i in range(l):
        data_SN[i][0] = sn_3_best_fake[i]
        data_SN[i][1] = sn_5_best_fake[i]
        data_SN[i][2] = sn_7_best_fake[i]
        data_SN[i][3] = sn_3_best_fake_sam[i]
        data_SN[i][4] = sn_5_best_fake_sam[i]
        data_SN[i][5] = sn_7_best_fake_sam[i]

    data_total = pd.DataFrame(data[:,:], columns=["spat_mean 3 best", "spat_mean 5 best","spat_mean 7 best","SAM 3 best", "SAM 5 best","SAM 7 best", "origin_annulus_method"])
    data_total.index = data_total.index + 1
    print("######### Flux of companion #######")
    print(data_total)
    #data_total.to_csv("Flux_of_companion.csv")
    #data_total.plot(kind='line', style='--o', title='comparation')
    sns.set(font_scale = 1.5)
    sns.relplot(kind='line',data=data_total)
    plt.title("Flux : fake companion, with different ref cube, SAM vs spat-mean", fontsize = 18)
    plt.xlabel("K_kilp")
    plt.ylabel("Flux of the companion absolute - diameter 4 px")
    plt.ylim(0,70)
    plt.show()

    data_total_SN = pd.DataFrame(data_SN[:,:6], columns=["spat_mean 3 best", "spat_mean 5 best","spat_mean 7 best","SAM 3 best", "SAM 5 best","SAM 7 best"])
    data_total_SN.index = data_total_SN.index + 1
    print("######### S/N ########")
    print(data_total_SN)
    #data_total_SN.to_csv("SN_companions.csv")
    #data_total.plot(kind='line', style='--o', title='comparation')
    ax = sns.relplot(kind='line',data=data_total_SN)
    #plt.legend(fontsize = '16')
    plt.title("S/N ratio : fake companion, with different ref cube, SAM vs spat-mean", fontsize=18)
    plt.xlabel("K_kilp", fontsize= "16")
    plt.ylabel("S/N - diameter 4 px", fontsize = "16")
    plt.ylim(0,20)
    plt.show()
    end_time = datetime.datetime.now()
    print("cost :", end_time - start_time)
    print("###### End of program ######")
