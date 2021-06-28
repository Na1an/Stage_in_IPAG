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
def get_one_contrast_and_SN(data, positions, fwhm, fwhm_flux):
    '''
    Args:
        path : a string. The path of repository where the files are.
        positions : a list of tuple (x,y). The coordinates of companions.
    Return:
        flux : a np.array, 1 dimension. Store the list of each companion's flux.
        SN : a np.array, 1 dimension. Store the list of each companion's Signal to Noise ratio.
    '''
    
    # contast
    aperture = CircularAperture(positions, r=2)
    annulus = CircularAnnulus(positions, r_in=4, r_out=6)
 
    # contrast
    flux_companion = aperture_photometry(data, [aperture, annulus])
    flux_companion['aperture_sum_0','aperture_sum_1'].info.format = '%.8g'
    contrast = (flux_companion['aperture_sum_0']/aperture.area)/fwhm_flux

    # SN
    ds9 = vip.Ds9Window()
    ds9.display(data)
    SN = vip.metrics.snr(data, source_xy=positions[0], fwhm=fwhm, plot=True)

    return contrast[0], SN

# get contrast and S/N ratio
def get_contrast_and_SN(path, positions, fwhm, fwhm_flux):
    '''
    Args:
        path : a string. The path of repository where the files are.
        positions : a list of tuple (x,y). The coordinates of companions.
    Return:
        contrast : a np.array, 1 dimension. Store the list of each companion's flux.
        SN : a np.array, 1 dimension. Store the list of each companion's Signal to Noise ratio.
    '''
    files = os.listdir(path)
    files.sort()
    l = len(files)
    
    # flux
    contrast = np.zeros(l)
    aperture = CircularAperture(positions, r=2)
    annulus = CircularAnnulus(positions, r_in=4, r_out=6)
 
    # SN
    SN = np.zeros(l)

    for i in range(l):
        file = path+'/'+files[i]
        print("file",i,"=", file)
        data = vip.fits.open_fits(file)
        
        # contrast
        flux_companion = aperture_photometry(data, [aperture, annulus])
        flux_companion['aperture_sum_0','aperture_sum_1'].info.format = '%.8g'
        #bkg_mean = flux_companion['aperture_sum_1']/annulus.area
        #bkg_sum_in_companion = bkg_mean * aperture.area
        #flux[i] = flux_companion['aperture_sum_0'] - bkg_sum_in_companion 
        contrast[i] = (flux_companion['aperture_sum_0']/aperture.area)/fwhm_flux

        # SN
        lets_plot = False
        if i==2:
            lets_plot = True
            #ds9.display(data)
        SN[i] = vip.metrics.snr(data, source_xy=positions, fwhm=fwhm, plot=lets_plot)

    return contrast, SN


if __name__ == "__main__":
    print("###### Start to process the data ######")
    
    # companion
    start_time = datetime.datetime.now()
    # real companion
    #positions = [(619, 455.875)]
    # fake comp close, 20pxs
    #positions = [(526, 524)]
    # fake comp far, 100pxs
    #positions = [(586, 576.75)]
    # fake disk close, 20 pxs
    #positions = [(501.4, 493)]
    # fake disk far, 100pxs
    #positions = [(464.5, 423.5)]
    positions = [(125.04514,248.31597), (109.25, 134.875), (33.9375, 162.03125), (102.8125, 127.125), (16.681285, 125.31547)]

    wl = 0
    # psf
    psf = vip.fits.open_fits(str(sys.argv[1]))
    fwhm = get_fwhm_from_psf(psf[0])
    psfn, fwhm_flux, fwhm_bis = vip.metrics.normalize_psf(psf[wl], fwhm, size=17, full_output=True)
    print("fwhm =", fwhm, "fwhm_bis =", fwhm_bis, "fwhm_flux[0] =", fwhm_flux[0])
    fwhm_for_snr=4
    
    # real in all algo with all refs numbers
    
    # ADI
    #adi_real_contrast, adi_real_sn = get_contrast_and_SN("./ADI/real", positions[0], fwhm_for_snr, fwhm_flux[0])
    
    # spat_mean
    sm_real_3_best_contrast, sm_real_3_best_sn = get_contrast_and_SN("./spat_mean/real/3_best", positions[0], fwhm_for_snr, fwhm_flux[0])
    sm_real_5_best_contrast, sm_real_5_best_sn = get_contrast_and_SN("./spat_mean/real/5_best", positions[0], fwhm_for_snr, fwhm_flux[0])
    sm_real_7_best_contrast, sm_real_7_best_sn = get_contrast_and_SN("./spat_mean/real/7_best", positions[0], fwhm_for_snr, fwhm_flux[0])
    sm_real_9_best_contrast, sm_real_9_best_sn = get_contrast_and_SN("./spat_mean/real/9_best", positions[0], fwhm_for_snr, fwhm_flux[0])
    sm_real_11_best_contrast, sm_real_11_best_sn = get_contrast_and_SN("./spat_mean/real/11_best", positions[0], fwhm_for_snr, fwhm_flux[0])
    
    # spat_annular_mean
    sam_real_3_best_contrast, sam_real_3_best_sn = get_contrast_and_SN("./spat_annular_mean/real/3_best", positions[0], fwhm_for_snr, fwhm_flux[0])
    sam_real_5_best_contrast, sam_real_5_best_sn = get_contrast_and_SN("./spat_annular_mean/real/5_best", positions[0], fwhm_for_snr, fwhm_flux[0])
    sam_real_7_best_contrast, sam_real_7_best_sn = get_contrast_and_SN("./spat_annular_mean/real/7_best", positions[0], fwhm_for_snr, fwhm_flux[0])
    sam_real_9_best_contrast, sam_real_9_best_sn = get_contrast_and_SN("./spat_annular_mean/real/9_best", positions[0], fwhm_for_snr, fwhm_flux[0])
    sam_real_11_best_contrast, sam_real_11_best_sn = get_contrast_and_SN("./spat_annular_mean/real/11_best", positions[0], fwhm_for_snr, fwhm_flux[0])

    nb_data = 11
    l = len(sam_real_3_best_sn)
    
    # contrast
    data = np.zeros((l, nb_data))
    print("data.shape =", data.shape, "adi_real_contrast.shape", sm_real_3_best_contrast.shape)
    for i in range(l):
        #data[i, 0] = adi_real_contrast 
        data[i, 1] = sm_real_3_best_contrast[i] 
        data[i, 2] = sm_real_5_best_contrast[i]
        data[i, 3] = sm_real_7_best_contrast[i]
        data[i, 4] = sm_real_9_best_contrast[i]
        data[i, 5] = sm_real_11_best_contrast[i]
        data[i, 6] = sam_real_3_best_contrast[i] 
        data[i, 7] = sam_real_5_best_contrast[i]
        data[i, 8] = sam_real_7_best_contrast[i] 
        data[i, 9] = sam_real_9_best_contrast[i]
        data[i, 10] = sam_real_11_best_contrast[i]  

    # snr
    data_sn = np.zeros((l, nb_data))
    for i in range(l):
        #data_sn[i, 0] = adi_real_sn 
        data_sn[i, 1] = sm_real_3_best_sn[i] 
        data_sn[i, 2] = sm_real_5_best_sn[i]
        data_sn[i, 3] = sm_real_7_best_sn[i] 
        data_sn[i, 4] = sm_real_9_best_sn[i]
        data_sn[i, 5] = sm_real_11_best_sn[i] 
        data_sn[i, 6] = sam_real_3_best_sn[i] 
        data_sn[i, 7] = sam_real_5_best_sn[i]
        data_sn[i, 8] = sam_real_7_best_sn[i] 
        data_sn[i, 9] = sam_real_9_best_sn[i]
        data_sn[i, 10] = sam_real_11_best_sn[i]  

    ##################
    # plot te result #
    ##################
    
    data_total = pd.DataFrame(data[:,1:], columns=["spat_mean_real_3_best", "spat_mean_real_5_best", "spat_mean_real_7_best","spat_mean_real_9_best",  "spat_mean_real_11_best", "spat_annular_mean_real_3_best", "spat_annular_mean_real_5_best", "spat_annular_mean_real_7_best","spat_annular_mean_real_9_best",  "spat_annular_mean_real_11_best"])
    data_total.index = data_total.index *5
    print("######### Contrast of companion #######")
    print(data_total)
    #data_total.to_csv("Flux_of_companion.csv")
    #data_total.plot(kind='line', style='--o', title='comparation')
    sns.set(font_scale = 1.5)
    sns.relplot(kind='line',data=data_total)
    plt.title("real - RDI spat-mean vs RDI spat-annular-mean : contrast", fontsize = 20)
    plt.xlabel("K_kilp", fontsize = "18")
    plt.ylabel("Contrast - diameter 4 px", fontsize = "18")
    #plt.ylim(0,70)
    plt.show()
    
    data_total_SN = pd.DataFrame(data_sn[:,1:], columns=["spat_mean_real_3_best", "spat_mean_real_5_best", "spat_mean_real_7_best","spat_mean_real_9_best",  "spat_mean_real_11_best", "spat_annular_mean_real_3_best", "spat_annular_mean_real_5_best", "spat_annular_mean_real_7_best","spat_annular_mean_real_9_best",  "spat_annular_mean_real_11_best"])
    data_total_SN.index = data_total_SN.index *5
    print("######### S/N ########")
    print(data_total_SN)
    #data_total_SN.to_csv("SN_companions.csv")
    #data_total.plot(kind='line', style='--o', title='comparation')
    ax = sns.relplot(kind='line',data=data_total_SN)
    #plt.legend(fontsize = '16')
    plt.title("real - RDI spat-mean vs RDI spat-annular-mean : signal to noise ratio", fontsize=20)
    plt.xlabel("K_kilp", fontsize= "18")
    plt.ylabel("S/N ratio - diameter 4 px", fontsize = "18")
    #plt.ylim(0,20)
    plt.show()
    end_time = datetime.datetime.now()
    print("cost :", end_time - start_time)
    print("###### End of program ######")
