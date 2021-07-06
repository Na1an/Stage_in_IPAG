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

    return flux_companion['aperture_sum_0'], contrast[0], SN

# get contrast and S/N ratio
def get_contrast_and_SN(path, positions, fwhm, fwhm_flux, path_real):
    '''
    Args:
        path : a string. The path of repository where the files are.
        positions : a list of tuple (x,y). The coordinates of companions.
        fwhm : a float. fwhm's diameter.
        fwhm_flux : a float. The flux of fwhm.
        path_real : a string. The path of another repository where the files are, for calculating snr.
    Return:
        contrast : a np.array, 1 dimension. Store the list of each companion's flux.
        SN : a np.array, 1 dimension. Store the list of each companion's Signal to Noise ratio.
    '''
    files = os.listdir(path)
    files.sort()
    
    files_real = os.listdir(path_real)
    files_real.sort()
    l = len(files)
    

    flux = np.zeros(l)
    # contrast
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
        flux[i] = flux_companion['aperture_sum_0'] 
        contrast[i] = (flux_companion['aperture_sum_0']/aperture.area)/fwhm_flux

        # SN
        lets_plot = False
        if i==2:
            lets_plot = True
            #ds9.display(data)
        file_real = path_real+'/'+files_real[i]
        print("array2 at ",i," =", file_real)
        data2 = vip.fits.open_fits(file_real)
        SN[i] = vip.metrics.snr(array=data, source_xy=positions, fwhm=fwhm, plot=lets_plot, array2 = data2, use2alone=True)
        
    return contrast, SN

if __name__ == "__main__":
    print("###### Start to process the data ######")
    
    # companion
    start_time = datetime.datetime.now()
    
    # far disk 100 pxs
    #positions = [(28.33241, 158.96815), (225.53186, 88.48893), (92.048477, 28.688365), (166.44321, 225.1759)]
    
    # close disk 27 pxs
    #positions = [(100.14647, 136.80991), (154.69668, 117.49931), (116.43144, 100.14647), (139.30159, 153.89578)]
    
    # close companion 27 pxs
    #positions = [(103.0831, 136.98788), (137.25485, 152.9169), (153.27285, 118.56717), (118.74516, 101.65928)]
    # far companion 100 pxs
    positions = [(34.330781, 161.45329), (162.53369, 221.5536), (222.11243, 93.479071),(94.038796, 33.459928)]
    
    wl = 0
    # psf
    psf = vip.fits.open_fits(str(sys.argv[1]))
    fwhm = get_fwhm_from_psf(psf[0])
    psfn, fwhm_flux, fwhm_bis = vip.metrics.normalize_psf(psf[wl], fwhm, size=17, full_output=True)
    print("fwhm =", fwhm, "fwhm_bis =", fwhm_bis, "fwhm_flux[0] =", fwhm_flux[0])
    fwhm_for_snr=4
    
    # ./res_0907_presentation/frame_050/no_scale/disk_close_27pxs/pos1
    # ./res_0907_presentation/frame_050/no_scale/companion_close_27pxs
    # (1) header -> with or without injection 
    header = "./res_0907_presentation"
    header_real = "./res_0907_presentation_real"

    # (3) no scale vs spat_mean vs spat_annular_mean
    obj = "/companion_far_100pxs" 
    
    '''
    # no scale
    contrast_050_pos1_1, sn_050_pos1_1 = get_contrast_and_SN(header+"/frame_050/no_scale"+obj+"/pos1", positions[0], fwhm_for_snr, fwhm_flux[0], header_real+"/frame_050/no_scale"+obj+"/pos1")
    contrast_050_pos1_2, sn_050_pos1_2 = get_contrast_and_SN(header+"/frame_050/no_scale"+obj+"/pos1", positions[1], fwhm_for_snr, fwhm_flux[0], header_real+"/frame_050/no_scale"+obj+"/pos1")
    contrast_050_pos2_1, sn_050_pos2_1 = get_contrast_and_SN(header+"/frame_050/no_scale"+obj+"/pos2", positions[2], fwhm_for_snr, fwhm_flux[0], header_real+"/frame_050/no_scale"+obj+"/pos2")
    contrast_050_pos2_2, sn_050_pos2_2 = get_contrast_and_SN(header+"/frame_050/no_scale"+obj+"/pos2", positions[3], fwhm_for_snr, fwhm_flux[0], header_real+"/frame_050/no_scale"+obj+"/pos2")

    contrast_100_pos1_1, sn_100_pos1_1 = get_contrast_and_SN(header+"/frame_100/no_scale"+obj+"/pos1", positions[0], fwhm_for_snr, fwhm_flux[0], header_real+"/frame_100/no_scale"+obj+"/pos1")
    contrast_100_pos1_2, sn_100_pos1_2 = get_contrast_and_SN(header+"/frame_100/no_scale"+obj+"/pos1", positions[1], fwhm_for_snr, fwhm_flux[0], header_real+"/frame_100/no_scale"+obj+"/pos1")
    contrast_100_pos2_1, sn_100_pos2_1 = get_contrast_and_SN(header+"/frame_100/no_scale"+obj+"/pos2", positions[2], fwhm_for_snr, fwhm_flux[0], header_real+"/frame_100/no_scale"+obj+"/pos2")
    contrast_100_pos2_2, sn_100_pos2_2 = get_contrast_and_SN(header+"/frame_100/no_scale"+obj+"/pos2", positions[3], fwhm_for_snr, fwhm_flux[0], header_real+"/frame_100/no_scale"+obj+"/pos2")

    contrast_150_pos1_1, sn_150_pos1_1 = get_contrast_and_SN(header+"/frame_150/no_scale"+obj+"/pos1", positions[0], fwhm_for_snr, fwhm_flux[0], header_real+"/frame_150/no_scale"+obj+"/pos1")
    contrast_150_pos1_2, sn_150_pos1_2 = get_contrast_and_SN(header+"/frame_150/no_scale"+obj+"/pos1", positions[1], fwhm_for_snr, fwhm_flux[0], header_real+"/frame_150/no_scale"+obj+"/pos1")
    contrast_150_pos2_1, sn_150_pos2_1 = get_contrast_and_SN(header+"/frame_150/no_scale"+obj+"/pos2", positions[2], fwhm_for_snr, fwhm_flux[0], header_real+"/frame_150/no_scale"+obj+"/pos2")
    contrast_150_pos2_2, sn_150_pos2_2 = get_contrast_and_SN(header+"/frame_150/no_scale"+obj+"/pos2", positions[3], fwhm_for_snr, fwhm_flux[0], header_real+"/frame_150/no_scale"+obj+"/pos2")

    contrast_200_pos1_1, sn_200_pos1_1 = get_contrast_and_SN(header+"/frame_200/no_scale"+obj+"/pos1", positions[0], fwhm_for_snr, fwhm_flux[0], header_real+"/frame_200/no_scale"+obj+"/pos1")
    contrast_200_pos1_2, sn_200_pos1_2 = get_contrast_and_SN(header+"/frame_200/no_scale"+obj+"/pos1", positions[1], fwhm_for_snr, fwhm_flux[0], header_real+"/frame_200/no_scale"+obj+"/pos1")
    contrast_200_pos2_1, sn_200_pos2_1 = get_contrast_and_SN(header+"/frame_200/no_scale"+obj+"/pos2", positions[2], fwhm_for_snr, fwhm_flux[0], header_real+"/frame_200/no_scale"+obj+"/pos2")
    contrast_200_pos2_2, sn_200_pos2_2 = get_contrast_and_SN(header+"/frame_200/no_scale"+obj+"/pos2", positions[3], fwhm_for_snr, fwhm_flux[0], header_real+"/frame_200/no_scale"+obj+"/pos2")

    contrast_250_pos1_1, sn_250_pos1_1 = get_contrast_and_SN(header+"/frame_250/no_scale"+obj+"/pos1", positions[0], fwhm_for_snr, fwhm_flux[0], header_real+"/frame_250/no_scale"+obj+"/pos1")
    contrast_250_pos1_2, sn_250_pos1_2 = get_contrast_and_SN(header+"/frame_250/no_scale"+obj+"/pos1", positions[1], fwhm_for_snr, fwhm_flux[0], header_real+"/frame_250/no_scale"+obj+"/pos1")
    contrast_250_pos2_1, sn_250_pos2_1 = get_contrast_and_SN(header+"/frame_250/no_scale"+obj+"/pos2", positions[2], fwhm_for_snr, fwhm_flux[0], header_real+"/frame_250/no_scale"+obj+"/pos2")
    contrast_250_pos2_2, sn_250_pos2_2 = get_contrast_and_SN(header+"/frame_250/no_scale"+obj+"/pos2", positions[3], fwhm_for_snr, fwhm_flux[0], header_real+"/frame_250/no_scale"+obj+"/pos2")
    '''
    
    contrast_050_1, sn_050_1 = get_contrast_and_SN(header+"/frame_050/no_scale"+obj, positions[0], fwhm_for_snr, fwhm_flux[0], header_real+"/frame_050/no_scale"+obj)
    contrast_050_2, sn_050_2 = get_contrast_and_SN(header+"/frame_050/no_scale"+obj, positions[1], fwhm_for_snr, fwhm_flux[0], header_real+"/frame_050/no_scale"+obj)
    contrast_050_3, sn_050_3 = get_contrast_and_SN(header+"/frame_050/no_scale"+obj, positions[2], fwhm_for_snr, fwhm_flux[0], header_real+"/frame_050/no_scale"+obj)
    contrast_050_4, sn_050_4 = get_contrast_and_SN(header+"/frame_050/no_scale"+obj, positions[3], fwhm_for_snr, fwhm_flux[0], header_real+"/frame_050/no_scale"+obj)

    contrast_100_1, sn_100_1 = get_contrast_and_SN(header+"/frame_100/no_scale"+obj, positions[0], fwhm_for_snr, fwhm_flux[0], header_real+"/frame_100/no_scale"+obj)
    contrast_100_2, sn_100_2 = get_contrast_and_SN(header+"/frame_100/no_scale"+obj, positions[1], fwhm_for_snr, fwhm_flux[0], header_real+"/frame_100/no_scale"+obj)
    contrast_100_3, sn_100_3 = get_contrast_and_SN(header+"/frame_100/no_scale"+obj, positions[2], fwhm_for_snr, fwhm_flux[0], header_real+"/frame_100/no_scale"+obj)
    contrast_100_4, sn_100_4 = get_contrast_and_SN(header+"/frame_100/no_scale"+obj, positions[3], fwhm_for_snr, fwhm_flux[0], header_real+"/frame_100/no_scale"+obj)

    contrast_150_1, sn_150_1 = get_contrast_and_SN(header+"/frame_150/no_scale"+obj, positions[0], fwhm_for_snr, fwhm_flux[0], header_real+"/frame_150/no_scale"+obj)
    contrast_150_2, sn_150_2 = get_contrast_and_SN(header+"/frame_150/no_scale"+obj, positions[1], fwhm_for_snr, fwhm_flux[0], header_real+"/frame_150/no_scale"+obj)
    contrast_150_3, sn_150_3 = get_contrast_and_SN(header+"/frame_150/no_scale"+obj, positions[2], fwhm_for_snr, fwhm_flux[0], header_real+"/frame_150/no_scale"+obj)
    contrast_150_4, sn_150_4 = get_contrast_and_SN(header+"/frame_150/no_scale"+obj, positions[3], fwhm_for_snr, fwhm_flux[0], header_real+"/frame_150/no_scale"+obj)

    contrast_200_1, sn_200_1 = get_contrast_and_SN(header+"/frame_200/no_scale"+obj, positions[0], fwhm_for_snr, fwhm_flux[0], header_real+"/frame_200/no_scale"+obj)
    contrast_200_2, sn_200_2 = get_contrast_and_SN(header+"/frame_200/no_scale"+obj, positions[1], fwhm_for_snr, fwhm_flux[0], header_real+"/frame_200/no_scale"+obj)
    contrast_200_3, sn_200_3 = get_contrast_and_SN(header+"/frame_200/no_scale"+obj, positions[2], fwhm_for_snr, fwhm_flux[0], header_real+"/frame_200/no_scale"+obj)
    contrast_200_4, sn_200_4 = get_contrast_and_SN(header+"/frame_200/no_scale"+obj, positions[3], fwhm_for_snr, fwhm_flux[0], header_real+"/frame_200/no_scale"+obj)

    contrast_250_1, sn_250_1 = get_contrast_and_SN(header+"/frame_250/no_scale"+obj, positions[0], fwhm_for_snr, fwhm_flux[0], header_real+"/frame_250/no_scale"+obj)
    contrast_250_2, sn_250_2 = get_contrast_and_SN(header+"/frame_250/no_scale"+obj, positions[1], fwhm_for_snr, fwhm_flux[0], header_real+"/frame_250/no_scale"+obj)
    contrast_250_3, sn_250_3 = get_contrast_and_SN(header+"/frame_250/no_scale"+obj, positions[2], fwhm_for_snr, fwhm_flux[0], header_real+"/frame_250/no_scale"+obj)
    contrast_250_4, sn_250_4 = get_contrast_and_SN(header+"/frame_250/no_scale"+obj, positions[3], fwhm_for_snr, fwhm_flux[0], header_real+"/frame_250/no_scale"+obj)

    nb_data = 5
    '''
    l_050 = min(len(contrast_050_pos1_1), len(contrast_050_pos2_1))
    l_100 = min(len(contrast_100_pos1_1), len(contrast_100_pos2_1))
    l_150 = min(len(contrast_150_pos1_1), len(contrast_150_pos2_1))
    l_200 = min(len(contrast_200_pos1_1), len(contrast_200_pos2_1))
    l_250 = min(len(contrast_250_pos1_1), len(contrast_250_pos2_1))
    '''

    l_050 = len(contrast_050_1) 
    l_100 = len(contrast_100_1)
    l_150 = len(contrast_150_1)
    l_200 = len(contrast_200_1)
    l_250 = len(contrast_250_1)

    l = max([l_050, l_100, l_150, l_200, l_250])
    print("l =", l)
    # contrast
    data = np.zeros((l, nb_data))
    print("data.shape =", data.shape, " l=", l_050, l_100, l_150, l_200, l_250)

    for i in range(l):
        # 050
        if i >= l_050: 
            data[i, 0] = data[i-1, 0]
        else:
            data[i, 0] = (contrast_050_1[i] + contrast_050_2[i]+ contrast_050_3[i] + contrast_050_4[i])/4
        
        # 100
        if i >= l_100: 
            data[i, 1] = data[i-1, 1]
        else:
            data[i, 1] = (contrast_100_1[i] + contrast_100_2[i] + contrast_100_3[i] + contrast_100_4[i])/4
        
        # 150
        if i >= l_150: 
            data[i, 2] = data[i-1, 2]
        else:
            data[i, 2] = (contrast_150_1[i] + contrast_150_2[i] + contrast_150_3[i] + contrast_150_4[i])/4

        # 200
        if i >= l_200: 
            data[i, 3] = data[i-1, 3]
        else:
            data[i, 3] = (contrast_200_1[i] + contrast_200_2[i] + contrast_200_3[i] + contrast_200_4[i])/4

        # 250
        if i >= l_250: 
            data[i, 4] = data[i-1, 4]
        else:
            data[i, 4] = (contrast_250_1[i] + contrast_250_2[i] + contrast_250_3[i] + contrast_250_4[i])/4
    
    # snr
    data_sn = np.zeros((l, nb_data))
    for i in range(l):
        # 050
        if i >= l_050: 
            data_sn[i, 0] = data_sn[i-1, 0]
        else:
            data_sn[i, 0] = (sn_050_1[i] + sn_050_2[i]+ sn_050_3[i] + sn_050_4[i])/4
        
        # 100
        if i >= l_100: 
            data_sn[i, 1] = data_sn[i-1, 1]
        else:
            data_sn[i, 1] = (sn_100_1[i] + sn_100_2[i] + sn_100_3[i] + sn_100_4[i])/4
        
        # 150
        if i >= l_150: 
            data_sn[i, 2] = data_sn[i-1, 2]
        else:
            data_sn[i, 2] = (sn_150_1[i] + sn_150_2[i] + sn_150_3[i] + sn_150_4[i])/4

        # 200
        if i >= l_200: 
            data_sn[i, 3] = data_sn[i-1, 3]
        else:
            data_sn[i, 3] = (sn_200_1[i] + sn_200_2[i] + sn_200_3[i] + sn_200_4[i])/4

        # 250
        if i >= l_250: 
            data_sn[i, 4] = data_sn[i-1, 4]
        else:
            data_sn[i, 4] = (sn_250_1[i] + sn_250_2[i] + sn_250_3[i] + sn_250_4[i])/4   
    

    ###################
    # plot the result #
    ###################
    
    data_total = pd.DataFrame(data[:,:], columns=["050 frames","100 frames","150 frames","200 frames","250 frames"])
    data_total.index = data_total.index *20
    print("######### Contrast of companion #######")
    print(data_total)
    #data_total.to_csv("Flux_of_companion.csv")
    #data_total.plot(kind='line', style='--o', title='comparation')
    sns.set(font_scale = 1.5)
    sns.relplot(kind='line',data=data_total)
    plt.subplots_adjust(left=0.07, bottom=0.09, right=0.86, top=0.92)
    plt.title("RDI "+obj.split('/')[-1] +": contrast - the number of best correlated frames", fontsize = 20)
    plt.xlabel("K_kilp", fontsize = "18")
    plt.ylabel("Contrast - diameter 4 px", fontsize = "18")
    #plt.ylim(0,70)
    plt.show()
    
    data_total_SN = pd.DataFrame(data_sn[:,:], columns=["050 frames","100 frames","150 frames","200 frames","250 frames"])
    data_total_SN.index = data_total_SN.index *20
    print("######### S/N ########")
    print(data_total_SN)
    #data_total_SN.to_csv("SN_companions.csv")
    #data_total.plot(kind='line', style='--o', title='comparation')
    ax = sns.relplot(kind='line',data=data_total_SN)
    #plt.legend(fontsize = '16')
    plt.subplots_adjust(left=0.07, bottom=0.09, right=0.86, top=0.92)
    plt.title("RDI "+obj.split('/')[-1] +": signal to noise - the number of best correlated frames", fontsize=20)
    plt.xlabel("K_kilp", fontsize= "18")
    plt.ylabel("S/N ratio - diameter 4 px", fontsize = "18")
    #plt.ylim(0,20)
    plt.show()
    end_time = datetime.datetime.now()
    print("cost :", end_time - start_time)
    print("###### End of program ######")
