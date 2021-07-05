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
        if i==-2:
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
    # positions = [(28.33241, 158.96815), (225.53186, 88.48893), (92.048477, 28.688365), (166.44321, 225.1759)]
    
    # close disk 27 pxs
    #positions = [(100.14647, 136.80991), (154.69668, 117.49931), (116.43144, 100.14647), (139.30159, 153.89578)]
    
    # close companion 27 pxs
    # positions = [(103.0831, 136.98788), (137.25485, 152.9169), (153.27285, 118.56717), (118.74516, 101.65928)]
    # far companion 100 pxs
    positions = [(34.330781, 161.45329), (162.53369, 221.5536), (222.11243, 93.479071),(94.038796, 33.459928)]
    
    wl = 0
    # psf
    psf = vip.fits.open_fits(str(sys.argv[1]))
    fwhm = get_fwhm_from_psf(psf[0])
    psfn, fwhm_flux, fwhm_bis = vip.metrics.normalize_psf(psf[wl], fwhm, size=17, full_output=True)
    print("fwhm =", fwhm, "fwhm_bis =", fwhm_bis, "fwhm_flux[0] =", fwhm_flux[0])
    fwhm_for_snr=4
    
    # (1) header -> with or without injection 
    header = "./res_0907_presentation"
    header_real = "./res_0907_presentation_real"
    
    # (2) nb frames
    path_frame = "/frame_250"
    
    # (3) no scale vs spat_mean vs spat_annular_mean
    obj = "/companion_far_100pxs" 
    '''
    path_no_scale_pos1 = "/no_scale" + obj + "/pos1"
    path_no_scale_pos2 = "/no_scale" + obj +"/pos2"

    path_spat_mean_pos1 = "/spat_mean" + obj + "/pos1"
    path_spat_mean_pos2 = "/spat_mean" + obj + "/pos2"
    
    path_sam_pos1 = "/spat_annular_mean" + obj + "/pos1"
    path_sam_pos2 = "/spat_annular_mean" + obj + "/pos2"
    '''
    path_no_scale = "/no_scale" + obj 
    path_spat_mean = "/spat_mean" + obj 
    path_sam = "/spat_annular_mean" + obj 

    # no scale
    ns_contrast1, ns_sn1 = get_contrast_and_SN(header+path_frame+path_no_scale, positions[0], fwhm_for_snr, fwhm_flux[0], header_real+path_frame+path_no_scale)
    ns_contrast2, ns_sn2 = get_contrast_and_SN(header+path_frame+path_no_scale, positions[1], fwhm_for_snr, fwhm_flux[0], header_real+path_frame+path_no_scale)
    ns_contrast3, ns_sn3 = get_contrast_and_SN(header+path_frame+path_no_scale, positions[2], fwhm_for_snr, fwhm_flux[0], header_real+path_frame+path_no_scale)
    ns_contrast4, ns_sn4 = get_contrast_and_SN(header+path_frame+path_no_scale, positions[3], fwhm_for_snr, fwhm_flux[0], header_real+path_frame+path_no_scale)

    
    # spat_mean
    sm_contrast1, sm_sn1 = get_contrast_and_SN(header+path_frame+path_spat_mean, positions[0], fwhm_for_snr, fwhm_flux[0], header_real+path_frame+path_spat_mean)
    sm_contrast2, sm_sn2 = get_contrast_and_SN(header+path_frame+path_spat_mean, positions[1], fwhm_for_snr, fwhm_flux[0], header_real+path_frame+path_spat_mean)
    sm_contrast3, sm_sn3 = get_contrast_and_SN(header+path_frame+path_spat_mean, positions[2], fwhm_for_snr, fwhm_flux[0], header_real+path_frame+path_spat_mean)
    sm_contrast4, sm_sn4 = get_contrast_and_SN(header+path_frame+path_spat_mean, positions[3], fwhm_for_snr, fwhm_flux[0], header_real+path_frame+path_spat_mean)

    # sam
    sam_contrast1, sam_sn1 = get_contrast_and_SN(header+path_frame+path_sam, positions[0], fwhm_for_snr, fwhm_flux[0], header_real+path_frame+path_sam)
    sam_contrast2, sam_sn2 = get_contrast_and_SN(header+path_frame+path_sam, positions[1], fwhm_for_snr, fwhm_flux[0], header_real+path_frame+path_sam)
    sam_contrast3, sam_sn3 = get_contrast_and_SN(header+path_frame+path_sam, positions[2], fwhm_for_snr, fwhm_flux[0], header_real+path_frame+path_sam)
    sam_contrast4, sam_sn4 = get_contrast_and_SN(header+path_frame+path_sam, positions[3], fwhm_for_snr, fwhm_flux[0], header_real+path_frame+path_sam)


    nb_data = 4
    l = min(len(ns_contrast1), len(ns_contrast3))
    # contrast
    data = np.zeros((l, nb_data))
    print("data.shape =", data.shape, "contrast.shape", ns_contrast1.shape, " ", ns_contrast2.shape," ",ns_contrast3.shape," ",ns_contrast4.shape)
    

    for i in range(l):
        print("i =", i)
        #data[i, 0] = adi_real_contrast 
        data[i, 1] = (ns_contrast1[i] + ns_contrast2[i] + ns_contrast3[i] + ns_contrast4[i])/4 
        data[i, 2] = (sm_contrast1[i] + sm_contrast2[i] + sm_contrast3[i] + sm_contrast4[i])/4
        data[i, 3] = (sam_contrast1[i] + sam_contrast2[i] + sam_contrast3[i] + sam_contrast4[i])/4
    
    # snr
    data_sn = np.zeros((l, nb_data))
    for i in range(l):
        #data_sn[i, 0] = adi_real_sn 
        data_sn[i, 1] = (ns_sn1[i] + ns_sn2[i] + ns_sn3[i] + ns_sn4[i])/4 
        data_sn[i, 2] = (sm_sn1[i] + sm_sn2[i] + sm_sn3[i] + sm_sn4[i])/4
        data_sn[i, 3] = (sam_sn1[i] + sam_sn2[i] + sam_sn3[i] + sam_sn4[i])/4
    
    ##################
    # plot te result #
    ##################
    
    data_total = pd.DataFrame(data[:,1:], columns=["no scale", "spat-mean", "spat-annular-mean"])
    data_total.index = data_total.index *20
    print("######### Contrast of companion #######")
    print(data_total)
    #data_total.to_csv("Flux_of_companion.csv")
    #data_total.plot(kind='line', style='--o', title='comparation')
    sns.set(font_scale = 1.5)
    sns.relplot(kind='line',data=data_total)
    plt.subplots_adjust(left=0.07, bottom=0.09, right=0.86, top=0.92)
    plt.title("RDI far companion in 100 pxs : contrast - no scale vs spat-mean vs spat_annular_mean [" + path_frame.replace('/','')+"]", fontsize = 20)
    plt.xlabel("K_kilp", fontsize = "18")
    plt.ylabel("Contrast - diameter 4 px", fontsize = "18")
    #plt.ylim(0,70)
    plt.show()
    
    data_total_SN = pd.DataFrame(data_sn[:,1:], columns=["no scale", "spat-mean", "spat-annular-mean"])
    data_total_SN.index = data_total_SN.index *20
    print("######### S/N ########")
    print(data_total_SN)
    #data_total_SN.to_csv("SN_companions.csv")
    #data_total.plot(kind='line', style='--o', title='comparation')
    ax = sns.relplot(kind='line',data=data_total_SN)
    #plt.legend(fontsize = '16')
    plt.subplots_adjust(left=0.07, bottom=0.09, right=0.86, top=0.92)
    plt.title("RDI far companion in 100 pxs : signal to noise - no scale vs spat-mean vs spat_annular_mean ["+ path_frame.replace('/','')+"]", fontsize=20)
    plt.xlabel("K_kilp", fontsize= "18")
    plt.ylabel("S/N ratio - diameter 4 px", fontsize = "18")
    #plt.ylim(0,20)
    plt.show()
    end_time = datetime.datetime.now()
    print("cost :", end_time - start_time)
    print("###### End of program ######")
