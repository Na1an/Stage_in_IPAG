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
        #bkg_mean = flux_companion['aperture_sum_1']/annulus.area
        #bkg_sum_in_companion = bkg_mean * aperture.area
        #flux[i] = flux_companion['aperture_sum_0'] - bkg_sum_in_companion 
        flux[i] = (flux_companion['aperture_sum_0']/aperture.area)

        # SN
        lets_plot = False
        if i==2:
            lets_plot = True
            #ds9.display(data)
        SN[i] = vip.metrics.snr(data, source_xy=positions[0], fwhm=fwhm, plot=lets_plot)

    return flux, SN

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
    
    # flux
    aperture = CircularAperture(positions, r=2)
    annulus = CircularAnnulus(positions, r_in=4, r_out=6)
 
    # flux
    flux_companion = aperture_photometry(data, [aperture, annulus])
    flux_companion['aperture_sum_0','aperture_sum_1'].info.format = '%.8g'
    flux = (flux_companion['aperture_sum_0']/aperture.area)/fwhm_flux

    # SN
    ds9 = vip.Ds9Window()
    ds9.display(data)
    SN = vip.metrics.snr(data, source_xy=positions[0], fwhm=fwhm, plot=True)

    return flux[0], SN

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
    positions = [(501.4, 493)]
    # fake disk far, 100pxs
    #positions = [(464.5, 423.5)]

    wl = 0
    # psf
    psf = vip.fits.open_fits(str(sys.argv[1]))
    fwhm = get_fwhm_from_psf(psf[0])
    psfn, fwhm_flux, fwhm_bis = vip.metrics.normalize_psf(psf[wl], fwhm, size=17, full_output=True)
    print("fwhm =", fwhm, "fwhm_bis =", fwhm_bis, "fwhm_flux[0] =", fwhm_flux[0])
    fwhm_for_snr=4
    data = vip.fits.open_fits(str(sys.argv[2]))
    contrast, snr = get_one_contrast_and_SN(data[3,0], positions, fwhm_for_snr, fwhm_flux[0])
    print("contrast =", contrast, "snr =", snr)
