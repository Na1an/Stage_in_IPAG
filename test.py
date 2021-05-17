import os
import cv2
import time
import numpy as np
import skimage
import vip_hci as vip
import matplotlib.pyplot as plt

from hciplot import plot_frames, plot_cubes

def start_and_end_program(start):
    '''
    Args:
       start : a boolean. If it is the start of the program 
    Return:
        None
    '''

    localtime = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))

    if(start):
        print("######### program start :", localtime,"###########")
    else:
        print("########## program end :", localtime, "############")

# get one fwhm of a frame
def get_fwhm_from_psf(psf):
    
    fwhm = vip.var.fit_2dgaussian(psf, crop=True, cropsize=9, debug=False)
    
    return np.mean([fwhm.loc[0,'fwhm_y'], fwhm.loc[0,'fwhm_x']])

if __name__ == "__main__":
    start_and_end_program(True)
    print("vip.version :", vip.__version__)
    
    science_target = vip.fits.open_fits("../SPHERE_DC_DATA/HD 156384Cf_DB_H23_2017-06-27_ird_convert_recenter_dc5_PUBLIC_208368/ird_convert_recenter_dc5-IRD_SCIENCE_REDUCED_MASTER_CUBE-center_im.fits")
    psf = vip.fits.open_fits("../SPHERE_DC_DATA/HD 156384Cf_DB_H23_2017-06-27_ird_convert_recenter_dc5_PUBLIC_208368/ird_convert_recenter_dc5-IRD_SCIENCE_PSF_MASTER_CUBE-median_unsat.fits")
    angles = vip.fits.open_fits("../SPHERE_DC_DATA/HD 156384Cf_DB_H23_2017-06-27_ird_convert_recenter_dc5_PUBLIC_208368/ird_convert_recenter_dc5-IRD_SCIENCE_PARA_ROTATION_CUBE-rotnth.fits")

    fwhm = vip.var.fit_2dgaussian(psf[0], crop=True, cropsize=9, debug=False)
    
    print("one fwhm = ", get_fwhm_from_psf(psf[0]))

    # display
    #ds9 = vip.Ds9Window()
    
    # take info of the file *.fits
    # vip.fits.info_fits("../SPHERE_DC_DATA/HD 156384Cf_DB_H23_2017-06-27_ird_convert_recenter_dc5_PUBLIC_208368/ird_convert_recenter_dc5-IRD_SCIENCE_REDUCED_MASTER_CUBE-center_im.fits")

    #ds9.display()

    start_and_end_program(False)
