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
    '''
    Args:
        psf : a 2D np.ndarray. The PSF in one wavelength. 
    Return:
        res : a float. The one fwhm of the psf.
    ''' 
    fwhm = vip.var.fit_2dgaussian(psf, crop=True, cropsize=9, debug=False)
    
    return np.mean([fwhm.loc[0,'fwhm_y'], fwhm.loc[0,'fwhm_x']])

# print info for a file *.fits
def print_info(path):
    '''
    Args:
        path : a string. The path of one file *.fits
    Return:
        None.
    '''
    vip.fits.info_fits(path)

# get pxscale of the IRDIS
def get_pxscale():
    '''
    Args:
        None.
    Return:
        res : a float. Return the pxscale of the IRDIS.
    '''
    res = vip.conf.VLT_SPHERE_IRDIS['plsc']
    print("In SPHERE IRDIS : pxscale =", res, "arcsec/px")
    return res 

# slice frame, we only take the interesting area
# for exemple, 1/4 in the center of each frame
def crop_frame(science_target, size, center_scale):
    '''
    Args:
        science_target : np.array, 4 dims. Contains all the frames on all wavelength in a cube
        size : a int. Frames size, size = 1024 in case 1024*1024.
        center_scale : a float. The scale in center that we will process.
    Return:
        Sliced frames, np.array, 3 dims.
    '''

    tmp = (1-center_scale)*0.5
    res = science_target[..., int(size*tmp):int(size*(1-tmp)), int(size*tmp):int(size*(1-tmp))]
    return res

if __name__ == "__main__":
    start_and_end_program(True)
    print("vip.version :", vip.__version__)
    
    # read data
    science_target = vip.fits.open_fits("../SPHERE_DC_DATA/HD 156384Cf_DB_H23_2017-06-27_ird_convert_recenter_dc5_PUBLIC_208368/ird_convert_recenter_dc5-IRD_SCIENCE_REDUCED_MASTER_CUBE-center_im.fits")
    psf = vip.fits.open_fits("../SPHERE_DC_DATA/HD 156384Cf_DB_H23_2017-06-27_ird_convert_recenter_dc5_PUBLIC_208368/ird_convert_recenter_dc5-IRD_SCIENCE_PSF_MASTER_CUBE-median_unsat.fits")
    angles = vip.fits.open_fits("../SPHERE_DC_DATA/HD 156384Cf_DB_H23_2017-06-27_ird_convert_recenter_dc5_PUBLIC_208368/ird_convert_recenter_dc5-IRD_SCIENCE_PARA_ROTATION_CUBE-rotnth.fits")
    print("science_target type =", type(science_target))

    # fwhm
    fwhm_df = vip.var.fit_2dgaussian(psf[0], crop=True, cropsize=9, debug=False)
    fwhm = get_fwhm_from_psf(psf[0])  
    print("one fwhm =", fwhm)

    #psfn = vip.metrics.normalize_psf(psf[0], fwhm=fwhm, size=None) 
    #plot_frames(psfn, grid=True, size_factor=4)
    #print_info("../SPHERE_DC_DATA/HD 156384Cf_DB_H23_2017-06-27_ird_convert_recenter_dc5_PUBLIC_208368/ird_convert_recenter_dc5-IRD_SCIENCE_REDUCED_MASTER_CUBE-center_im.fits")
    
    #
    try_1 = vip.fits.open_fits("../K_kilp_ADI_RDI/RDI_WITH_MASK_3_best_32/RDI_Masked06.fits")
    #science_target_croped = crop_frame(try_1,len(science_target[0,0,0]),0.25)
    
    print(len(science_target[0,0,0]))
    #vip.pca.pca(science_target_croped, angles, ncomp=20, verbose=False)
    sn = vip.metrics.snr(try_1, source_xy=(126.22,249.025), fwhm=fwhm, plot=True)
    print("sn = ", sn)
    # display
    #ds9 = vip.Ds9Window()
    
    # take info of the file *.fits
    # vip.fits.info_fits("../SPHERE_DC_DATA/HD 156384Cf_DB_H23_2017-06-27_ird_convert_recenter_dc5_PUBLIC_208368/ird_convert_recenter_dc5-IRD_SCIENCE_REDUCED_MASTER_CUBE-center_im.fits")

    #ds9.display()

    start_and_end_program(False)
