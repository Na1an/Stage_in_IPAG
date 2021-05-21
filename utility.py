import os
import cv2
import time
import numpy as np
import skimage
import vip_hci as vip
import matplotlib.pyplot as plt

from hciplot import plot_frames, plot_cubes

# start or end of the program
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
    #print("In SPHERE IRDIS : pxscale =", res, "arcsec/px")
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

# remvove the target from the reference list
def remove_target(target, refs):
    '''
    Args:
        target : a string.
        refs : a list of string
    Return:
        refs : a list of string. Target string removed.
    '''
    #res = refs
    for s in refs:
        if s.split('/')[-2] == target.split('/')[-1]:
            refs.remove(s)
            break
    return refs

# median of cube
def median_of_cube(cube, wl=0):
    '''
    Args:
        cube : a numpy.ndarray. (wavelengths, nb_frames, x, y)
        wl : a integer. Wave length of cube.
    Return:
        res : a numpy.ndarray, 2 dimensions. Ex. (256, 256).
    '''
    wave_length, sc_fr_nb, w, h = cube.shape
    res = np.zeros((w,h))
    for i in range(w):
        for j in range(h):
            res[i,j] = np.median(cube[wl,:,i,j])
    return res

