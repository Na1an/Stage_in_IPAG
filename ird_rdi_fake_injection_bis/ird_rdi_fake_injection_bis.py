"""
Injecte a fake companion into a IRD_SCIENCE_REDUCED_MASTER_CUBE

@Author : Yuchen BAI
@Date   : 05/08/2021
@Contact: yuchenbai@hotmail.com
"""
import os
import copy
import argparse
import warnings
import datetime
import numpy as np
import pandas as pd
import vip_hci as vip
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning
from photutils.aperture import CircularAperture, aperture_photometry, CircularAnnulus

"""
We have problem about ird_rdi_fake_injectioni_bis input one master_cube not all 
"""
###########
# setting #
###########

warnings.simplefilter('ignore', category=AstropyWarning)

############
# function #
############

# distance between two points
def distance(x1, y1, x2, y2):
    '''
    Args:
        x1 : an integer. object 1 - coordinate X
        y1 : an integer. object 1 - coordinate Y
        x2 : an integer. object 2 - coordinate X
        y2 : an integer. object 2 - coordinate Y
    Return:
        res : an integer. The distance between two points.
    '''
    return ((x1-x2)**2+(y1-y2)**2)**0.5

# take data from header
def take_data_from_header(header):
    '''
    Args:
        header : a fits header. Juest for displaying the detail.
    Return:
        None.
    '''
    res = []
    res.append(str(header["OBJECT"]))
    res.append(str(header["DATE-OBS"]))
    res.append(str(header["ESO OBS START"]))
    res.append(str(header["NAXIS3"]))
    res.append(str(header["DIT_MIN"]))

    return res

# get full width half maximum
def get_fwhm_from_psf(psf):
    '''
    Args:
        psf : a 2D np.ndarray. The PSF in one wavelength. 
    Return:
        res : a float. The one fwhm of the psf.
    ''' 
    fwhm = vip.var.fit_2dgaussian(psf, crop=True, cropsize=9, debug=False)
    return np.mean([fwhm.loc[0,'fwhm_y'], fwhm.loc[0,'fwhm_x']])

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

# take flux form the contrast
def get_flux_from_contrast(contrast):


#############
# main code #
#############
print("######### Start program : ird_rdi_injection_and_compute_contrast.py #########")
print("> [IMPORTANT] This recipe works only for the standard reduction! \n")
start_time = datetime.datetime.now()
parser = argparse.ArgumentParser(description="Inject a fake companion and compute the contrast, S/N and flux.")

# file .sof whille contain :
parser.add_argument("sof", help="file name of the sof file", type=str)
parser.add_argument("--psfn_size",help="this size will be used to calculate psfn", type=int, default=17)
parser.add_argument("--contrast", help="the contrast we want for our fake companion, the unit is factor of 6", type=float, default=0.000001)
parser.add_argument("--rad_dist", help="the distance/radius from the fake companion to star, default is 25", type=int, default=25)
parser.add_argument("--theta", help="the theta, default is 60", type=int, default=60)
parser.add_argument("--n_branches", help="how many brances we want", type=int, default=1)
parser.add_argument("--wl_channels", help="Spectral channel to use (to choose between 0 for channel 0, 1 for channel 1, 2 for both channels)", type=int, choices=[0,1,2], default=0)
parser.add_argument("--diameter", help="diameter of the fake companion", type=float, default=4.0)

########################### 
# Step-0 Handle arguments #
###########################
# all parameters needed are here
args = parser.parse_args()

# sof
sofname=args.sof

# --wl_channels
dico_conversion_wl_channels = {0 : [0], 1 : [1], 2 : [0,1]}
wl_channels = dico_conversion_wl_channels[args.wl_channels]
nb_wl = len(wl_channels)

# --psfn_size
psfn_size = args.psfn_size

# --contrast
contrast = args.contrast

# --rad_dist
rad_dist = args.rad_dist

# --theta
theta = args.theta

# --n_branches
n_branches = args.n_branches

# --diameter
diameter = args.diameter

# for the fake injection
# We need to prepare science cube, parallactic angle, psf, fwhm, psfn, pxscale 

###############################
# Step-1 Reading the sof file #
###############################
# Read Data from file .sof
data=np.loadtxt(sofname,dtype=str)
filenames=data[:,0]
datatypes=data[:,1]

science_cube_paths = filenames[np.where(datatypes == 'IRD_SCIENCE_REDUCED_MASTER_CUBE')[0]]
if len(science_cube_paths) != 1: 
    raise Exception('The sof file must contain only one IRD_SCIENCE_PSF_MASTER_CUBE file')
science_cube_path = science_cube_paths[0]

psf_paths = filenames[np.where(datatypes == 'IRD_SCIENCE_PSF_MASTER_CUBE')[0]]
if len(psf_paths) != 1: 
    raise Exception('The sof file must contain only one IRD_SCIENCE_PSF_MASTER_CUBE file')
psf_path = psf_paths[0]

angle_paths = filenames[np.where(datatypes == 'IRD_SCIENCE_PARA_ROTATION_CUBE')[0]]
if len(angle_paths) != 1: 
    raise Exception('The sof file must contain only one IRD_SCIENCE_PARA_ROTATION_CUBE file')
angle_path = angle_paths[0]

####################################################
# Step-2 take science cube, parallactic angle, psf #
####################################################

# science_cube
science_cube = fits.getdata(science_cube_path)
science_header = fits.getheader(science_cube_path)
print("> science cube path:", science_cube_path)

print("\n>> science cube - info\n")
data_sc = []
data_sc.append(take_data_from_header(science_header))
df_sc = pd.DataFrame(data=data_sc, columns=["OBJECT","DATE-OBS","OBS_STA","NB_FRAMES","DIT"])
print(df_sc.to_string())

print("\n=================== science cube, angle and psf =======================")
print(">> science cube DATE-OBS:", science_header["DATE-OBS"])
print(">> science cube OBJECT:", science_header["OBJECT"])
print(">> science cube EXPTIME:", science_header["EXPTIME"])
print(">> science cube ESO INS COMB ICOR:", science_header["ESO INS COMB ICOR"])
print(">> science cube ESO INS COMB IFLT:", science_header["ESO INS COMB IFLT"])

# science shape
nb_science_wl, nb_science_frames, nx, ny = science_cube.shape
print("> science_cube.shape =", science_cube.shape)

# take anglename
derotation_angles = fits.getdata(angle_path)
derotation_angles_header = fits.getheader(angle_path)
print("\n> corresponding parallactic angle", angle_path)
print(">> para DATE-OBS:", derotation_angles_header["DATE-OBS"])
print(">> para OBJECT:", derotation_angles_header["OBJECT"])
print(">> para EXPTIME:", derotation_angles_header["EXPTIME"])
print(">> para ESO INS COMB ICOR:", derotation_angles_header["ESO INS COMB ICOR"])
print(">> para ESO INS COMB IFLT:", derotation_angles_header["ESO INS COMB IFLT"])

if len(derotation_angles) != nb_science_frames:
    raise Exception('The science cube IRD_SCIENCE_REDUCED_MASTER_CUBE contains {0:d} frames while the list IRD_SCIENCE_PARA_ROTATION_CUBE contains {1:d} angles'.format(nb_science_frames,len(derotation_angles)))

# take psf
#psf = fits.getdata(get_psf_from_science_cube(science_cube_name))
psf = fits.getdata(psf_path)
psf_header = fits.getheader(psf_path)
print("\n> corresponding psf", psf_path)
print(">> psf DATE-OBS:", psf_header["DATE-OBS"])
print(">> psf OBJECT:", psf_header["OBJECT"])
print("\n=================== show info up =======================\n")

psf_0 = None
psf_1 = None
wl_final = wl_channels[0]

# fwhm psfn
fwhm = get_fwhm_from_psf(psf[wl_final])
psfn = vip.metrics.normalize_psf(psf[wl_final], fwhm, size=psfn_size)
print("psfn =", psfn.shape, "psfn.ndim =", psfn.ndim)

if nb_wl >1:
    fwhm_bis = get_fwhm_from_psf(psf[1])
    psfn_bis = vip.metrics.normalize_psf(psf[1], fwhm_bis, size=psfn_size)
    print("psfn =", psfn_bis.shape, "psfn.ndim =", psfn_bis.ndim)

# pxscale of IRDIS
pxscale = get_pxscale()

# get flux level
flux_level = get_flux_from_contrast()

################################
# Step-3 do the fake injection #
################################

# use vip to inject a fake companion
science_cube_fake_comp = np.zeros((2, nb_science_frames, nx, ny))
science_cube_fake_comp[wl_final] = vip.metrics.cube_inject_companions(science_cube[wl_final], psf_template=psfn, angle_list=-derotation_angles, flevel=flux_level, plsc=pxscale, rad_dists=[rad_dist], theta=theta, n_branches = n_branches)
if nb_wl>1:
    science_cube_fake_comp[1] = vip.metrics.cube_inject_companions(science_cube[1], psf_template=psfn_bis, angle_list=-derotation_angles, flevel=flux_level, plsc=pxscale, rad_dists=[rad_dist], theta=theta, n_branches = n_branches)

file_name = "rdi_res_"+str(wl_final)+".fits"
print("> The result will be stored in :", file_name)
hdu = fits.PrimaryHDU(data=res_0, header=science_header_0)
hdu.writeto(file_name)

file_name_fake = "rdi_res_fake_"+str(wl_final)+".fits"
print("> The result fake will be stored in :", file_name_fake)
hdu = fits.PrimaryHDU(data=res_0_fake, header=science_header_0)
hdu.writeto(file_name_fake)

# if we need two wavelengths
ref_frames_selected_bis = []
target_ref_coords_bis = []

if nb_wl>1:
    ref_frames_selected_bis, target_ref_coords_bis = selection_frame_based_score(corr_matrix[wl_channels[1]] ,science_cube_croped, n_corr, ref_frames, ref_cube_nb_frames, score, wave_length=wl_channels[1])
    dict_ref_in_target_bis = get_dict(ref_cube_path, target_ref_coords_bis)
    print(">> wave_length=1", dict_ref_in_target_bis)
    print(">> ref_frames_selected_bis.shape =", ref_frames_selected_bis.shape)
    res_1 = vip.pca.pca_fullfr.pca(science_cube_croped[wl_channels[1]]*mask, -derotation_angles, ncomp=ncomp, mask_center_px=inner_radius, cube_ref=ref_frames_selected_bis*mask, scaling=scaling)
    res_1_fake = vip.pca.pca_fullfr.pca(science_cube_croped[wl_channels[1]]*mask, -derotation_angles, ncomp=ncomp, mask_center_px=inner_radius, cube_ref=ref_frames_selected_bis*mask, scaling=scaling)

    # deep copy science header
    science_header_1 = copy.deepcopy(science_header)   
    science_header_1["HIERARCH wave_length"] = 1

    file_name = "rdi_res_1.fits"
    print("> The result will be stored in :", file_name)
    hdu = fits.PrimaryHDU(data=res_1, header=science_header_1)
    hdu.writeto(file_name)

    file_name_fake = "rdi_res_fake_1.fits"
    print("> The fake result will be stored in :", file_name_fake)
    hdu = fits.PrimaryHDU(data=res_1_fake, header=science_header_1)
    hdu.writeto(file_name_fake)

end_time = datetime.datetime.now()
print("######### End program : no error! Take:", end_time - start_time, "#########")