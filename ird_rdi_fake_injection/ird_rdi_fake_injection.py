"""
Injecte a fake companion into a IRD_SCIENCE_REDUCED_MASTER_CUBE and compute the contrast

@Author : Yuchen BAI
@Date   : 30/07/2021
@Contact: yuchenbai@hotmail.com
"""

import argparse
import warnings
import datetime
import numpy as np
import pandas as pd
import vip_hci as vip
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning
from photutils.aperture import CircularAperture, aperture_photometry, CircularAnnulus

###########
# setting #
###########

warnings.simplefilter('ignore', category=AstropyWarning)

############
# function #
############

# generate a list of tuple
def get_coords_of_ref_frames(nth, nb_frames):
    '''
    This function can help to preserve the information of frame coordinates. 
    Can help us find which image is used to process the target frame. 
    Args:
        nth : a integer. The nth cube in the reference library.
        nb_frames : a integer. The nth frame number of a cube.
    Return:
        res : a list of tuple. (nth cube, nth frame of the cube)
    '''
    res = []
    for i in range(nb_frames):
        res.append((nth, i, nb_frames))
    return res

# collect reference data from datacenter
def collect_frames(files_path, crop_size, full_output=True):
    '''
    Args:
        files_path : a list of string. files path contain keyword.
        crop_size : a integer. The size in center that we want process.
        full_output : a boolean. Default value is False. If it is true, we will return 3 result, if not, we inly return ref_frames. 
    Rrturn:
        ref_frames : ndarray, 4 dimensions. Return (wavelengths, nb_frames, x, y)
        ref_frames_coords : a list of tuple. [(0,0), (0,1), ...(nth cube, nth frame of the cube)]
        ref_cube_nb_frames : a list of integer. The list contains all frame numbers of the reference cube. 
    '''

    if not files_path:
        raise Exception("In the function collect_frames, there is nothing in the files_path which means no reference cube!!!")

    ref_frames_coords = []
    hd = fits.getdata(files_path[0])

    # frames in the first wavelength and second wavelength
    # K1/K2, H2/H3, etc...
    wl, nb_fr, w, h = hd.shape
    start = int((w-crop_size)//2)
    end = start + crop_size

    ref_frames = hd[..., start:end, start:end]
    ref_frames_coords = ref_frames_coords + get_coords_of_ref_frames(0, nb_fr)
    ref_cube_nb_frames = []
    ref_cube_nb_frames.append(nb_fr)

    for i in range(1,len(files_path)):
        hd = fits.getdata(files_path[i])
        wl, nb_fr, w, h = hd.shape
        ref_frames =np.append(ref_frames, hd[..., start:end, start:end], axis=1)
        ref_frames_coords = ref_frames_coords + get_coords_of_ref_frames(i, nb_fr)
        ref_cube_nb_frames.append(nb_fr)

    if full_output is False:
        return ref_frames, ref_frames_coords

    return ref_frames, ref_frames_coords, ref_cube_nb_frames

# get histogram of reference stars
def get_histogram_of_ref_stars_score(ref_star_scores, ref_cube_nb_frames):
    '''
    This function will count how many frames we use for each star in the reference library. 
    Args:
        ref_star_scores : a list of integer. The list of indice, nth frame in the reference frame library.
        ref_cube_nb_frames : a list of integer. Each element is the frame number of a reference star.
    Return:
        res : a ndarray list of integer. The number of integer for each reference star we use. 
    '''
    l = len(ref_cube_nb_frames)
    res = np.zeros(l)
    for i in ref_star_scores:
        # indice plus 1, then we can deal with it with the length of 
        i = i
        for n in range(l):
            i = i - ref_cube_nb_frames[n]
            if i<=0:
                res[n] = res[n] + 1
                break
    
    return res

# frame based version selection but with score system
def selection_frame_based_score(corr_matrix, target, nb_best_frame, ref_frames, ref_cube_nb_frames, score, wave_length):
    '''
    Args:
        corr_matrix : a numpy.ndarray, 2 dims. The correlation matrix.
        target : a numpy.ndarray, 4 dims. The science target cube, (wavelengths, nb_frames, x, y).
        nb_best : a integer. How many best frames fo the references stars array we want for each target frame.
        ref_frames : a numpy.ndarry, 4 dims. The reference stars data we have.
        ref_cube_nb_frames : a list of integer. Each element is the frame number of a reference star.
        score : a integer. We will pick all the reference stars which has higher or equal score.
        wave_length : a integer. Wave length of the reference cube.
    Rrturn:
        res : a ndarray, 3 dimensions. Return (nb_frames, x, y).
    '''
    # target shape
    wl_t, nb_fr_t, w, h = target.shape
    wl_ref, nb_fr_ref, w_ref, h_ref = ref_frames.shape

    # score_system
    ref_scores = np.zeros((nb_fr_ref))

    for i in range(nb_fr_t):
        tmp = {}
        for j in range(nb_fr_ref):
            tmp[j] = corr_matrix[i,j]
        
        if nb_best_frame > len(tmp):
            raise Exception("!!! inside the function selection_frame_based, tmp", len(tmp),"is samller than nb_best_frame", nb_best_frame)
        
        res_tmp = sorted(tmp.items(),key = lambda r:(r[1],r[0]), reverse=True)[0:nb_best_frame]
        
        for (ind, pcc) in res_tmp:
            ref_scores[ind] = ref_scores[ind] + 1

    res_coords = np.where(ref_scores>=score)
    print("res_coords.shape =", res_coords[0].shape, "res_coords.type = ", type(res_coords), " res_coords =", res_coords)
    res = ref_frames[wave_length][res_coords]
    print("res.shape =", res.shape)
    
    return res, get_histogram_of_ref_stars_score(res_coords[0], ref_cube_nb_frames)

# make a dictionary from two list
def get_dict(key, value):
    '''
    This function will count how many frames we use for each star in the reference library. 
    Args:
        key : a list of element, string or integer... Input key is the ref_files.
        value : a list of element, string or integer...
    Return:
        res : a dict. For drawing a barplot. 
    '''
    res = {}
    
    for i in range(len(key)):
        
        k = fits.open(key[i])[0].header['OBJECT']
        v = value[i]

        if v == 0:
            continue

        if k not in res.keys():
            res[k] = value[i]
        else:
            res[k] = res[k] + value[i]
        
    return res

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

# use inner mask and outer mask to calculate the pcc(pearson correlation coeffient)
def create_mask(crop_size, inner_radius, outer_radius):
    '''
    Args:
        crop_size : an integer. The size of frame/image.
        inner_radius : an integer.
        outer_radius : an integer. 
    Return:
        res : a numpy.ndarray, 2 dimens. Ex. (256, 256) but the center is all 0.
    '''
    count = 0
    res = np.full((crop_size, crop_size), True)
    x = crop_size//2
    y = crop_size//2
    for i in range(crop_size):
        for j in range(crop_size):
            if distance(i, j, x, y) >= outer_radius or distance(i, j, x, y) <= inner_radius:
                res[i,j] = False
                count = count + 1
    return res

# take data from header
def take_data_from_header(science_header):
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

# get angle from the science cube path
def get_para_angle_from_science_cube(path):
    '''
    Args:
        path : a string. Replace the last element after '/', then we can have the parallactic angle path.
    Return:
        res : a string. The parallactic angle path.
    '''
    return path.replace("ird_convert_recenter_dc5-IRD_SCIENCE_REDUCED_MASTER_CUBE-center_im.fits","ird_convert_recenter_dc5-IRD_SCIENCE_PARA_ROTATION_CUBE-rotnth.fits")

# get psf from the science cube path
def get_psf_from_science_cube(path):
    '''
    Args:
        path : a string. Replace the last element after '/', then we can have the PSF path.
    Return:
        res : a string. PSF path.
    '''
    return path.replace("ird_convert_recenter_dc5-IRD_SCIENCE_REDUCED_MASTER_CUBE-center_im.fits","IRD_SCIENCE_PSF_MASTER_CUBE-median_unsat.fits")

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

#############
# main code #
#############
print("######### Start program : ird_rdi_injection_and_compute_contrast.py #########")
print("> [IMPORTANT] This recipe works only for the standard reduction! \n")
start_time = datetime.datetime.now()
parser = argparse.ArgumentParser(description="Inject a fake companion and compute the contrast, S/N and flux.")

# file .sof whille contain :
parser.add_argument("sof", help="file name of the sof file", type=str)
parser.add_argument("--inner_radius",help="inner radius where the reduction starts", type=int, default=10)
parser.add_argument("--outer_radius",help="outer radius where the reduction starts", type=int, default=100)
parser.add_argument("--psfn_size",help="this size will be used to calculate psfn", type=int, default=17)
parser.add_argument("--science_object", help="the OBJECT keyword of the science target", type=str, default='unspecified')
parser.add_argument("--flux_level", help="flux level we will use in the fake injection, default is 40", type=int, default=40)
parser.add_argument("--rad_dist", help="the distance/radius from the fake companion to star, default is 25", type=int, default=25)
parser.add_argument("--theta", help="the theta, default is 60", type=int, default=60)
parser.add_argument("--n_branches", help="how many brances we want", type=int, default=1)
parser.add_argument("--wl_channels", help="Spectral channel to use (to choose between 0 for channel 0, 1 for channel 1, 2 for both channels)", type=int, choices=[0,1,2], default=0)
parser.add_argument("--score", help="which decide how we choose the reference frame (>=1)", type=int, default=1)
parser.add_argument("--n_corr", help="the number of best correalted frames for each frame of science target", type=int, default=150)
parser.add_argument("--ncomp",help="number of principal components to remove (5 by default)", type=int, default=5)
parser.add_argument("--scaling", help="scaling for the PCA (to choose between 0 for spat-mean, 1 for spat-standard, 2 for temp-mean, 3 for temp-standard or 4 for None)",\
                    type=int, choices=[0,1,2,3,4], default=0)
parser.add_argument("--r_aperture", help="radius to compute the flux/contrast", type=int, default=2)
parser.add_argument("--r_in_annulus", help="inner radius of annulus around the fake companion", type=int, default=4)
parser.add_argument("--r_out_annulus", help="outer radius of annulus around the fake companion", type=int, default=6)

########################### 
# Step-0 Handle arguments #
###########################
# all parameters needed are here
args = parser.parse_args()

# sof
sofname=args.sof

# --score
score = args.score

# --n_corr
n_corr = args.n_corr

# --ncomp
ncomp = args.ncomp

# --wl_channels
dico_conversion_wl_channels = {0 : [0], 1 : [1], 2 : [0,1]}
wl_channels = dico_conversion_wl_channels[args.wl_channels]
nb_wl = len(wl_channels)

# --scaling
scaling_dict = {0 : 'spat-mean', 1 : 'spat-standard', 2 : 'temp-mean', 3 : 'temp-standard', 4 : None}
scaling = scaling_dict[args.scaling]

# --crop_size and inner/outer radius
inner_radius = args.inner_radius
outer_radius = args.outer_radius
crop_size = 2*outer_radius+1

# --psfn_size
psfn_size = args.psfn_size

# --science_object
science_object = args.science_object

# --flux_level
flux_level = args.flux_level

# --rad_dist
rad_dist = args.rad_dist

# --theta
theta = args.theta

# --n_branches
n_branches = args.n_branches

# --r_apperture
r_aperture = args.r_aperture
r_in_annulus = args.r_in_annulus
r_out_annulus = args.r_out_annulus

# check crop_size type
if type(crop_size) not in [np.int64,np.int,int]:
    crop_size = int(crop_size)

if crop_size<=21:
    crop_size=21
    print('Warning cropsize<=21, too small! Value set to 21')

if outer_radius <= inner_radius:
    print("Warning outer_radius <= inner_radius! Value set to {0:d}".format(inner_radius+1))

# for the fake injection
# We need to prepare science cube, parallactic angle, psf, fwhm, psfn, pxscale 

###############################
# Step-1 Reading the sof file #
###############################
# Read Data from file .sof
data=np.loadtxt(sofname,dtype=str)
filenames=data[:,0]
datatypes=data[:,1]
cube_names = filenames[np.where(datatypes == 'IRD_SCIENCE_REDUCED_MASTER_CUBE')[0]]
nb_cubes = len(cube_names)

if nb_cubes < 2: 
    raise Exception('The sof file must contain at least 2 IRD_SCIENCE_REDUCED_MASTER_CUBE (science and reference)')

# except one science cube, the rest are reference cubes
nb_reference_cubes = nb_cubes - 1 

if science_object != 'unspecified':
    for i,cube_name in enumerate(cube_names):
        header =fits.getheader(cube_name)
        if header['OBJECT'].strip() == science_object.strip():
            science_cube_name = cube_name
            reference_cube_names = [cube_name for cube_name in cube_names if cube_name != science_cube_name]
            science_object_final = header['OBJECT']
            break
    
    try:
        print('\nScience OBJECT set to {0:s}'.format(science_object_final))
    except:
        print('Unable to detect the science cube from the IRD_SCIENCE_REDUCED_MASTER_CUBE. Using by default option the first cube as science')
        science_cube_name = cube_names[0]
        reference_cube_names = cube_names[1:]
else:
    science_cube_name = cube_names[0]
    reference_cube_names = cube_names[1:]

print("> science cube path:", science_cube_name)

####################################################
# Step-2 take science cube, parallactic angle, psf #
####################################################
# science_cube
science_cube = fits.getdata(science_cube_name)
science_header = fits.getheader(science_cube_name)

print("\n>> science cube - info\n")
data_sc = []
data_sc.append(take_data_from_header(science_header))
df_sc = pd.DataFrame(data=data_sc, columns=["OBJECT","DATE-OBS","OBS_STA","NB_FRAMES","DIT"])
print(df_sc.to_string())

print("\n=================== science cube and angle =======================")
print("> start test")
print(">> science cube DATE-OBS:", science_header["DATE-OBS"])
print(">> science cube OBJECT:", science_header["OBJECT"])
print(">> science cube EXPTIME:", science_header["EXPTIME"])
print(">> science cube ESO INS COMB ICOR:", science_header["ESO INS COMB ICOR"])
print(">> science cube ESO INS COMB IFLT:", science_header["ESO INS COMB IFLT"])

# science shape
nb_science_wl, nb_science_frames, nx, ny = science_cube.shape

# take anglename
anglename = get_para_angle_from_science_cube(science_cube_name)

derotation_angles = fits.getdata(anglename)
derotation_angles_header = fits.getheader(anglename)
print("> corresponding parallactic angle", anglename)
print(">> para DATE-OBS:", derotation_angles_header["DATE-OBS"])
print(">> para OBJECT:", derotation_angles_header["OBJECT"])
print(">> para EXPTIME:", derotation_angles_header["EXPTIME"])
print(">> para ESO INS COMB ICOR:", derotation_angles_header["ESO INS COMB ICOR"])
print(">> para ESO INS COMB IFLT:", derotation_angles_header["ESO INS COMB IFLT"])

if len(derotation_angles) != nb_science_frames:
    raise Exception('The science cube IRD_SCIENCE_REDUCED_MASTER_CUBE contains {0:d} frames while the list IRD_SCIENCE_PARA_ROTATION_CUBE contains {1:d} angles'.format(nb_science_frames,len(derotation_angles)))

# take psf
psf = fits.getdata(get_psf_from_science_cube(science_cube_name))
psf_0 = None
psf_1 = None
wl_final = wl_channels[0]
psf_0 = psf[wl_final]

# fwhm psfn
fwhm = get_fwhm_from_psf(psf[wl_final])
psfn, fwhm_flux, fwhm_2 = vip.metrics.normalize_psf(psf[wl_final], fwhm, size=psfn_size, full_output=True)
print("psfn =", psfn.shape, "psfn.ndim =", psfn.ndim)

if nb_wl >1:
    psf_1 = psf[1]
    fwhm_bis = get_fwhm_from_psf(psf[1])
    psfn_bis = vip.metrics.normalize_psf(psf[1], fwhm_bis, size=psfn_size)
    print("psfn =", psfn_bis.shape, "psfn.ndim =", psfn_bis.ndim)

# pxscale of IRDIS
pxscale = get_pxscale()

################################
# Step-3 do the fake injection #
################################

# use vip to inject a fake companion
science_cube_fake_comp = np.zeros((2, nb_science_frames, nx, ny))
science_cube_fake_comp[wl_final] = vip.metrics.cube_inject_companions(science_cube[wl_final], psf_template=psfn, angle_list=-derotation_angles, flevel=flux_level, plsc=pxscale, rad_dists=[rad_dist], theta=theta, n_branches = n_branches)
if nb_wl>1:
    science_cube_fake_comp[1] = vip.metrics.cube_inject_companions(science_cube[1], psf_template=psfn_bis, angle_list=-derotation_angles, flevel=flux_level, plsc=pxscale, rad_dists=[rad_dist], theta=theta, n_branches = n_branches)

##############################
# Step-4 take reference cube #
##############################
# sort reference cube names/paths
reference_cube_names.sort()

# collect data, then we have reference frames
border_l = ny//2 - crop_size//2
border_r = ny//2 + crop_size//2 + 1
ref_frames = None
ref_nb_frames = []
reference_cube_names_remove_dup = []

# indice start
ind_start = 0
print("\n>> reference cube - info \n")

data_ref = []
for i in range(len(reference_cube_names)):
    name = reference_cube_names[i]
    tmp_cube = fits.getdata(name)
    tmp_header = fits.getheader(name)
    if tmp_header["OBJECT"] == science_header["OBJECT"]:
        ind_start = ind_start+1
        continue
    data_ref.append(take_data_from_header(tmp_header))
    if i==ind_start:
        ref_frames = tmp_cube[..., border_l:border_r, border_l:border_r]
        ref_nb_frames.append(len(tmp_cube[0]))
    else:
        ref_nb_frames.append(len(tmp_cube[0]))
        ref_frames = np.append(ref_frames, tmp_cube[..., border_l:border_r, border_l:border_r], axis=1)
    reference_cube_names_remove_dup.append(name)

df_ref = pd.DataFrame(data=data_ref, columns=["OBJECT","DATE-OBS","OBS_STA","NB_FRAMES","DIT"])
print(df_ref.to_string())

print("\n> ref_frames.shape =", ref_frames.shape)
wl_ref, nb_ref_frames, ref_x, ref_y = ref_frames.shape

# correlation matrix
mask = create_mask(crop_size, inner_radius, outer_radius)
corr_matrix = np.zeros((nb_wl, nb_science_frames, nb_ref_frames))
science_cube_croped = science_cube[..., border_l:border_r, border_l:border_r]
science_cube_croped_fake = science_cube_fake_comp[..., border_l:border_r, border_l:border_r]

for w in range(nb_wl):
    wl = wl_channels[w]
    for i in range(nb_science_frames):
        for j in range(nb_ref_frames):
            corr_matrix[wl, i, j] = np.corrcoef(np.reshape(science_cube_croped[wl, i]*mask, ref_x*ref_y), np.reshape(ref_frames[wl, j]*mask, ref_x*ref_y))[0,1]
print("> corr_matrix.shape", corr_matrix.shape)

# store the fwhm_flux in to the science header
science_header["HIERARCH fwhm_flux_0"] = fwhm_flux[0]
science_header["HIERARCH fwhm_flux_1"] = fwhm_flux[1]

# do the selection
ref_frames_selected, target_ref_coords = selection_frame_based_score(corr_matrix[wl_final], science_cube_croped, n_corr, ref_frames, ref_nb_frames, score, wave_length=wl_final)
dict_ref_in_target = get_dict(reference_cube_names, target_ref_coords)
print(">> wave_length=0", dict_ref_in_target)
print(">> ref_frames_selected.shape =", ref_frames_selected.shape)
res_0 = vip.pca.pca_fullfr.pca(science_cube_croped[wl_channels[0]]*mask, -derotation_angles, ncomp=ncomp, mask_center_px=inner_radius, cube_ref=ref_frames_selected*mask, scaling=scaling)
res_0_fake = vip.pca.pca_fullfr.pca(science_cube_croped[wl_channels[0]]*mask, -derotation_angles, ncomp=ncomp, mask_center_px=inner_radius, cube_ref=ref_frames_selected*mask, scaling=scaling)

file_name = "rdi_res_"+str(wl_final)+".fits"
print("> The result will be stored in :", file_name)
hdu = fits.PrimaryHDU(data=res_0, header=science_header)
hdu.writeto(file_name)

file_name_fake = "rdi_res_fake_"+str(wl_final)+".fits"
print("> The result fake will be stored in :", file_name_fake)
hdu = fits.PrimaryHDU(data=res_0_fake, header=science_header)
hdu.writeto(file_name_fake)

# if we need two wavelengths
ref_frames_selected_bis = []
target_ref_coords_bis = []
if nb_wl>1:
    ref_frames_selected_bis, target_ref_coords_bis = selection_frame_based_score(corr_matrix[wl_channels[1]] ,science_cube_croped, n_corr, ref_frames, ref_nb_frames, score, wave_length=wl_channels[1])
    dict_ref_in_target_bis = get_dict(reference_cube_names, target_ref_coords_bis)
    print(">> wave_length=1", dict_ref_in_target_bis)
    print(">> ref_frames_selected_bis.shape =", ref_frames_selected_bis.shape)
    res_1 = vip.pca.pca_fullfr.pca(science_cube_croped[wl_channels[1]]*mask, -derotation_angles, ncomp=ncomp, mask_center_px=inner_radius, cube_ref=ref_frames_selected_bis*mask, scaling=scaling)
    res_1_fake = vip.pca.pca_fullfr.pca(science_cube_croped[wl_channels[1]]*mask, -derotation_angles, ncomp=ncomp, mask_center_px=inner_radius, cube_ref=ref_frames_selected_bis*mask, scaling=scaling)

    file_name = "rdi_res_1.fits"
    print("> The result will be stored in :", file_name)
    hdu = fits.PrimaryHDU(data=res_1, header=science_header)
    hdu.writeto(file_name)

    file_name_fake = "rdi_res_fake_1.fits"
    print("> The fake result will be stored in :", file_name_fake)
    hdu = fits.PrimaryHDU(data=res_1_fake, header=science_header)
    hdu.writeto(file_name_fake)

end_time = datetime.datetime.now()
print("######### End program : no error! Take:", end_time - start_time, "#########")