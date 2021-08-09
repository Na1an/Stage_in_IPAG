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

# get coordinates from a string
def get_coords_from_str(coords_str):
    '''
    Args:
        coords_str : a string. "(1,2);(3,4);(5,6);" just like this.
    Return:
        res : a list of tuple. [(1,2),(3,4),(5,6)]
    '''
    res = []
    for c in coords_str.split(';'):
        if ',' not in c:
            continue
        tmp = i.replace('(','').replace(')','').split(',')
        res.append(float(tmp[0]), float(tmp[1]))

    return res

# get contrast and S/N ratio
def get_contrast_and_SN(res_fake, res_real, positions, fwhm_for_snr, fwhm_flux, r_aperture, r_in_annulus, r_out_annulus):
    '''
    Args:
        res_fake : a 2D np.array. The path of repository where the files are.
        res_real : a 2D np.array. The path of another repository where the files are, for calculating snr.
        positions : a list of tuple (x,y). The coordinates of companions.
        fwhm : a float. fwhm's diameter.
        fwhm_flux : a float. The flux of fwhm.
    Return:
        contrast : a np.array, 1 dimension. Store the list of each companion's flux.
        SN : a np.array, 1 dimension. Store the list of each companion's Signal to Noise ratio.
    '''
    
    aperture = CircularAperture(positions, r=r_aperture)
    annulus = CircularAnnulus(positions, r_in=r_in_annulus, r_out=r_out_annulus)
    
    # contrast
    flux_companion = aperture_photometry(res_fake, [aperture, annulus])
    flux_companion['aperture_sum_0','aperture_sum_1'].info.format = '%.8g'
    flux = flux_companion['aperture_sum_0'] 
    contrast = (flux_companion['aperture_sum_0']/aperture.area)/fwhm_flux

    # SN
    SN = vip.metrics.snr(array=res_fake, source_xy=positions, fwhm=fwhm_for_snr, plot=False, array2 =res_real, use2alone=True)
        
    return contrast, SN, flux

#############
# main code #
#############
print("######### Start program : ird_rdi_compute_contrast.py #########")
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

# position
parser.add_argument("coordinates", help="positions of fake companion, a string", type=str)
parser.add_argument("--fwhm", help="the diameter for calculating snr", type=int, default=4)

###########################
# Step-0 Handle arguments #
###########################
# all parameters needed are here
args = parser.parse_args()

# sof
sofname=args.sof

# coordinates
coords = get_coords_from_str(args.coordinates)

# diameter is 4 pixels for calculating S/N
fwhm_for_snr= args.fwhm

# flux, contrast and S/N
#pos = (10,10)

# calculating contrast, S/N and flux
contrast, sn, flux = get_contrast_and_SN(res_0_fake, res_0, pos, fwhm_for_snr, fwhm_flux[0], r_aperture, r_in_annulus, r_out_annulus)
res_contrast = {'fake_comp_wl_0':{'contrast':contrast, 'sn':sn, 'flux':flux}}

df = pd.DataFrame(data=res_contrast)
df.to_csv(r'ird_rdi_fake_injeciton_res.txt', sep=' ', mode='a')

end_time = datetime.datetime.now()
print("######### End program : no error! Take:", end_time - start_time, "#########")