"""
Compute the contrast, flux and S/N of a science cube with fake injection, companion

@Author : Yuchen BAI
@Date   : 09/08/2021
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
        res.append((float(tmp[0]), float(tmp[1])))

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

# get real res from the fake res path
def get_real_res_path_from_fake_res_path(path, wl):
    '''
    Args:
        path : a string. Replace the last element after '/', then we can have the parallactic angle path.
        wl : a int. Which wave length we will replace.
    Return:
        res : a string. The parallactic angle path.
    '''
    res = ""
    if wl == 0:
        res = path.replace("rdi_res_fake_0.fits","rdi_res_0.fits")
    else:
        res = path.replace("rdi_res_fake_1.fits","rdi_res_1.fits")
    return res

#############
# main code #
#############
print("######### Start program : ird_rdi_compute_contrast.py #########")
print("> [IMPORTANT] This recipe works only for the standard reduction! \n")
start_time = datetime.datetime.now()
parser = argparse.ArgumentParser(description="Inject a fake companion and compute the contrast, S/N and flux.")

# file .sof whille contain :
parser.add_argument("sof", help="file name of the sof file", type=str)
# position
parser.add_argument("--coordinates", help="positions of fake companion, a string", type=str, default="empty")
parser.add_argument("--fwhm", help="the diameter for calculating snr", type=int, default=4)
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

# coordinates
if args.coordinates == "empty":
    raise Exception("The coordinates can't be empty!!! The programe ends here.")

coords = get_coords_from_str(args.coordinates)

# diameter is 4 pixels for calculating S/N
fwhm_for_snr= args.fwhm

# --r_apperture
r_aperture = args.r_aperture
r_in_annulus = args.r_in_annulus
r_out_annulus = args.r_out_annulus

###############################
# Step-1 Reading the sof file #
###############################
# Read Data from file .sof
data=np.loadtxt(sofname,dtype=str)
filenames=data[:,0]
datatypes=data[:,1]
cube_names = filenames[np.where(datatypes == "IRD_RDI_RES_FAKE_INJECTION")[0]]
nb_cubes = len(cube_names)
print("> we have", nb_cubes, "inputs")
print("> input name =", cube_names)

# result
res_final = {}

for i in cube_names:
    # get res of rdi
    fake = fits.getdata(i)
    fake_header = fits.getheader(i)
    wl = fake_header["wave_length"]
    real = fits.getdata(get_real_res_path_from_fake_res_path(i, wl))

    # get fwhm_flux from header
    if wl == 0:
        fwhm_flux = fake_header["HIERARCH fwhm_flux_0"]
    else:    
        fwhm_flux = fake_header["HIERARCH fwhm_flux_1"]

    # calculating contrast, S/N and flux
    obj = fake_header["OBJECT"]
    for pos in coords:
        contrast, sn, flux = get_contrast_and_SN(fake, real, pos, fwhm_for_snr, fwhm_flux, r_aperture, r_in_annulus, r_out_annulus)
        res_final.update({obj+'_wl='+str(wl)+'_'+str(pos):{'contrast':contrast, 'sn':sn, 'flux':flux}})

df = pd.DataFrame(data=res_final)
df.to_csv(r'ird_rdi_fake_injeciton_contrast_sn_flux.txt', sep=' ', mode='a')

end_time = datetime.datetime.now()
print("######### End program : no error! Take:", end_time - start_time, "#########")