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
        tmp = c.replace('(','').replace(')','').split(',')
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
    contrast = (flux_companion['aperture_sum_0'])/fwhm_flux
    
    # SN
    SN = vip.metrics.snr(array=res_fake, source_xy=positions, fwhm=fwhm_for_snr, plot=False, array2 =res_real, use2alone=True)

    return contrast.data[0], SN, flux.data[0]

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
print("> we have fake", nb_cubes, "inputs")
print("> input name =", cube_names)

cube_names_real = filenames[np.where(datatypes == "IRD_RDI_RES_REAL")[0]]
nb_cubes_real = len(cube_names_real)
print("> we have real", nb_cubes_real, "inputs")
print("> input name =", cube_names_real)

if nb_cubes != nb_cubes_real:
    raise Exception("Warning: real res and fake res must correspond one-to-one, two input size is different")

# result
res_final = {}

for i in range(len(cube_names)):
    print(">> we are processing the cube :", cube_names[i])
    # get res of rdi
    fake = fits.getdata(cube_names[i])
    fake_header = fits.getheader(cube_names[i])
    wl = fake_header["WL_CHOSE"]
    real = fits.getdata(cube_names_real[i])
    real_header = fits.getheader(cube_names_real[i])

    # check
    if real_header["OBJECT"] != fake_header["OBJECT"]:
        raise Exception(">>> Traget different!")
    if real_header["WL_CHOSE"] != fake_header["WL_CHOSE"]:
        raise Exception(">>> Wave_length different!")

    # get fwhm_flux from header
    fwhm_flux = fake_header["FWHM_F"]

    # calculating contrast, S/N and flux
    obj = fake_header["OBJECT"]
    for pos in coords:
        contrast, sn, flux = get_contrast_and_SN(fake, real, pos, fwhm_for_snr, fwhm_flux, r_aperture, r_in_annulus, r_out_annulus)
        print(">>> contrast =", contrast, "sn =", sn, "flux =", flux)
        x,y = pos
        res_final.update({obj+'_wl='+str(wl)+'_('+str(x)+','+str(y)+')':{'ctr':contrast, 'sn':sn, 'flux':flux}})

df = pd.DataFrame(data=res_final)
df.to_csv(r'ird_rdi_fake_injeciton_contrast_sn_flux.csv', sep='\t', mode='a', encoding='utf-8', na_rep='NaN', float_format='%8.8f')

end_time = datetime.datetime.now()
print("######### End program : no error! Take:", end_time - start_time, "#########")