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

# get contrast and S/N ratio
def get_contrast_and_SN_only_real(res_real, positions, fwhm_for_snr, fwhm_flux, r_aperture, r_in_annulus, r_out_annulus):
    '''
    Args:
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
    flux_companion = aperture_photometry(res_real, [aperture, annulus])
    flux_companion['aperture_sum_0','aperture_sum_1'].info.format = '%.8g'
    flux = flux_companion['aperture_sum_0'] 
    contrast = (flux_companion['aperture_sum_0'])/fwhm_flux
    
    # SN
    SN = vip.metrics.snr(array=res_real, source_xy=positions, fwhm=fwhm_for_snr, plot=False)

    return contrast.data[0], SN, flux.data[0]

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

#############
# main code #
#############

print("######### Start program : ird_rdi_compute_contrast.py #########")
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
fake_exist = False
if nb_cubes < 1:
    print("> Well, we don't have the fake injection this time")
else:
    fake_exist = True
    
print("> we have fake", nb_cubes, "inputs")
print("> input name =", cube_names)

cube_names_real = filenames[np.where(datatypes == "IRD_RDI_RES_REAL")[0]]
nb_cubes_real = len(cube_names_real)
real_exist = False
if nb_cubes_real < 1:
    raise Exception("> Well, we don't have the real data this time, but the real data is necessary, so end program!")
else:
    real_exist = True

print("> we have real", nb_cubes_real, "inputs")
print("> input name =", cube_names_real)

if nb_cubes != nb_cubes_real and fake_exist:
    raise Exception("Warning: real res and fake res must correspond one-to-one, two input size is different")

# take psf
psf_paths = filenames[np.where(datatypes == 'IRD_SCIENCE_PSF_MASTER_CUBE')[0]]
if (not fake_exist) and len(psf_paths) < 1: 
    raise Exception('The sof file must contain IRD_SCIENCE_PSF_MASTER_CUBE file when we only process the original')

# get psf
print("> Process only science data without fake injection")
psf_path = psf_paths[0]
psf = fits.getdata(psf_path)
psf_header = fits.getheader(psf_path)
print("\n> corresponding psf", psf_path)
print(">> psf DATE-OBS:", psf_header["DATE-OBS"])
print(">> psf OBJECT:", psf_header["OBJECT"])
print("psf.shape =", psf.shape)

# result
res_final = {}

for i in range(len(cube_names_real)):
    if fake_exist:
        print(">> we are processing the data :", cube_names[i])
    print(">> we are processing the data :", cube_names_real[i])
    # get res of rdi
    if fake_exist:
        fake = fits.getdata(cube_names[i])
        fake_header = fits.getheader(cube_names[i])
        wl = fake_header["WL_CHOSE"]
    real = fits.getdata(cube_names_real[i])
    real_header = fits.getheader(cube_names_real[i])

    # check
    if fake_exist:
        if real_header["OBJECT"] != fake_header["OBJECT"]:
            raise Exception(">>> Traget different!")
        if real_header["WL_CHOSE"] != fake_header["WL_CHOSE"]:
            raise Exception(">>> Wave_length different!")

    # get fwhm_flux from header
    if fake_exist:
        fwhm_flux = fake_header["FWHM_F"]
    else:
        wl = real_header["WL_CHOSE"]
        
    # calculating contrast, S/N and flux
    obj = real_header["OBJECT"]
    pct = [int(e) for e in real_header["D_PCT"].split(' ')] 
    n_corr = [int(e) for e in real_header["D_N_COR"].split(' ')]
    ncomp = [int(e) for e in real_header["D_NCOMP"].split(' ')]

    for i in range(len(pct)):
        for j in range(len(n_corr)):
            res_final = {}
            for k in range(len(ncomp)):
                contrast = 0
                sn = 0
                flux = 0
                # the average of contrast, sn, flux from position
                for pos in coords:
                    if fake_exist:
                        ct_tmp, sn_tmp, flux_tmp = get_contrast_and_SN(fake[i,j,k], real[i,j,k], pos, fwhm_for_snr, fwhm_flux, r_aperture, r_in_annulus, r_out_annulus)
                    else:
                        aperture = CircularAperture(pos, r=(fwhm_for_snr/2))
                        annulus = CircularAnnulus(pos, r_in=fwhm_for_snr, r_out=fwhm_for_snr*(3/2))
                        flux_psf = aperture_photometry(psf[wl], [aperture, annulus])
                        flux_psf['aperture_sum_0','aperture_sum_1'].info.format = '%.8g'
                        fwhm_flux = flux_psf['aperture_sum_0'][0]
                        ct_tmp, sn_tmp, flux_tmp = get_contrast_and_SN_only_real(real[i,j,k], pos, fwhm_for_snr, fwhm_flux, r_aperture, r_in_annulus, r_out_annulus)
                    
                    contrast = contrast + ct_tmp
                    sn = sn + sn_tmp
                    flux = flux + flux_tmp

                contrast = contrast/len(pos)
                sn = sn/len(pos)
                flux = flux/len(pos)
                print(">>> object =", obj, "pos =", pos, "contrast =", contrast, "sn =", sn, "flux =", flux)
                res_final.update({str(ncomp[k]):{'ctr':contrast, 'sn':sn, 'flux':flux}})
            # write data to file
            df = pd.DataFrame(data=res_final)
            df.columns = pd.MultiIndex.from_product([["pct_"+str(pct[i])+"_ncorr_"+str(n_corr[j])], df.columns])
            df.to_csv(r'ird_rdi_fake_injeciton_contrast_sn_flux.csv', sep='\t', mode='a', encoding='utf-8', na_rep='NaN', float_format='%8.8f')

    for i in range(len(pct)):
        for k in range(len(ncomp)):
            res_final = {}
            for j in range(len(n_corr)):    
                contrast = 0
                sn = 0
                flux = 0
                # the average of contrast, sn, flux from position
                for pos in coords:
                    if fake_exist:
                        ct_tmp, sn_tmp, flux_tmp = get_contrast_and_SN(fake[i,j,k], real[i,j,k], pos, fwhm_for_snr, fwhm_flux, r_aperture, r_in_annulus, r_out_annulus)
                    else:
                        aperture = CircularAperture(pos, r=(fwhm_for_snr/2))
                        annulus = CircularAnnulus(pos, r_in=fwhm_for_snr, r_out=fwhm_for_snr*(3/2))
                        flux_psf = aperture_photometry(psf[wl], [aperture, annulus])
                        flux_psf['aperture_sum_0','aperture_sum_1'].info.format = '%.8g'
                        fwhm_flux = flux_psf['aperture_sum_0'][0]
                        ct_tmp, sn_tmp, flux_tmp = get_contrast_and_SN_only_real(real[i,j,k], pos, fwhm_for_snr, fwhm_flux, r_aperture, r_in_annulus, r_out_annulus)
                    
                    contrast = contrast + ct_tmp
                    sn = sn + sn_tmp
                    flux = flux + flux_tmp

                contrast = contrast/len(pos)
                sn = sn/len(pos)
                flux = flux/len(pos)
                print(">>> object =", obj, "pos =", pos, "contrast =", contrast, "sn =", sn, "flux =", flux)
                res_final.update({str(n_corr[j]):{'ctr':contrast, 'sn':sn, 'flux':flux}})
            # write data to file
            df = pd.DataFrame(data=res_final)
            df.columns = pd.MultiIndex.from_product([["pct_"+str(pct[i])+"_ncomp_"+str(ncomp[k])], df.columns])
            df.to_csv(r'ird_rdi_fake_injeciton_contrast_sn_flux.csv', sep='\t', mode='a', encoding='utf-8', na_rep='NaN', float_format='%8.8f')    

    
    for j in range(len(n_corr)):
        for k in range(len(ncomp)):
            res_final = {}
            for i in range(len(pct)):
                contrast = 0
                sn = 0
                flux = 0
                # the average of contrast, sn, flux from position
                for pos in coords:
                    if fake_exist:
                        ct_tmp, sn_tmp, flux_tmp = get_contrast_and_SN(fake[i,j,k], real[i,j,k], pos, fwhm_for_snr, fwhm_flux, r_aperture, r_in_annulus, r_out_annulus)
                    else:
                        aperture = CircularAperture(pos, r=(fwhm_for_snr/2))
                        annulus = CircularAnnulus(pos, r_in=fwhm_for_snr, r_out=fwhm_for_snr*(3/2))
                        flux_psf = aperture_photometry(psf[wl], [aperture, annulus])
                        flux_psf['aperture_sum_0','aperture_sum_1'].info.format = '%.8g'
                        fwhm_flux = flux_psf['aperture_sum_0'][0]
                        ct_tmp, sn_tmp, flux_tmp = get_contrast_and_SN_only_real(real[i,j,k], pos, fwhm_for_snr, fwhm_flux, r_aperture, r_in_annulus, r_out_annulus)

                    contrast = contrast + ct_tmp
                    sn = sn + sn_tmp
                    flux = flux + flux_tmp

                contrast = contrast/len(pos)
                sn = sn/len(pos)
                flux = flux/len(pos)
                print(">>> object =", obj, "pos =", pos, "contrast =", contrast, "sn =", sn, "flux =", flux)
                res_final.update({str(pct[i]):{'ctr':contrast, 'sn':sn, 'flux':flux}})
            
            # write data to file
            df = pd.DataFrame(data=res_final)
            df.columns = pd.MultiIndex.from_product([["ncorr_"+str(n_corr[j])+"_ncomp_"+str(ncomp[k])], df.columns])
            df.to_csv(r'ird_rdi_fake_injeciton_contrast_sn_flux.csv', sep='\t', mode='a', encoding='utf-8', na_rep='NaN', float_format='%8.8f')

end_time = datetime.datetime.now()
print("######### End program : no error! Take:", end_time - start_time, "#########")