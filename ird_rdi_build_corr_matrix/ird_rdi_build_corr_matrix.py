"""
Reduce a IRD_SCIENCE_REDUCED_MASTER_CUBE

@Author : Yuchen BAI
@Date   : 17/07/2021
@Contact: yuchenbai@hotmail.com
"""

import argparse
import warnings
import datetime
import numpy as np
import vip_hci as vip
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning

###########
# setting #
###########

warnings.simplefilter('ignore', category=AstropyWarning)

############
# function #
############

# add information into the header
def complete_header(science_header, reference_cube_names, ref_nb_frames):
    '''
    This function is for adding some additional information on science header.
    Useful for the next stage.  
    Args:
        science_header : a fits.header. The science header.
        reference_cube_names : a list of reference cube names.
        ref_nb_frames : a list of number of cubes.
    Return:
        None.   
    '''
    nb_ref_cube = len(reference_cube_names)
    science_header["NB_REF_CUBES"] = nb_ref_cube
    ind = 0
    for i in range(nb_ref_cube):
        nb_str = "{0:06d}".format(i)
        science_header["RN"+nb_str] = reference_cube_names[i]
        science_header["RF"+nb_str] = ref_nb_frames[i]
        science_header["RS"+nb_str] = ind
        ind = ind + ref_nb_frames[i]
 
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

# display header
def display_header(header):
    '''
    Args:
        header : a fits header. Juest for displaying the detail.
    Return:
        None.
    '''
    print(str(header["OBJECT"])+"\t\t\t"+str(header["DATE-OBS"])+'\t'+ str(header["ESO OBS START"]) +'\t'+str(header["NAXIS3"])+"\t  "+str(header["DIT_MIN"]))

# get angle from the science cube path
def get_para_angle_from_science_cube(path):
    '''
    Args:
        path : a string. Replace the last element after '/', then we can have the parallactic angle path.
    Return:
        res : a string. The parallactic angle path.
    '''
    return path.replace("ird_convert_recenter_dc5-IRD_SCIENCE_REDUCED_MASTER_CUBE-center_im.fits","ird_convert_recenter_dc5-IRD_SCIENCE_PARA_ROTATION_CUBE-rotnth.fits")

#############
# main code #
#############
print("######### Start program : ird_rdi_corr_matrix.py #########")
start_time = datetime.datetime.now()
parser = argparse.ArgumentParser(description="For build the Pearson Correlation Coefficient matrix for the science target and the reference master cubes, we need the following parameters.")
parser.add_argument("sof", help="file name of the sof file",type=str)
parser.add_argument("--inner_radius",help="inner radius where the reduction starts", type=int, default=10)
parser.add_argument("--outer_radius",help="outer radius where the reduction starts", type=int, default=100)
parser.add_argument("--science_object", help="the OBJECT keyword of the science target", type=str, default='unspecified')
parser.add_argument("--wl_channels", help="Spectral channel to use (to choose between 0 for channel 0, 1 for channel 1, 2 for both channels)", type=int, choices=[0,1,2], default=0)

# handle args
args = parser.parse_args()

# sof
sofname=args.sof

# --crop_size and inner/outer radius
inner_radius = args.inner_radius
outer_radius = args.outer_radius
crop_size = 2*outer_radius+1

# --science_object
science_object = args.science_object

# --wl_channels
dico_conversion_wl_channels = {0 : [0], 1 : [1], 2 : [0,1]}
wl_channels = dico_conversion_wl_channels[args.wl_channels]
nb_wl = len(wl_channels)

# start the program
if type(crop_size) not in [np.int64,np.int,int]:
    crop_size = int(crop_size)

if crop_size<=21:
    crop_size=21
    print('Warning cropsize<=21, too small! Value set to 21')

if outer_radius <= inner_radius:
    print("Warning outer_radius <= inner_radius! Value set to {0:d}".format(inner_radius+1))

# Reading the sof file
data=np.loadtxt(sofname, dtype=str)
filenames=data[:,0]
datatypes=data[:,1]

# Read Data from file .sof
cube_names = filenames[np.where(datatypes == 'IRD_SCIENCE_REDUCED_MASTER_CUBE')[0]]
nb_cubes = len(cube_names)

if nb_cubes < 2: 
    raise Exception('The sof file must contain at least 2 IRD_SCIENCE_REDUCED_MASTER_CUBE (science and reference)')

'''
# Read angle
anglenames = filenames[np.where(datatypes == 'IRD_SCIENCE_PARA_ROTATION_CUBE')[0]]
if len(anglenames) < 1: 
    raise Exception('The sof file must contain more than one IRD_SCIENCE_PARA_ROTATION_CUBE file')
'''

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
print("> science cube :", science_cube_name)


# take science cube
science_cube = fits.getdata(science_cube_name)
science_header = fits.getheader(science_cube_name)
print(">> science cube - info ")
print("OBJECT\t\t\tDATE-OBS\t\t\tOBS_STA\t\t\tNB_FRAMES\tDIT")
display_header(science_header)
print("=================== science cube and angle =======================")
print("> start test")
print(">> science cube DATE-OBS:", science_header["DATE-OBS"])
print(">> science cube OBJECT:", science_header["OBJECT"])
print(">> science cube EXPTIME:", science_header["EXPTIME"])
print(">> science cube ESO INS COMB ICOR:", science_header["ESO INS COMB ICOR"])
print(">> science cube ESO INS COMB IFLT:", science_header["ESO INS COMB IFLT"])

# for test
anglename = get_para_angle_from_science_cube(science_cube_name)

derotation_angles_header = fits.getheader(anglename)
print("> corresponding parallactic angle", anglename)
print(">> para DATE-OBS:", derotation_angles_header["DATE-OBS"])
print(">> para OBJECT:", derotation_angles_header["OBJECT"])
print(">> para EXPTIME:", derotation_angles_header["EXPTIME"])
print(">> para ESO INS COMB ICOR:", derotation_angles_header["ESO INS COMB ICOR"])
print(">> para ESO INS COMB IFLT:", derotation_angles_header["ESO INS COMB IFLT"])
nb_wl_channelss, nb_science_frames, ny, nx = science_cube.shape

# sort reference cube names
reference_cube_names.sort()

# collect data, then we have reference frames
border_l = ny//2 - crop_size//2
border_r = ny//2 + crop_size//2 + 1
ref_frames = None
ref_nb_frames = []
reference_cube_names_remove_dup = []

# indice start
ind_start = 0
print(">> reference cube - info ")
print("OBJECT\t\t\tDATE-OBS\t\t\tOBS_STA\t\t\tNB_FRAMES\tDIT")

for i in range(len(reference_cube_names)):
    name = reference_cube_names[i]
    tmp_cube = fits.getdata(name)
    tmp_header = fits.getheader(name)
    if tmp_header["OBJECT"] == science_header["OBJECT"]:
        ind_start = ind_start+1
        continue
    display_header(tmp_header)
    if i==ind_start:
        ref_frames = tmp_cube[..., border_l:border_r, border_l:border_r]
        ref_nb_frames.append(len(tmp_cube[0]))
    else:
        ref_nb_frames.append(len(tmp_cube[0]))
        ref_frames = np.append(ref_frames, tmp_cube[..., border_l:border_r, border_l:border_r], axis=1)
    reference_cube_names_remove_dup.append(name)

print("> ref_frames.shape =", ref_frames.shape)
wl_ref, nb_ref_frames, ref_x, ref_y = ref_frames.shape

# correlation matrix
mask = create_mask(crop_size, inner_radius, outer_radius)
res = np.zeros((nb_wl, nb_science_frames, nb_ref_frames))
science_cube_croped = science_cube[..., border_l:border_r, border_l:border_r]
for w in range(nb_wl):
    wl = wl_channels[w]
    for i in range(nb_science_frames):
        for j in range(nb_ref_frames):
            res[wl, i, j] = np.corrcoef(np.reshape(science_cube_croped[wl, i]*mask, ref_x*ref_y), np.reshape(ref_frames[wl, j]*mask, ref_x*ref_y))[0,1]

# compelte header
science_header["PATH_TAR"] = science_cube_name
science_header["CROPSIZE"] = crop_size
science_header["INNER_R"] = inner_radius
science_header["OUTER_R"] = outer_radius
science_header["PA_ANGLE"] = anglename

complete_header(science_header, reference_cube_names_remove_dup, ref_nb_frames)

file_name = "pcc_matrix.fits"
print("> The result will be stored in :", file_name)

hdu = fits.PrimaryHDU(data=res, header=science_header)

hdu.writeto(file_name)
end_time = datetime.datetime.now()
print("######### End program : no error! Take:", end_time - start_time, "#########")