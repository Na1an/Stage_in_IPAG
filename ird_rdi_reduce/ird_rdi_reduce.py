"""
Reduce a IRD_SCIENCE_REDUCED_MASTER_CUBE

@Author : Yuchen BAI
@Date   : 17/07/2021
@Contact: yuchenbai@hotmail.com
"""

import argparse
import warnings
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

#############
# main code #
#############
parser = argparse.ArgumentParser(description="For build the Pearson Correlation Coefficient matrix for the science target and the reference master cubes, we need the following parameters.")
parser.add_argument("sof", help="file name of the sof file",type=str)
parser.add_argument("--mask_center_px",help="inner radius where the reduction starts", type=int, default=10)
parser.add_argument("--crop_size", help="size of the output image (201 by default, safer to use an odd value, and smaller than 1024 in any case)", type=int, default=201)
parser.add_argument("--science_object", help="the OBJECT keyword of the science target", type=str, default='unspecified')
parser.add_argument("--wl_channels", help="Spectral channel to use (to choose between 0 for channel 0, 1 for channel 1, 2 for both channels)", type=int, choices=[0,1,2], default=0)

# handle args
args = parser.parse_args()

# sof
sofname=args.sof

# --crop_size and --mask_center_px
crop_size = args.crop_size
mask_center_px = args.mask_center_px

# --science_object
science_object = args.science_object

# --wl_channels
dico_conversion_wl_channels = {0 : [0], 1 : [1], 2 : [0,1]}
wl_channels = dico_conversion_wl_channels[args.wl_channels]
nb_wl = len(wl_channels)

# start the program
if type(crop_size) not in [np.int64,np.int,int]:
    crop_size = int(crop_size)
if crop_size>1024:
    crop_size = 1024
elif crop_size<=21:
    crop_size=21
    print('Warning cropsize<=21 ! Value set to {0:d}'.format(crop_size))

# Reading the sof file
data=np.loadtxt(sofname,dtype=str)
filenames=data[:,0]
datatypes=data[:,1]

cube_names = filenames[np.where(datatypes == 'IRD_SCIENCE_REDUCED_MASTER_CUBE')[0]]
nb_cubes = len(cube_names)

if nb_cubes < 2: 
    raise Exception('The sof file must contain at least 2 IRD_SCIENCE_REDUCED_MASTER_CUBE (science and reference)')

'''
anglenames = filenames[np.where(datatypes == 'IRD_SCIENCE_PARA_ROTATION_CUBE')[0]]
if len(anglenames) != 1: 
    raise Exception('The sof file must contain exactly one IRD_SCIENCE_PARA_ROTATION_CUBE file')
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
print("> reference cuebs :", reference_cube_names)

# take science cube
science_cube = fits.getdata(science_cube_name)
science_header = fits.getheader(science_cube_name)
nb_wl_channelss, nb_science_frames, ny, nx = science_cube.shape

'''
derotation_angles = fits.getdata(anglenames[0])
if len(derotation_angles) != nb_science_frames:
    raise Exception('The science cube IRD_SCIENCE_REDUCED_MASTER_CUBE contains {0:d} frames while the list IRD_SCIENCE_PARA_ROTATION_CUBE contains {1:d} angles'.format(nb_science_frames,len(derotation_angles)))
'''
# sort reference cube names
reference_cube_names.sort()
tmp_cube = fits.getdata(reference_cube_names[0])

# collect data, then we have reference frames
border_l = ny//2 - crop_size//2
border_r = ny//2 + crop_size//2 + 1
ref_frames = tmp_cube[..., border_l:border_r, border_l:border_r]
ref_nb_frames = []
ref_nb_frames.append(len(tmp_cube[0]))

for name in reference_cube_names[1:]:
    tmp_cube = fits.getdata(name)
    ref_nb_frames.append(len(tmp_cube[0]))
    ref_frames = np.append(ref_frames, tmp_cube[..., border_l:border_r, border_l:border_r], axis=1)

print("> ref_frames.shape =", ref_frames.shape)
print("> ref_nb_frames =", ref_nb_frames)
wl_ref, nb_ref_frames, ref_x, ref_y = ref_frames.shape

# correlation matrix
res = np.zeros((nb_wl, nb_science_frames, nb_ref_frames))
science_cube_croped = science_cube[..., border_l:border_r, border_l:border_r]
for w in range(nb_wl):
    wl = wl_channels[w]
    for i in range(nb_science_frames):
        for j in range(nb_ref_frames):
            res[wl, i, j] = np.corrcoef(np.reshape(science_cube_croped[wl, i], ref_x*ref_y), np.reshape(ref_frames[wl, j], ref_x*ref_y))[0,1]

file_name = "pcc_matrix.fits"
print("> The result will be stored in :", file_name)
complete_header(science_header, reference_cube_names, ref_nb_frames)
hdu = fits.PrimaryHDU(data=res, header=science_header)
hdu.writeto(file_name)