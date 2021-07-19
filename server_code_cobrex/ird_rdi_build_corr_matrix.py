"""
Reduce a IRD_SCIENCE_REDUCED_MASTER_CUBE

@Author : Yuchen BAI
@Date   : 17/07/2021
@Contact: yuchenbai@hotmail.com
"""

import argparse
import numpy as np
import vip_hci as vip
import warnings
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)

parser = argparse.ArgumentParser(description="For build the Pearson Correlation Coefficient matrix for the science target and the reference master cubes, we need the following parameters.")
parser.add_argument("sof", help="file name of the sof file",type=str)
parser.add_argument("--ncomp",help="number of principal components to remove (5 by default)", type=int, default=5)
parser.add_argument("--mask_center_px",help="inner radius where the reduction starts", type=int, default=10)
parser.add_argument("--crop_size", help="size of the output image (201 by default, safer to use an odd value, and smaller than 1024 in any case)", type=int, default=201)
parser.add_argument("--scaling", help="scaling for the PCA (to choose between 0 for spat-mean, 1 for spat-standard, 2 for temp-mean, 3 for temp-standard or 4 for None)", type=int, choices=[0,1,2,3,4], default=0)
parser.add_argument("--science_object", help="the OBJECT keyword of the science target", type=str, default='unspecified')
parser.add_argument("--wl_channels", help="Spectral channel to use (to choose between 0 for channel 0, 1 for channel 1, 2 for both channels)", type=int, choices=[0,1,2], default=0)

# handle args
args = parser.parse_args()

# sof
sofname=args.sof

# --ncomp
ncomp = args.ncomp

# --crop_size and --mask_center_px
crop_size = args.crop_size
mask_center_px = args.mask_center_px

# --scaling
all_scalings = {0 : 'spat-mean', 1 : 'spat-standard', 2 : 'temp-mean', 3 : 'temp-standard', 4 : None}
scaling = all_scalings[args.scaling]

# --science_object
science_object = args.science_object

# --wl_channels
dico_conversion_wl_channels = {0 : [0], 1 : [1], 2 : [0,1]}
wl_channels = dico_conversion_wl_channels[args.wl_channels]
nb_wl = len(wl_channels)

'''
print(type(args))
print("sofname =", sofname)
print("ncomp =", ncomp)
print("mask center =", mask_center_px)
print("sclaing =", scaling)
print("science_obj =", science_object)
print("wl =", wl)
'''

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
print('\nIRD_SCIENCE_REDUCED_MASTER_CUBE files:\n'.join(cube_names))

if nb_cubes < 2: 
    raise Exception('The sof file must contain at least 2 IRD_SCIENCE_REDUCED_MASTER_CUBE (science and reference)')

anglenames = filenames[np.where(datatypes == 'IRD_SCIENCE_PARA_ROTATION_CUBE')[0]]
if len(anglenames) != 1: 
    raise Exception('The sof file must contain exactly one IRD_SCIENCE_PARA_ROTATION_CUBE file')

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

# take science cube
science_cube = fits.getdata(science_cube_name)
science_header = fits.getheader(science_cube_name)
nb_wl_channelss,nb_science_frames,ny,nx = science_cube.shape
derotation_angles = fits.getdata(anglenames[0])
if len(derotation_angles) != nb_science_frames:
    raise Exception('The science cube IRD_SCIENCE_REDUCED_MASTER_CUBE contains {0:d} frames while the list IRD_SCIENCE_PARA_ROTATION_CUBE contains {1:d} angles'.format(nb_science_frames,len(derotation_angles)))

# sort reference cube names
reference_cube_names.sort()
tmp_cube = fits.getdata(reference_cube_names[0])

# collect data, then we have reference frames
border_l = ny//2-crop_size//2
border_r = ny//2+crop_size//2+1
ref_frames = tmp_cube[...,border_l:border_r, border_l:border_r]
ref_nb_frames = []
ref_nb_frames.append(len(tmp_cube[1]))
for name in reference_cube_names[1:]:
    tmp_cube = fits.getdata(name)
    ref_frames = np.append(ref_frames, tmp_cube[..., border_l:border_r, border_l:border_r])

print("ref_frames.shape =", ref_frames.shape)
wl_ref,nb_ref_frames, ref_x, ref_y = ref_frames.shape

# correlation matrix
res = np.zeros((nb_wl,nb_science_frames, nb_ref_frames))
for w in range(nb_wl):
    for i in range(nb_science_frames):
        for j in range(nb_ref_frames):
            res[w, i, j] = np.corrcoef(np.reshape(science_cube[wl_channels[w], i], ny*nx), np.reshape(ref_frames[wl_channels[w], j], ny*nx))[0,1]
    file_name = "correlation_matrix_"+ science_cube_name +"_"+str(wl_channels[w])+".csv"
    np.savetext(file_name, res[wl_channels[w]], delimiter=",")
