"""
Process that reduces a IRD_SCIENCE_REDUCED_MASTER_CUBE using RDI as implemented 
in the vip_hci pipeline. 


@author: Julien Milli
Creation date: 2021-07-07
Modification history:    
"""

# import sys
import argparse
from astropy.io import fits
import numpy as np
import vip_hci as vip
import warnings
# warnings.filterwarnings('ignore', category=UserWarning, append=True)
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)

parser = argparse.ArgumentParser()
parser.add_argument("sof", help="file name of the sof file",type=str)
parser.add_argument("--ncomp",help="number of principal components to remove (5 by default)",type=int,default=5)
parser.add_argument("--mask_center_px",help="inner radius where the reduction starts",type=int,default=10)
parser.add_argument("--crop_size",help="size of the output image (201 by default, safer to use an odd value, and smaller than 1024 in any case)",type=int,default=201)
parser.add_argument("--scaling", help="scaling for the PCA (to choose between 0 for spat-mean, 1 for spat-standard, 2 for temp-mean, 3 for temp-standard or 4 for None)",\
                    type=int, choices=[0,1,2,3,4],default=0)
parser.add_argument("--science_object", help="the OBJECT keyword of the science target",\
                    type=str,default='unspecified')
parser.add_argument("--wl_channels", help="Spectral channel to use (to choose between 0 for channel 0, 1 for channel 1, 2 for both channels)",\
                    type=int, choices=[0,1,2],default=0)

args = parser.parse_args()

sofname=args.sof

ncomp =  args.ncomp

mask_center_px = args.mask_center_px

scaling = args.scaling
dico_conversion_scaling = {0 : 'spat-mean', 1 : 'spat-standard', 2 : 'temp-mean', 3 : 'temp-standard', 4 : None}
scaling = dico_conversion_scaling[scaling]

science_object = args.science_object

crop_size = args.crop_size # has to be an odd number for vip to work fine
if type(crop_size) not in [np.int64,np.int,int]:
    crop_size = int(crop_size)
if crop_size>1024:
    crop_size = 1024
elif crop_size<=21:
    crop_size=21
    print('Warning cropsize<=21 ! Value set to {0:d}'.format(crop_size))

wl_channels = args.wl_channels
dico_conversion_wl_channels = {0 : [0], 1 : [1], 2 : [0,1]}
wl_channels = dico_conversion_wl_channels[wl_channels]
nb_wl = len(wl_channels)

print('sof:',sofname)
print('ncomp:',ncomp)
print('mask_center_px:',mask_center_px)
print('scaling:',scaling)
print('science_object:',science_object)
print('crop_size:',crop_size)
print('wl_channels:',wl_channels)

# if scaling not in ['spat-mean','spat-standard','temp-mean','temp-standard','None']:
#     raise Exception('The scaling should be either spat-mean, spat-standard, temp-mean, temp-standard or None')
# elif scaling == 'None':
#     scaling = None    

# Reading the sof file
data=np.loadtxt(sofname,dtype=str)
filenames=data[:,0]
datatypes=data[:,1]

cube_names = filenames[np.where(datatypes == 'IRD_SCIENCE_REDUCED_MASTER_CUBE')[0]]
nb_cubes = len(cube_names)
print('\nIRD_SCIENCE_REDUCED_MASTER_CUBE files:')
print('\n'.join(cube_names))
if nb_cubes < 2: 
    raise Exception('The sof file must contain at least 2 IRD_SCIENCE_REDUCED_MASTER_CUBE (science and reference)')

anglenames = filenames[np.where(datatypes == 'IRD_SCIENCE_PARA_ROTATION_CUBE')[0]]
if len(anglenames) != 1: 
    raise Exception('The sof file must contain exactly one IRD_SCIENCE_PARA_ROTATION_CUBE file')

nb_reference_cubes = nb_cubes - 1 # 1 science cube, the rest are reference cubes

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
         
science_cube = fits.getdata(science_cube_name)
science_header = fits.getheader(science_cube_name)
nb_wl_channelss,nb_science_frames,ny,nx = science_cube.shape
derotation_angles = fits.getdata(anglenames[0])
if len(derotation_angles) != nb_science_frames:
    raise Exception('The science cube IRD_SCIENCE_REDUCED_MASTER_CUBE contains {0:d} frames while the list IRD_SCIENCE_PARA_ROTATION_CUBE contains {1:d} angles'.format(nb_science_frames,len(derotation_angles)))

reference_objects  = []
number_frames_reference_cubes = np.ndarray((nb_reference_cubes),dtype=int)
print('The following cubes will be used as reference cubes:')
for i,reference_cube_name in enumerate(reference_cube_names):
    header_tmp = fits.getheader(reference_cube_name)
    number_frames_reference_cubes[i] = header_tmp['NAXIS3']
    reference_objects.append(header_tmp['OBJECT'])
    print('   - {0:s}: {1:d} frames'.format(reference_objects[i],number_frames_reference_cubes[i]))
nb_reference_frames = np.sum(number_frames_reference_cubes)


reduced_image = np.ndarray((nb_wl,crop_size,crop_size))

for i_wl,wl_channel in enumerate(wl_channels):    
    reference_cube = np.ndarray((nb_reference_frames,crop_size,crop_size),dtype=float)
    counter = 0
    for i,reference_cube_name in enumerate(reference_cube_names):
            cube_tmp = fits.getdata(reference_cube_name)
            nb_frames_tmp = cube_tmp.shape[1]
            reference_cube[counter:counter+nb_frames_tmp,:,:] = cube_tmp[wl_channel,:,ny//2-crop_size//2:ny//2+crop_size//2+1,nx//2-crop_size//2:nx//2+crop_size//2+1]
            counter = counter+nb_frames_tmp    
    science_cube_croped = science_cube[wl_channel,:,ny//2-crop_size//2:ny//2+crop_size//2+1,nx//2-crop_size//2:nx//2+crop_size//2+1]
    
    reduced_image[i_wl,:,:] = vip.pca.pca_fullfr.pca(science_cube_croped, -derotation_angles, ncomp=ncomp, \
                                           mask_center_px=mask_center_px, cube_ref=reference_cube, \
                                           scaling=scaling)


# *outer_mask

    
# # We update the header
science_header['IRD PIP ALGO'] = ('RDI', 'Type of reduction')
science_header['IRD PIP REF FRAMES'] = (nb_reference_frames, 'Number of reference frames')

fits.writeto('rdi_image_median.fits', reduced_image,science_header,overwrite=True,output_verify='ignore')

