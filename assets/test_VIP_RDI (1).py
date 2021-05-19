#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 10:35:48 2017

@author: jmilli
"""

from astropy.io import fits
import radial_data as rd
import sys,os
import numpy as np
import matplotlib.pyplot as plt
import vip_hci as vip
from pathlib import Path

ds9=vip.Ds9Window()

path_root=Path('/Users/millij/Documents/RDI/stage_yuchen')

path_input = path_root.joinpath('input_data')
path_output = path_root.joinpath('output_data')

cube_science = fits.getdata(path_input.joinpath(\
    'HD 156384Cf_DB_H23_2017-06-27_ird_convert_recenter_dc5_PUBLIC_208368/ird_convert_recenter_dc5-IRD_SCIENCE_REDUCED_MASTER_CUBE-center_im.fits'))

size=1024
crop_size = 256
cube_science_channel1_cropped = cube_science[0,:,size//2-crop_size//2:size//2+crop_size//2,size//2-crop_size//2:size//2+crop_size//2]
cube_science_channel1_cropped.shape

ds9.display(cube_science_channel1_cropped)
parang_science = fits.getdata(path_input.joinpath('HD 156384Cf_DB_H23_2017-06-27_ird_convert_recenter_dc5_PUBLIC_208368/ird_convert_recenter_dc5-IRD_SCIENCE_PARA_ROTATION_CUBE-rotnth.fits'))
    
pca_adi_10 = vip.pca.pca(cube_science_channel1_cropped,-parang_science,ncomp=10,mask_center_px=32)

ds9.display(pca_adi_10)

# vip.pca.pca(
#     cube,
#     angle_list,
#     cube_ref=None,
#     scale_list=None,
#     ncomp=1,
#     svd_mode='lapack',
#     scaling=None,
#     mask_center_px=None,
#     source_xy=None,
#     delta_rot=1,
#     fwhm=4,
#     adimsdi='single',
#     crop_ifs=True,
#     imlib='opencv',
#     interpolation='lanczos4',
#     collapse='median',
#     ifs_collapse_range='all',
#     check_memory=True,
#     batch=None,
#     nproc=1,
#     full_output=False,
#     verbose=True,
# )

# reference

cube_ref1 = fits.getdata(path_input.joinpath(\
    'HIP 86214_DB_H23_2017-06-23_ird_convert_recenter_dc5_PUBLIC_208373/ird_convert_recenter_dc5-IRD_SCIENCE_REDUCED_MASTER_CUBE-center_im.fits'))
cube_ref2 = fits.getdata(path_input.joinpath(\
    'BD+11 3149c_DB_H23_2017-05-14_ird_convert_recenter_dc5_PUBLIC_176255/ird_convert_recenter_dc5-IRD_SCIENCE_REDUCED_MASTER_CUBE-center_im.fits'))
cube_ref3 = fits.getdata(path_input.joinpath(\
    'HIP 85523_DB_H23_2017-07-14_ird_convert_recenter_dc5_PUBLIC_210792/ird_convert_recenter_dc5-IRD_SCIENCE_REDUCED_MASTER_CUBE-center_im.fits'))

cube_ref1_channel1_cropped = cube_ref1[0,:,size//2-crop_size//2:size//2+crop_size//2,size//2-crop_size//2:size//2+crop_size//2]
cube_ref2_channel1_cropped = cube_ref2[0,:,size//2-crop_size//2:size//2+crop_size//2,size//2-crop_size//2:size//2+crop_size//2]
cube_ref3_channel1_cropped = cube_ref3[0,:,size//2-crop_size//2:size//2+crop_size//2,size//2-crop_size//2:size//2+crop_size//2]
nframes_ref1 = cube_ref1_channel1_cropped.shape[0]
nframes_ref2 = cube_ref2_channel1_cropped.shape[0]
nframes_ref3 = cube_ref3_channel1_cropped.shape[0]

ref_library = np.concatenate([cube_ref1_channel1_cropped,cube_ref2_channel1_cropped,cube_ref2_channel1_cropped])

# ref_library2 = np.concatenate([cube_science_channel1_cropped,cube_ref1_channel1_cropped,cube_ref2_channel1_cropped,cube_ref2_channel1_cropped])


pca_rdi_10 = vip.pca.pca(cube_science_channel1_cropped,-parang_science,ncomp=100,mask_center_px=32,\
                       cube_ref=ref_library,scaling='spat-mean')
# scaling : {None, "temp-mean", spat-mean", "temp-standard",
#         "spat-standard"}

ds9.display(pca_adi_10,pca_rdi_10)

pca_rdi_10_yuchen = fits.getdata(path_root.joinpath('Stage_in_IPAG/K_kilp_ADI_RDI/RDI_WITH_MASK_3_best_32/RDI_Masked10.fits'))


ds9.display(pca_rdi_10,pca_rdi_10_yuchen)
# pd_contrast_curve = vip.metrics.contrast_curve(cube_science, parang, psf_template, fwhm, pxscale, starphot, algo, algo_dict)