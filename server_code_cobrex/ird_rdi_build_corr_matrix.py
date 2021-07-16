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

args = parser.parse_args()
sofname=args.sof
ncomp =  args.ncomp
mask_center_px = args.mask_center_px
scaling = args.scaling
dico_conversion_scaling = {0 : 'spat-mean', 1 : 'spat-standard', 2 : 'temp-mean', 3 : 'temp-standard', 4 : None}
scaling = dico_conversion_scaling[scaling]
science_object = args.science_object



