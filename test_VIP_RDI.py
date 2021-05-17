#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 10:35:48 2017

@author: jmilli
"""

from astropy.io import fits
#import radial_data as rd
import sys,os
import numpy as np
import matplotlib.pyplot as plt
import vip_hci as vip
from pathlib import Path

ds9 = vip.vip_ds9()

path_root=Path('/Users/jmilli/Documents/RDI/')
path_result = path_root.joinpath('result')

cube_science = fits.getdata(path_root.joinpath('hip21986A_fc.fits'))
cube_ref = fits.getdata(path_root.joinpath('hip21986B_nfc.fits'))

parang = np.loadtxt(path_root.joinpath('hip21986A_fc_angs.txt'))


pd_contrast_curve = vip.metrics.contrast_curve(cube_science, parang, psf_template, fwhm, pxscale, starphot, algo, algo_dict)
