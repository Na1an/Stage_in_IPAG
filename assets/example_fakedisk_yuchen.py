#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 14:22:02 2019

@author: jmilli
"""
import numpy as np
import vip_hci as vip
ds9=vip.Ds9Window()


pixel_scale=0.01225 # pixel scale in arcsec/px
dstar= 80 # distance to the star in pc
 # 1pc means 1arcsec represents 1au
 
nx = 200 # number of pixels of your image in X
ny = 200 # number of pixels of your image in Y

# Warning: 
#        The star is assumed to be centered at the frame center as defined in
#        the vip_hci.var.frame_center function (geometric center of the image,
#        e.g. either in the middle of the central pixel for odd-size images or 
#        in between 4 pixel for even-size images).

itilt = 0 # inclination of your disk in degreess (0 means pole-on, 90 means edge on)
pa= 30 # position angle of the disk in degrees (0 means north, 90 means east)
a = 50 # semimajoraxis of the disk in au 
    # semimajor axis in arcsec is 80 au/80px = 1 arcsec

# 1. Let's try a pole-on symmetric diks without asnisotropy of scattering
fake_disk1 = vip.metrics.scattered_light_disk.ScatteredLightDisk(\
                        nx=nx,ny=ny,distance=dstar,\
                        itilt=itilt,omega=0,pxInArcsec=pixel_scale,pa=pa,\
                        density_dico={'name':'2PowerLaws','ain':4,'aout':-4,\
                        'a':a,'e':0.0,'ksi0':1.,'gamma':2.,'beta':1.},\
                        spf_dico={'name':'HG', 'g':0., 'polar':False})
fake_disk1_map = fake_disk1.compute_scattered_light()

fake_disk1_map = fake_disk1_map/np.max(fake_disk1_map)

ds9.display(fake_disk1_map)

# you can play with a, ain>0, aout<0 itilt and pa. 
# a in arcsec could be between 0.5arcsec and 4 arcsec

#%% then you can inject a fake disk in a cube
scaling_factor = 0.1
cube_fakeddisk = vip.metrics.cube_inject_fakedisk(fake_disk1_map*scaling_factor ,parang,psf=psf)



