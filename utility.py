import os
import cv2
import time
import numpy as np
import skimage
import vip_hci as vip
import matplotlib.pyplot as plt

from hciplot import plot_frames, plot_cubes

# start or end of the program
def start_and_end_program(start):
    '''
    Args:
       start : a boolean. If it is the start of the program 
    Return:
        None
    '''
    localtime = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))
    if(start):
        print("######### program start :", localtime,"###########")
    else:
        print("########## program end :", localtime, "############")
# get one fwhm of a frame
def get_fwhm_from_psf(psf):
    '''
    Args:
        psf : a 2D np.ndarray. The PSF in one wavelength. 
    Return:
        res : a float. The one fwhm of the psf.
    ''' 
    fwhm = vip.var.fit_2dgaussian(psf, crop=True, cropsize=9, debug=False)
    
    return np.mean([fwhm.loc[0,'fwhm_y'], fwhm.loc[0,'fwhm_x']])

# print info for a file *.fits
def print_info(path):
    '''
    Args:
        path : a string. The path of one file *.fits
    Return:
        None.
    '''
    vip.fits.info_fits(path)

# get pxscale of the IRDIS
def get_pxscale():
    '''
    Args:
        None.
    Return:
        res : a float. Return the pxscale of the IRDIS.
    '''
    res = vip.conf.VLT_SPHERE_IRDIS['plsc']
    #print("In SPHERE IRDIS : pxscale =", res, "arcsec/px")
    return res 

# slice frame, we only take the interesting area
# for exemple, 1/4 in the center of each frame
def crop_frame(science_target, size, center_scale):
    '''
    Args:
        science_target : np.array, 4 dims. Contains all the frames on all wavelength in a cube
        size : a int. Frames size, size = 1024 in case 1024*1024.
        center_scale : a float. The scale in center that we will process.
    Return:
        Sliced frames, np.array, 3 dims.
    '''

    tmp = (1-center_scale)*0.5
    res = science_target[..., int(size*tmp):int(size*(1-tmp)), int(size*tmp):int(size*(1-tmp))]
    return res

# remvove the target from the reference list
def remove_target(target, refs):
    '''
    Args:
        target : a string.
        refs : a list of string
    Return:
        refs : a list of string. Target string removed.
    '''
    #res = refs
    for s in refs:
        if s.split('/')[-2] == target.split('/')[-1]:
            refs.remove(s)
            break
    return refs

# median of cube
def median_of_cube(cube, wl=0):
    '''
    Args:
        cube : a numpy.ndarray. (wavelengths, nb_frames, x, y)
        wl : a integer. Wave length of cube.
    Return:
        res : a numpy.ndarray, 2 dimensions. Ex. (256, 256).
    '''
    wave_length, sc_fr_nb, w, h = cube.shape
    res = np.zeros((w,h))
    for i in range(w):
        for j in range(h):
            res[i,j] = np.median(cube[wl,:,i,j])
    return res

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

# class : template of frame, for sapt-annular-mean
class FrameTemp:
    '''
    This class is the objet of a frame template. It simplifie the computation when we want to have the background mean in different separations.
    Note:
        Do not include the `self` parameter in the ``Args`` section.
    Args:
        side (int): the side length of a frame.
    Attributes:
        side (int): the side length of a frame.
    '''

    # init function, we need make the template in this step
    def __init__(self, side):
        '''
        Args:
            self : object itself.
            side : side lenght of a frame. Normally, it is a odd number 
        '''
        if (side%2) != 1:
            raise Exception("Side is not a odd number! So crop the frame. side =", side)
        
        self.side = side
        self.radius = (side//2)+1
        self.values_mean = []  
        self.coords = [None]*self.radius

        # center
        cent_x = side//2
        cent_y = side//2

        # make the template : repartition the coordinates
        for rth in range(self.radius):
            coord = []
            for i in range(side):
                for j in range(side):
                    d = distance(i,j,cent_x,cent_y)
                    if rth-1 < d and d <= rth:
                        coord.append((i,j))
            self.coords[rth] = coord
    
    # process the image, have it seperation mean
    def separation_mean(self, frame, detail=False):
        '''
        Process the input frame, do the mean option
        Args:
            self : object it self.
            frame : a ndarry, 2 dims. The input frame.
            detail : a boolean. The default is false. 
        Return:
            res : a ndarry, 2 dims. The output frame separation mean.
        '''
        if len(frame) != self.side:
            raise Exception("The shape of input frame is different from template, len(frame) =", len(frame))
        
        res = np.zeros((frame.shape))

        for i in range(self.radius):
            tmp = 0
            for (x,y) in self.coords[i]:
                tmp = tmp + frame[x, y]
            
            mean = tmp/len(self.coords[i])
            if detail is True:
                print( "mean =", mean)
                print("len this layer =", len(self.coords[i]))
            for (x,y) in self.coords[i]:
                res[x, y] = mean
        
        return res 
    
    # print the property
    def print_property(self):
        '''
        Just print the self.side, self.radius, nothing special.
        Args:
            self : object it self.
        Return:
            None.
        '''
        print("self.side =", self.side)
        print("self.radius =", self.radius)
    
    # print the coords
    def print_coords(self):
        '''
        Just print the self.coords, nothing special.
        Args:
            self : object it self.
        Return:
            None.
        '''
        for i in range(self.radius):
            print(self.coords[i]) 
        
