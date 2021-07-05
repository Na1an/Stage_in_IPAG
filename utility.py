import os
import cv2
import math
import time
import numpy as np
import skimage
import vip_hci as vip
import matplotlib.pyplot as plt
from astropy.io import fits
from hciplot import plot_frames, plot_cubes

# Global constant
MASK_RADIUS = 15

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

# A mask, cover the center of image, inner mask
def create_inner_mask(w, h, radius=MASK_RADIUS):
    '''
    Args:
        w : an integer. The weight of image.
        h : an integer. The height of image.
        radius : an integer. The radius of mask.
    Return:
        res : a numpy.ndarray, 2 dimens. Ex. (256, 256) but the center is all 0.
    '''
    count = 0
    res = np.full((w,h),True)
    x = w//2
    y = h//2
    for i in range(w):
        for j in range(h):
            if distance(i, j, x, y) <= radius:
                res[i,j] = False
                count = count + 1
    return res, count

# A mask, cover the center of image, outer mask
def create_outer_mask(w, h, radius):
    '''
    Args:
        w : an integer. The weight of image.
        h : an integer. The height of image.
        radius : an integer. The radius of mask.
    Return:
        res : a numpy.ndarray, 2 dimens. Ex. (256, 256) but the center is all 0.
    '''
    count = 0
    res = np.full((w,h),True)
    x = w//2
    y = h//2
    for i in range(w):
        for j in range(h):
            if distance(i, j, x, y) >= radius:
                res[i,j] = False
                count = count + 1
    return res, count

# rotate the frames in a cube
def rotate(image, angle, center=None, scale=1.0):
    '''
    Args:
        image : a 2 dimension list, a image.
    Return:
        rotated : a 2 dimension list, the image rotated.
    '''
    # grab the dimensions of the image
    # should be 128 * 128
    (h, w) = image.shape[:2]

    # if the center is None, initialize it as the center of the image
    if center is None:
        center = (w // 2, h // 2)

    # perform the rotation
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, rotation_matrix, (w, h))

    # return the rotated image
    return rotated

# store the median of the cube and rotate -- Not work
def median_of_cube_test(science_frames, rotations, scale):
    '''
    Args:
        science_frames : a numpy.ndarray. (wavelengths, nb_frames, x, y)
        rotations : a numpy.ndarry. The angles of ratations
        scale : a float. The scale of study area in the center of frames.
    Return:
        res : a numpy.ndarray, 3 dimensions. Ex. (2 wavelengths, 256, 256).
    '''
    wave_length, sc_fr_nb, w, h = science_frames.shape
    f_median = np.zeros((wave_length, w, h))
    res = np.zeros((wave_length, int(w*scale), int(h*scale)))

    for wl in range(wave_length):
        for i in range(w):
            for j in range(h):
                f_median[wl, i, j] = np.median(science_frames[wl, :, i, j])

    for wl in range(wave_length):
        for n in range(sc_fr_nb):
            res[wl] = res[wl] + rotate((science_frames[wl, n] - f_median[wl]), rotations[n])

    return None

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

# chose wave length and 
def chose_reference_files(refs, ky_wl, ky_type):
    '''
    Args:
        target : a string.
        refs : a list of string
    Return:
        refs : a list of string. Target string removed.
    '''
    res = []
    for s in refs:
        if (ky_wl in s) and (ky_type in s):
            res.append(s)
            print(s)
    return res

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

# plot a image
def plot_image(img):
    '''
    Args:
        img : an 2D ndarray, a image. 
    Return:
        None 
    '''
    c = plt.imshow(img, interpolation='nearest', origin='lower',extent=(0,3,0,3))
    plt.colorbar(c)
    plt.title('lala')
    plt.show()
    
    return None

# sotre data as fits
def store_to_fits(data, path):
    '''
    Args:
        data : a ndarray, 2 or 3 dims. The cube we want to sotre.
        path : a string. The file path. 
    Return:
        None
    '''
    hdu = fits.PrimaryHDU(data)
    hdu.writeto(path)

# give value to a image
def give_value(ll, t, v):
    for (x,y) in ll:
        t[x, y] = v
    return None

# class : template of frame, for sapt-annular-mean
class FrameTemp:
    '''
    This class is the objet of a frame template. It simplifie the computation when we want to have the background mean in different separations.
    Note:
        Do not include the `self` parameter in the ``Args`` section.
    Args:
        side : an integer. The side length of a frame.
    Attributes:
        side : an integer. The side length of a frame.
        radius : an integer. The radius of this frame template.
        values_mean : a list of integer. The mean value for each annulus/radius.
        coords : a list of tuple (x,y). The coordinates of each pixels in the frame template.
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
        self.values_mean = [None]*self.radius
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
            raise Exception("The shape of input frame is different from template, len(frame) =", len(frame), "and self.side =", self.side)
        
        res = np.zeros((frame.shape))

        for i in range(self.radius):
            tmp = 0
            for (x,y) in self.coords[i]:
                tmp = tmp + frame[x, y]
            
            mean = tmp/len(self.coords[i])
            self.values_mean[i] = mean
            
            if detail is True:
                print("mean =", mean)
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

# remove separation mean from a cube
def remove_separation_mean_from_cube(cube):
    '''
    Only consider the input cube has even shape. Ex. (..., 256, 256) 
    Args:
        cube : a ndarry, 3 dims. The input cube. ( nb_frames, x, y)
    Return:
        res : a ndarry, 3 dims. The result of seperation mean image.
    '''
    nb_fr, w, h = cube.shape
    res = np.zeros((nb_fr, w, h))
    temp = FrameTemp(w-1)
    # traversal the cube in wave length = wl
    for i in range(nb_fr):
        sep_mean = temp.separation_mean(cube[i, 1:,1:])
        cube[i, 1:, 1:] = cube[i, 1:, 1:] - sep_mean
        res[i, 1:, 1:] = sep_mean
        print("===", i+1, "of", nb_fr, " separation_mean removed ===")

    return res

"""
# amplitude
def get_amplitude_b(thetas, theta_zero, B):
    '''
    This is a function for calculating the amplutude of sinus.
    Used in the calss FrameTempRadian.eliminate_wdh_influence.
    Args:
        thetas : a list of angle. More explicit, the angles of each pixels in a annulus.
        thera_zero : a float. The image direction of a frame.
        B : a float. The average velue of the pixels in a annulus.
    Returns:
        A float. The amplitude of sinus for each annulus.
    '''
    tmp = 0
    sum_a = 0

    for (d, theta, x, y) in self.coords[i]:
        tmp = tmp + frame[x, y]
        sum_a = sum_a + frame[x, y]*math.sin(math.radians(2*theta-direction))/((np.linalg.norm(math.sin(math.radians(2*theta-direction))))**2)

    # B, theta_zero = direction
    b = tmp/len(self.coords[i])
    
    # A, amplitude

    return None
"""

# class : template of frame, in radian
class FrameTempRadian:
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
        self.amp_layers = [None]*self.radius

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
                        x = j - cent_y 
                        y = cent_x - i
                        coord.append((d, math.degrees(math.atan2(y, x)), i, j))
            self.coords[rth] = coord
    
    # process the image, remove the inlfluence of wind driven halo
    def wdh_influence(self, frame, direction, detail=False):
        '''
        Process the input frame, give the wind driven halo influence.
        Args:
            self : object it self.
            frame : a ndarry, 2 dims. The input frame.
            directions : a float. The image direction of this frame, wind driven halo, theta zero.
            detail : a boolean. The default is false. 
        Return:
            res : a ndarry, 2 dims. The output frame separation mean.
        '''
        if len(frame) != self.side:
            raise Exception("The shape of input frame is different from template, len(frame) =", len(frame))
        
        res = np.zeros((frame.shape))

        # try
        #direction = direction - 90

        for i in range(self.radius):
            # A, amplitude
            a = 0
            # B, theta_zero = direction
            b = 0
            angles = np.ones(len(self.coords[i]))

            k=0
            for (d, theta, x, y) in self.coords[i]:
                b = b + frame[x, y]
                angle_simp = theta-direction
                a = a + frame[x, y]*math.cos(math.radians(angle_simp))
                angles[k] = angle_simp
                k = k+1 
                '''
                if ((np.linalg.norm(math.cos(math.radians(angle_simp))))**2) <= 0:
                    a = a + frame[x, y]
                    #continue
                else:
                    a = a + frame[x, y]*math.cos(math.radians(angle_simp))/((np.linalg.norm(math.cos(math.radians(angle_simp))))**2)
                '''
            
            sum_down = np.linalg.norm(list(map(lambda x: math.cos(math.radians(x)), angles)))**2
            a = a/sum_down
            b = b/len(self.coords[i])

            if detail is True:
                print("This is layer", i, "- len(layer) =", len(self.coords[i]))
                print("angles =", angles)
                print("a =", a)
                print("b =", b)

            for (d, theta, x, y) in self.coords[i]:
                # case 1, same to case 2, but a is different
                
                angle_simp = theta-direction
                #angle_simp = theta-direction+90
                
                res[x, y] = a*math.cos(angle_simp) + b
                # case 2
                ##res[x, y] = a*math.sin(math.radians(2*theta-direction)) + b
                # case 3
                #res[x, y] = a*math.sin(math.radians(2*theta-direction))
                # case 4
                #res[x, y] = (a-b)*math.sin(math.radians(2*theta-direction))
                #print("a*math.sin(...)", res[x, y])
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

# attenuate wdh influence from a cube
def attenuate_wdh_influence_from_cube(cube, directions, detail=False):
    '''
    Only consider the input cube has even shape. Ex. (..., 256, 256) 
    Args:
        self : object it self.
        cube : a ndarry, 3 dims. The input cube. ( nb_frames, x, y).
        detail : a boolean. To see the detail.
    Return:
        wdh_influence : a ndarry, 3 dims. The output is the wdh influence of the input cube.
    '''
    nb_fr, w, h = cube.shape
    wdh_influence = np.zeros((nb_fr, w, h))
    temp = FrameTempRadian(w-1)
    # traversal the cube in wave length = wl
    for i in range(nb_fr):
        wdh_influence[i, 1:, 1:] = temp.wdh_influence(cube[i, 1:,1:], directions[i], detail)
        cube[i, 1:, 1:] = cube[i, 1:, 1:] - wdh_influence[i, 1:, 1:] # wdh_influence[i, 1:, 1:]*0.1...0.9
        print("===", i+1, "of", nb_fr, " attenuate wdh influence ===")

    return wdh_influence
    
# raw contrast
def get_raw_contrast(fwhm, frame):
    '''
    Args:
        fwhm : a float. The psf of the frame.
        frame : a ndarray, 2 dims. The input frame of a cube.
    Return:
        res : a ndarray, 1 dims. The raw contrast of this frame.
    '''
    w, h = frame.shape
    temp = FrameTemp(w-1)
    temp.separation_mean(frame[1:, 1:])

    return list(map(lambda x: x/fwhm, temp.values_mean))

# raw contrast for a cube
def get_raw_contrast_cube(fwhm, frame):
    '''
    Args:
        fwhm : a float. The psf of the frame.
        frame : a ndarray, 3 dims. The input frame of a cube. (nb_frames, x, y)
    Return:
        res : a ndarray, 1 dims. The raw contrast of this frame.
    '''
    nb_fr, w, h = frame.shape
    temp = FrameTemp(w-1)
    res = np.zeros((nb_fr, temp.radius))
    for i in range(nb_fr):
        temp.separation_mean(frame[i, 1:, 1:])
        res[i] = list(map(lambda x: x/fwhm, temp.values_mean))

    return res

# generate a list of tuple
def get_coords_of_ref_frames(nth, nb_frames):
    '''
    This function can help to preserve the information of frame coordinates. 
    Can help us find which image is used to process the target frame. 
    Args:
        nth : a integer. The nth cube in the reference library.
        nb_frames : a integer. The nth frame number of a cube.
    Return:
        res : a list of tuple. (nth cube, nth frame of the cube)
    '''
    res = []
    for i in range(nb_frames):
        res.append((nth, i, nb_frames))
    return res

# get histogram of reference stars
def get_histogram_of_ref_stars_score(ref_star_scores, ref_cube_nb_frames):
    '''
    This function will count how many frames we use for each star in the reference library. 
    Args:
        ref_star_scores : a list of integer. The list of indice, nth frame in the reference frame library.
        ref_cube_nb_frames : a list of integer. Each element is the frame number of a reference star.
    Return:
        res : a ndarray list of integer. The number of integer for each reference star we use. 
    '''
    l = len(ref_cube_nb_frames)
    print("l =", l)
    res = np.zeros(l)
    for i in ref_star_scores:
        # indice plus 1, then we can deal with it with the length of 
        i = i
        for n in range(l):
            i = i - ref_cube_nb_frames[n]
            if i<=0:
                res[n] = res[n] + 1
                break
    
    return res

# make a dictionary from two list
def get_dict(key, value):
    '''
    This function will count how many frames we use for each star in the reference library. 
    Args:
        key : a list of element, string or integer... Input key is the ref_files.
        value : a list of element, string or integer...
    Return:
        res : a dict. For drawing a barplot. 
    '''
    res = {}
    
    for i in range(len(key)):
        
        k = fits.open(key[i])[0].header['OBJECT']
        v = value[i]

        if v == 0:
            continue

        if k not in res.keys():
            res[k] = value[i]
        else:
            res[k] = res[k] + value[i]
        
    return res

# convert a list of tuple to list 
def list_of_tuple_to_2_list(l):
    '''
    This function will convert a list of double tuple to 2 list. 
    Args:
        l : a list of tuple (x,y).
    Return:
        res1 : a list of string. The name of reference star.
        res2 : a list of integer. The value of counting.
    '''
    res1 = []
    res2 = []
    for (x,y) in l:
        res1.append(x)
        res2.append(y)
    
    return res1, res2