#################################################
# Private module, put some useful funtions here #
#################################################
import numpy as np
import matplotlib.pyplot as plt

# Global variable
MASK_RADIUS = 32

# just the affichage for program start and program end
def start_and_end(start):
    '''
    Args:
       start : a boolean. If it is the start of the program 
    Return:
        None
    '''
    if(start):
        print("######### program start ###########")
    else:
        print("########## program end ############")


# slice frame, we only take the interesting area
# for exemple, 1/4 in the center of each frame
def slice_frame(frames, size, center_scale):
    '''
    Args:
        frames : np.array, 4 dims. Contains all the frames on all wavelength in a cube
        size : a int. Frames size, size = 1024 in case 1024*1024.
        center_scale : a float. The scale in center that we will process.
    Return:
        Sliced frames, np.array, 3 dims.
    '''

    tmp = (1-center_scale)*0.5
    res = frames[..., int(size*tmp):int(size*(1-tmp)), int(size*tmp):int(size*(1-tmp))]

    return res

# print the plot of the best n reference stars' pearson coefs 
def print_best_ref_stars_pearson_coef(res_coef):
    '''
    Args:
       res_coef : a dict. Store the name of the best reference and its pearson coef.
    Return:
        None. Print the plot of best Pearson correlated coef value.
    '''
    print(res_coef)
    plt.bar(res_coef.keys(), res_coef.values())
    plt.ylim(0.6,1)
    plt.xticks(rotation=45)
    plt.ylabel("Pearson correlation coefficient", fontsize= "16")
    plt.show()

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

# display the list of rotation angles
def display_rotaion_angles(data):
    '''
    Args:
        data : a string. A list of pixels/value of derotation for the frame.
    Return:
        No return.
    '''
    print("frame numbers = ",len(data))
    for i in range(len(data)):
        print(str(i)+"th rotation =", data[i])

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

# A mask, cover the center of image
def create_mask(w, h, radius=MASK_RADIUS):
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
