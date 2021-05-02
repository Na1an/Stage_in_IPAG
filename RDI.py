import os
import cv2
import sys
import heapq
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from astropy.io import fits
from matplotlib.widgets import Slider
from sklearn.metrics import log_loss
from scipy import stats
from skimage.metrics import structural_similarity as ssim

# program start and program end 
def start_and_end(start):
    if(start):
        print("######### program start ###########")
    else:
        print("########## program end ############")

# 1. travesal the SPHERE_DC_DATA and get all the reference master cubes
def get_reference_cubes(repository_path, keyword):
    '''
    Args:
        repository_path : a string. The path of SPHERE DC
        keyword : a string. What kind of files we want (ex. MASTER_CUBE)
    Rrturn:
        res : a list of string. Return the path of all related files.
    '''
    res = []
    reps = os.listdir(repository_path)
    
    for rep in reps:
        files_sub = os.path.join(repository_path, rep)
        # print(files_sub)
        if os.path.isdir(files_sub):
            res = res + get_reference_cubes(files_sub, keyword)
        if keyword in files_sub:
            res = res + [files_sub]
    return res

def remove_target(target,refs):
    #res = refs
    for s in refs:
        if s.split('/')[-2] == target.split('/')[-1]:
            refs.remove(s)
            break
    return refs

# read one file and return its data
def read_file(file_path, keyword):
    '''
    Args:
        file_path : a string. The file path!
    Return:
        return the data of hd[0],hd type HDUList.
    '''
    
    '''
    hd = fits.open(get_reference_cubes(file_path, keyword)[0])
    # print("filename =", hd.fileinfo(0)["filename"].split("/")[-1])
    hd.info()
    print('\n')
    data = hd[0].data
    hd.close()
    '''
    return fits.getdata(get_reference_cubes(file_path, keyword)[0])

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

# 2. Collect the data from SPHERE_DC_DATA
def collect_data(files_path, scale=0.25):
    '''
    Args:
        files_path : a list of string. files path contain keyword.
        scale : a float. The scale in center that we want process, is equal to 1/4 by default.
    Rrturn:
        ref_frames : ndarray, 4 dimensions. Return (wavelengths, nb_frames, x, y)
    '''
    hd = fits.getdata(files_path[0])
    
    # frames in the first wavelength and second wavelength
    # K1/K2, H2/H3, etc...
    tmp = (1-scale)*0.5
    size = len(hd[0][0])
    start = int(size*tmp)
    end = int(size*(1-tmp))
   
    ref_frames = hd[..., start:end, start:end]

    for i in range(1,len(files_path)):
        hd = fits.getdata(files_path[i])
        ref_frames =np.append(ref_frames, hd[..., start:end, start:end], axis=1)

    return ref_frames

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

# chose the best correlated reference stars, not all
def selection(nb_best, target, refs, scale, wave_length=0):
    '''
    Args:
        nb_best : a integer. How many best ref stars we want.
        target : a numpy.ndarray. (wavelengths, nb_frames, x, y)
        refs : a list of string. All stars data we have.
        wave_length : a integer. Wave length of the cube.
    Rrturn:
        res : a list of string. The int(nb_best) best chosen ref stars.
    '''
    res = {}
    # target_median is 2 dims. (256, 256)
    target_median = median_of_cube(target, wave_length)
    w, h = target_median.shape
    target_median_vector = np.reshape(target_median,(w*h)) 

    for i in range(len(refs)):
        # hd is 4 dims: (wl, nb frmes, x, y)
        hd = fits.getdata(refs[i])
        # ref_median is 2 dims. (256, 256)
        ref_meidan = median_of_cube(slice_frame(hd,len(hd[wave_length,i,0]),scale), wave_length)
        ref_meidan_vector = np.reshape(ref_meidan, (w*h)) 
        coef_corr = np.corrcoef(target_median_vector, ref_meidan_vector)
        #print(refs[i],"=",coef_corr[0,1])
        res[refs[i]] = coef_corr[0,1]
    tmp = sorted(res.items(),key = lambda r:(r[1],r[0]), reverse=True)
    #print(tmp)
    print(">> There are", len(tmp), "reference stars in the library")
    print(">> we will chose", nb_best, "correlated cube to do PCA on RDI")
    res_bis = []
    for k in range(nb_best):
        (x,y) = tmp[k]
        res_bis.append(x)
        print(k,"- corrcoef value =", y)

    return res_bis 

# 3. Classic ADI
def process_ADI(science_frames, rotations):
    '''
    Args:
        science_frames : a numpy.ndarray. (wavelengths, nb_frames, x, y)
        rotations : a numpy.ndarry. The angles of ratations    
    Return:
        res : a numpy.ndarray, 4 dimesi. Ex. (2 wavelengths, 24 frames, 256, 256).
    '''
    
    wave_length, sc_fr_nb, w, h = science_frames.shape
    f_median = np.zeros((wave_length, w, h))
    res = np.zeros((wave_length, w, h))
    
    for wl in range(wave_length):
        for i in range(w):
            for j in range(h):
                f_median[wl, i, j] = np.median(science_frames[wl, :, i, j])
             
    for wl in range(wave_length):
        for n in range(sc_fr_nb):
            res[wl] = res[wl] + rotate((science_frames[wl, n] - f_median[wl]), rotations[n])
    
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

# A mask, cover the center of image
def mask(w, h, radius):
    '''
    Args:
        w : an integer. The weight of image.
        h : an integer. The height of image.
        radius : an integer. The radius of mask.
    Return:
        res : a numpy.ndarray, 2 dimens. Ex. (256, 256) but the center is all 0.
    '''
    res = np.full((w,h),True)
    x = w//2
    y = y//2
    for i in range(w):
        for j in range(h):
            if distance(i, j, x, y) < radius:
                res[i,j] = False
    
    return res

# 4. KLIP/PCA - Principle Component Analysis
#   Be attention! The values of science_frames and ref_frames will change !!!
#   copy/deepcopy may solve the probleme
#   For corelation, we may multiple standard variance in the end.
def PCA(science_frames, ref_frames, K, wl=0):
    '''
    Args:
        science_frames : a numpy.ndarray. (wavelengths, nb_frames, x, y)
        ref_frames : a numpy.ndarray. (wavelengths, nb_frames, x, y)
        K : a integer. K is K_klip here (K_klip <= rf_fr_nb).
        wl : a interger. The wavelength.
    Return:
        res : a numpy.ndarray, 3 dimens. Ex. (24 frames, 256, 256).
    '''
    # 0 reshape image to a vector : (nb of frames, 128, 128) -> (nb of frames, 128*128) 
    sc_fr_nb, w, h = science_frames[wl].shape
    rf_fr_nb = len(ref_frames[wl])
    
    science_frames_vector = np.reshape(science_frames[wl], (sc_fr_nb, w*h))
    ref_frames_vector = np.reshape(ref_frames[wl], (rf_fr_nb, w*h))

    # 1 zero both science_frales and ref_frames, partition target T and refs R in the library 
    for f in range(sc_fr_nb):
        mean = np.mean(science_frames_vector[f])
        science_frames_vector[f] = (science_frames_vector[f] - mean)/np.std(science_frames_vector[f])
        print("---", f+1, "of", sc_fr_nb,"--- substract mean from science_frames")

    for f_r in range(rf_fr_nb):
        mean_r = np.mean(ref_frames_vector[f_r])
        ref_frames_vector[f_r] = (ref_frames_vector[f_r] - mean_r)/np.std(ref_frames_vector[f_r]) 
        print("---", f_r+1, "of", rf_fr_nb,"--- substract mean from rf_frames")

    # 2 compute the Karhunen-Loève transform of the set of reference PSFs Rk(N)? 
    # inner product for each frame of target and each frame of references
    # K = rf_fr_nb
    
    # cov(bias=false), so it is divided by N-1 not N
    # ERR Covariance matrix
    N = w*h
    ERR = np.cov(ref_frames_vector) * (N-1)
    # lambda_k - eigenvalue (increase), C_k - eigenvector
    lambda_incre, C_incre = np.linalg.eigh(ERR)
    lambda_k = np.flip(lambda_incre)
    C_k = np.flip(C_incre, axis=1)
    # Z_KL_k.shape = (N, K) , N because c_k*ref[p], k in K
    Z_KL_k = np.zeros((N, rf_fr_nb)) 
    for k in range(rf_fr_nb): # put the biggest eigenvalue at first
        for p in range(rf_fr_nb):
            Z_KL_k[:,k] = Z_KL_k[:,k] + (1/np.sqrt(lambda_k[k]))*(C_k[p, k] * ref_frames_vector[p]) 
        print("---", k+1, "of", rf_fr_nb,"--- eigenvalue")
    
    # 3 choose a number of modes K = 30
    Z_KL_chosen = Z_KL_k[:,:K]
    
    # 4 compute the best estimate
    # 5 substact the I[i] from science_frames[i]
    res = np.zeros((sc_fr_nb, N))
    for f in range(sc_fr_nb):
        for k in range(K):
            inner_product = np.dot(Z_KL_chosen[:,k], science_frames_vector[f])
            # res[f] stock the PSF image
            res[f] = res[f] + inner_product*(Z_KL_chosen[:,k])
        res[f] = science_frames_vector[f] - res[f] 
        print("---", f+1, "of", sc_fr_nb,"--- substracut res from science_frames")

    return res.reshape((sc_fr_nb, w, h))

# 4. process the science frames, substract the starlight
# we do care wavelength!
def process_RDI(science_frames, ref_frames):
    '''
    Args:
        science_frames : a numpy.ndarray. (wavelengths, nb_frames, x, y)
        ref_frames : a numpy.ndarray. (wavelengths, nb_frames, x, y)
        K : a interger. The mode of KL or the number of references frames
        Normally, the scale in 4 dimens is consistent for two args.
    Return:
        res : a numpy.ndarray, 4 dimesi. Ex. (2 wavelengths, 24 frames, 256, 256).
    '''
    print(">> science_frame shap =", science_frames.shape) 
    print(">> ref_frame shap =", ref_frames.shape)
   
    wave_length = len(science_frames)
    sc_fr_nb = len(science_frames[0])
    side_len = len(science_frames[0,0])
    rf_fr_nb = len(ref_frames[0])
    
    res = np.zeros((wave_length, sc_fr_nb, side_len, side_len))
    # for both wavelength
    for wl in range(wave_length):
        print(">> wavelength =", wl, "start process")
        for i in range(sc_fr_nb):
            tmp = -1
            pearson_corr = -1
            indice = 0 #most relevent indicfor j in range(ref_frames):
            for j in range(rf_fr_nb):
                pearson_corr = np.corrcoef(science_frames[wl, i], ref_frames[wl, j])[0,1]
                if(pearson_corr>tmp):
                    tmp = pearson_corr
                    indice = j
            res[wl, i] = science_frames[wl, i] - ref_frames[wl, indice]

            print("---", i+1, "of", sc_fr_nb,"---")
        print("<< wavelength = ", wl, "end process")
    return res

def process_RDI_ssim(science_frames, ref_frames):
    '''
    Args:
        science_frames : a numpy.ndarray. (wavelengths, nb_frames, x, y)
        ref_frames : a numpy.ndarray. (wavelengths, nb_frames, x, y)
        Normally, the scale in 4 dimens is consistent for two args.
    Return:
        don't know yet.
    '''
    print(">> science_frame shap =", science_frames.shape) 
    print(">> ref_frame shap =", ref_frames.shape)
   
    wave_length = len(science_frames)
    sc_fr_nb = len(science_frames[0])
    rf_fr_nb = len(ref_frames[0])
    side_len = len(science_frames[0,0])
    
    res = np.zeros((wave_length, sc_fr_nb, side_len, side_len))
    # for both wavelength
    for wl in range(wave_length):
        print(">> wavelength =", wl, "start process")
        for i in range(sc_fr_nb):
            tmp = -1
            ssim_value = -1
            indice = 0 #most relevent indicfor j in range(ref_frames):
            for j in range(rf_fr_nb):
                ssim_value = ssim(science_frames[wl, i], ref_frames[wl, j])
                if(ssim_value>tmp):
                    tmp = ssim_value
                    indice = j
            res[wl, i] = science_frames[wl, i] - ref_frames[wl, indice]

            print("---", i+1, "of", sc_fr_nb,"---")
        print("<< wavelength = ", wl, "end process")
    return res

# main bloc
if __name__ == "__main__":
    '''
    Args:
        argv1 : the algo mode cADI/PCA/RDI
        argv2 : the path of target/science object repository
        *argv3 : the path of reference objects repository. (for cADI, argv3 is the scale)
        last argv : the scale
    '''
    # the interesting ares of the image
    scale = 0.25
    start_and_end(True)
    
    # algo option
    opt = str(sys.argv[1]).upper()
    if opt == "CADI":
        # cADI - ok!
        print(">> Algo cADI is working! ")
        
        # 1. get the target frames from Master cube
        target_frames = read_file(str(sys.argv[2]), "MASTER_CUBE-center")
        
        # 2. argv[3] may give us the scale
        if(sys.argv[3] is not None):
            scale = float(sys.argv[3])
        
        # 3. result of cADI
        res_cADI = process_ADI(slice_frame(target_frames, len(target_frames[0, 0, 0]), scale), read_file(str(sys.argv[2]),"ROTATION"))
        hdu = fits.PrimaryHDU(res_cADI)
        hdu.writeto("./res_tmp/GJ_667C_cADI_0.5.fits") 

    elif opt == "PCA":
        # PCA - ok!
        print(">> Algo PCA is working! ")
        
        if(len(sys.argv) >4):
            scale = float(sys.argv[4])
        
        # 1. get target frames 
        target_frames = read_file(str(sys.argv[2]), "MASTER_CUBE-center")
        side_len = len(target_frames[0, 0, 0])
        target_frames = slice_frame(target_frames, side_len, scale) 
        print("Scale = ", scale, "\ntarget_frames shape =", target_frames.shape)
        # 2. get the list of files contain keyword and remove the target
        ref_files = get_reference_cubes(str(sys.argv[3]), "MASTER_CUBE-center")
        
        # now, the target is not in the references frames
        ref_files = remove_target(str(sys.argv[2]),ref_files)
        print(">> what we have in ref_res")
        for s in ref_files:
            print(s)
        
        # select the best correlated targets
        ref_files = selection(5, target_frames, ref_files, scale, 0) # 0 is the default wave length
        #print(ref_files) 
        
        # 3. put the related data (all frames of the reference cubes) in np.array
        ref_frames = collect_data(ref_files, scale)
        print("ref_frames shape =", ref_frames.shape)
        
        # 4. PCA
        # last arg is the K_klip
        for n in range(1,21):
            res = PCA(target_frames, ref_frames, n) #K_klip
            tmp = np.zeros((int(side_len*scale), int(side_len*scale))) 
            
            rotations_tmp = read_file(str(sys.argv[2]),"ROTATION") 
            for i in range(len(res)):
                tmp = tmp + rotate(res[i] , rotations_tmp[i])
            hdu = fits.PrimaryHDU(tmp)
            path = "./K_kilp_ADI_RDI/RDI_After" + str(n) + ".fits"
            hdu.writeto(path) 
            print(">>===", n, "of", 20,"=== fits writed ===")
    
    elif opt == "RDI":
        # RDI - wroking on
        # argv1 : the path of repository contains science object
        science_frames = read_file(str(sys.argv[1]), "MASTER_CUBE-center")

        # Step 1: get the list of files contain keyword
        all_files = get_reference_cubes(str(sys.argv[2]), "MASTER_CUBE-center")
        
        # Step 2: put the related data (all frames of the reference cubes) in np.array
        ref_frames = collect_data(all_files, scale)
        
        # Step 3: process the science frames
        sc_frames_procced = process_RDI(slice_frame(science_frames, len(science_frames[0][0][0]), 0.25), ref_frames)
    
    elif opt== "TEST":
        target_frames = read_file(str(sys.argv[2]), "MASTER_CUBE-center")
        rotations = read_file(str(sys.argv[2]), "ROTATION") 
        res = median_of_cube_test(target_frames, rotations, 0.25)
        hdu = fits.PrimaryHDU(res)
        hdu.writeto("./res_tmp/target_rotated_median.fits") 

    else:
        print("Option is available for now.")

    start_and_end(False)

