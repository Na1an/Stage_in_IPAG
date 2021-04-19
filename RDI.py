import os
import cv2
import sys
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
        print("######### program start ###########\n")
    else:
        print("\n########## program end ############")

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

# 3. Classic ADI
def process_ADI(science_frames, rotations):
    '''
    Args:
        science_frames : a numpy.ndarray. (wavelengths, nb_frames, x, y)
    Return:
        res : a numpy.ndarray, 4 dimesi. Ex. (2 wavelengths, 24 frames, 256, 256).
    '''
    
    wave_length, sc_fr_nb, w, h = science_frames.shape
    f_median = np.zeros((wave_length, w, h))
    #f_residual = np.zeros((wave_length, sc_fr_nb, w, h))
    res = np.zeros((wave_length, w, h))
    
    for wl in range(wave_length):
        for i in range(w):
            for j in range(h):
                f_median[wl, i, j] = np.median(science_frames[wl, :, i, j])
             
    for wl in range(wave_length):
        for n in range(sc_fr_nb):
            #f_residual[wl, n] = rotate((science_frames[wl, n] - f_median[wl]), rotations[n])
            res[wl] = res[wl] + rotate((science_frames[wl, n] - f_median[wl]), rotations[n])
    
    return res 

# 4. KLIP/PCA - Principle Component Analysis
#   Be attention! The values of science_frames and ref_frames will change !!!
#   copy/deepcopy may solve the probleme
def PCA(science_frames, ref_frames, K, wl=0):
    '''
    Args:
        science_frames : a numpy.ndarray. (wavelengths, nb_frames, x, y)
        ref_frames : a numpy.ndarray. (wavelengths, nb_frames, x, y)
        Normally, the scale in 4 dimens is consistent for two args.
    Return:
        res : a numpy.ndarray, 4 dimesi. Ex. (2 wavelengths, 24 frames, 256, 256).
    '''
    # 0 partition target T and refs R in the library 
    
    # 1 zero both science_frales and ref_frames
    sc_fr_nb, w, h = science_frames[wl].shape
    rf_fr_nb = len(ref_frames[wl])
    
    print("shape = ", science_frames.shape)
    print("shape = ", ref_frames.shape)

    science_frames_pca = np.reshape(science_frames[wl], (sc_fr_nb, w*h))
    ref_frames_pca = np.reshape(ref_frames[wl], (rf_fr_nb, w*h))
    print("shape = ", science_frames_pca.shape)
    print("shape = ", ref_frames_pca.shape)

    for f in range(sc_fr_nb):
        mean = np.mean(science_frames_pca[f])
        science_frames_pca[f] = science_frames_pca[f] - mean

    for f_r in range(rf_fr_nb):
        mean_r = np.mean(ref_frames_pca[f_r])
        ref_frames_pca[f_r] = ref_frames_pca[f_r] - mean_r 
    
    # 2 compute the Karhunen-Loève transform of the set of reference PSFs Rk(N)? 
    # inner product for each frame of target and each frame of references
    cov_matrix_ref = np.cov(ref_frames_pca) * (w*h-1)
    #plt.imshow(cov_matrix_ref, origin='lower')
    #plt.show()
    lambda_k, C_k = np.linalg.eigh(cov_matrix_ref)
    lambda_k_reverse = np.flip(lambda_k)
    C_k_reverse = np.flip(C_k, axis=1)
    
    N = w*h
    Z_KL_k = np.zeros((N, rf_fr_nb)) 
    for k in range(rf_fr_nb):
        for p in range(rf_fr_nb):
            Z_KL_k[:,k] = Z_KL_k[:,k] + (1/np.sqrt(lambda_k_reverse[k]))*(C_k_reverse[p, k] * ref_frames_pca[p]) 
    
    #Z_KL_images = Z_KL_k.T.reshape((rf_fr_nb, w, h))
    
    # 3 choose a number of modes K = 30
    Z_KL_chosen = Z_KL_k[:,:K]
    # 4 compute the best estimate
    # 5 substact the I[i] from science_frames[i]
    I = np.zeros((sc_fr_nb, N))
    for f in range(sc_fr_nb):
        for k in range(K):
            inner_product = np.dot(Z_KL_chosen[:,k], science_frames_pca[f])
            I[f] = I[f] + inner_product*(Z_KL_chosen[:,k])
        I[f] = science_frames_pca[f] - I[f] 

    return I.reshape((sc_fr_nb, w, h))

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
        argv1 : the target/science object
        argv2 : the repository path
    '''
     
    '''
    data = read_file(str(sys.argv[1]))
    display_rotaion_angles(data)
    '''
    scale = 0.125

    start_and_end(True)
    
    # argv1 : the path of repository contains science object
    science_frames = read_file(str(sys.argv[1]), "MASTER_CUBE-center")
    #print(">> Science frames type", type(science_frames), " shape=", science_frames.shape,'\n')
    
    # Step 1: get the list of files contain keyword
    all_files = get_reference_cubes(str(sys.argv[2]), "MASTER_CUBE-center")
   
    # Step 2: put the related data (all frames of the reference cubes) in np.array
    ref_frames = collect_data(all_files, scale)
    
    # Step 3: process the science frames
    '''
    sc_frames_procced = process_ADI(slice_frame(science_frames, len(science_frames[0][0][0]), 1.0), read_file(str(sys.argv[1]),"ROTATION"))
    hdu = fits.PrimaryHDU(sc_frames_procced)
    hdu.writeto("./res_tmp/res01.fits") 
    '''
    #sc_frames_procced = process_RDI(slice_frame(science_frames, len(science_frames[0][0][0]), 0.25), ref_frames)
    
    # Step 4: PCA
    res = PCA(slice_frame(science_frames, len(science_frames[0, 0, 0]), scale), ref_frames, 30)
    #hdu = fits.PrimaryHDU(slice_frame(science_frames, len(science_frames[0, 0, 0]), scale)[0])
    #hdu = fits.PrimaryHDU(res)
    #hdu.writeto("./res_tmp/target.fits") 
    tmp = np.zeros((128, 128))
    rotations_tmp = read_file(str(sys.argv[1]),"ROTATION") 
    for i in range(len(res)):
        tmp = tmp + rotate(res[i] , rotations_tmp[i])
    hdu = fits.PrimaryHDU(tmp)
    hdu.writeto("./res_tmp/res_after_pca.fits") 

    #hdul.close()
    start_and_end(False)

