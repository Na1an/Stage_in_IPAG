import os
import sys
import numpy as np
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

# read one file and return its data
def read_file(file_path):
    '''
    Args:
        file_path : a string. The file path!
    Return:
        return the data of hd[0],hd type HDUList.
    '''
    hd = fits.open(file_path)
    # print("filename =", hd.fileinfo(0)["filename"].split("/")[-1])
    hd.info()
    print('\n')
    data = hd[0].data
    hd.close()
    return data

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
        frames : np.array, 3 dims. Contains all the frames on one wavelength in a cube
        size : a int. Frames size, size = 1024 in case 1024*1024.
        center_scale : a float. The scale in center that we will process.
    Return:
        Sliced frames, np.array, 3 dims.
    '''
    tmp = (1-center_scale)*0.5
    res = frames[..., int(size*tmp):int(size*(1-tmp)), int(size*tmp):int(size*(1-tmp))] 
    return res

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
    #print("shape = ", ref_frames.shape) 

    for i in range(1,len(files_path)):
        #print(f)
        hd = fits.getdata(files_path[i])
        ref_frames =np.append(ref_frames, hd[..., start:end, start:end], axis=1)

    return ref_frames

# 3. process the science frames, substract the starlight
# wavelength to wavelength
def process_RDI(science_frames, ref_frames):
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

# frames to frames
def process_RDI_bis(science_frames, ref_frames):
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
            pearson_corr = -1
            indice = 0 #most relevent indicfor j in range(ref_frames):
            for j in range(rf_fr_nb):
                for wl2 in range(wave_length):
                    pearson_corr = np.corrcoef(science_frames[wl, i], ref_frames[wl2, j])[0,1]
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
    
    start_and_end(True)
    
    # argv1 : science object
    science_frames = read_file(str(sys.argv[1]))
    #print(">> Science frames type", type(science_frames), " shape=", science_frames.shape,'\n')
    
    # Step 1: get the list of files contain keyword
    all_files = get_reference_cubes(str(sys.argv[2]), "MASTER_CUBE-center")
   
    # Step 2: put the related data (all frames of the reference cubes) in np.array
    ref_frames = collect_data(all_files)
    #print("type ref_frames", type(ref_frames))
    #print(ref_frames[0].shape, "test value luminosity =", ref_frames[0, 50, 121, 122])
    #print(ref_frames[1].shape, "test value luminosity =", ref_frames[1, 50, 121, 122])
    
    # Step 3: process the science frames
    sc_frames_procced = process_RDI_ssim(slice_frame(science_frames, len(science_frames[0][0][0]), 0.25), ref_frames)
    plt.subplot(2,3,1)
    plt.imshow(sc_frames_procced[0][0], cmap=plt.cm.hot)
    plt.subplot(2,3,2)
    plt.imshow(sc_frames_procced[0][1], cmap=plt.cm.hot)
    plt.subplot(2,3,3)
    plt.imshow(sc_frames_procced[0][2], cmap=plt.cm.hot)
    plt.subplot(2,3,4)
    plt.imshow(sc_frames_procced[1][0], cmap=plt.cm.hot)
    plt.subplot(2,3,5)
    plt.imshow(sc_frames_procced[1][1], cmap=plt.cm.hot)
    plt.subplot(2,3,6)
    plt.imshow(sc_frames_procced[1][2], cmap=plt.cm.hot)
    plt.show()

    # Step 4: comparaison
    '''
    #plt.style.use('seaborn-white')
    plt.subplot(1,2,1)
    plt.imshow(ref_frames[0][50], cmap=plt.cm.hot)
    plt.subplot(1,2,2)
    plt.imshow(ref_frames[1][50], cmap=plt.cm.hot)
    plt.show()
    '''
    start_and_end(False)

