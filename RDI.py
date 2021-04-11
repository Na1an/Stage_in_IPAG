import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from matplotlib.widgets import Slider

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
        size : a int. Frames sizei, size = 1024 in case 1024*1024.
        center_scale : a float. The scale in center that we will process.
    Return:
        Sliced frames, np.array, 3 dims.
    '''
    tmp = (1-center_scale)*0.5
    res = frames[:, int(size*tmp):int(size*(1-tmp)), size*tmp:size*(1-tmp)] 
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
        res : a list of float. Return ...
    '''
    hd = fits.getdata(files_path[0])
    # frames in the first wavelength and second wavelength
    # K1/K2, H2/H3, etc...
    tmp = (1-scale)*0.5
    size = len(hd[0][0])
    start = int(size*tmp)
    end = int(size*(1-tmp))
    
    frames_fir_wl = hd[0][..., start:end, start:end]
    frames_sec_wl = hd[1][..., start:end, start:end]

    #print(">> len(first) = ", len(frames_fir_wl))

    for i in range(1,len(files_path)):
        #print(f)
        hd = fits.getdata(files_path[i])
        frames_fir_wl = np.append(frames_fir_wl, hd[0][..., start:end, start:end], axis=0)
        frames_sec_wl = np.append(frames_sec_wl, hd[1][..., start:end, start:end], axis=0)
        #print(">> len(first) = ", len(frames_fir_wl))
    
    return frames_fir_wl, frames_sec_wl

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
    print(">> Science frames type", type(science_frames),'\n')

    # Step 1: get the list of files contain keyword
    all_files = get_reference_cubes(str(sys.argv[2]), "MASTER_CUBE-center")
   
    # Step 2: put the related data (all frames of the reference cubes) in np.array
    ref_frames_wl1, ref_frames_wl2 = collect_data(all_files)
    print(ref_frames_wl1.shape, "test value luminosity =", ref_frames_wl1[50][121][122])
    print(ref_frames_wl2.shape, "test value luminosity =", ref_frames_wl2[50][121][122])
    
    #plt.style.use('seaborn-white')
    plt.subplot(1,2,1)
    plt.imshow(ref_frames_wl1[50], cmap=plt.cm.hot)
    plt.subplot(1,2,2)
    plt.imshow(ref_frames_wl2[50], cmap=plt.cm.hot)
    plt.show()
    start_and_end(False)

