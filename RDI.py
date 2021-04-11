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
def collect_data(files_path):
    '''
    Args:
        files_path : a list of string. files path contain keyword.
    Rrturn:
        res : a list of float. Return ...
    '''
    hd = fits.getdata(files_path[0])
    '''
    for f in range(1,len(files_path)):
        #print(f)
        hd.append(fits.open(f))
        hd.info()
    '''
    print(">> type hd =", type(hd), " shape", hd.shape)
    print(">> len(hd) = ", len(hd))
    res = []
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
    print(">> Science frames type", type(science_frames),'\n')

    # Step 1: get the list of files contain keyword
    all_files = get_reference_cubes(str(sys.argv[2]), "MASTER_CUBE-center")
   
    # Step 2: put the related data (all frames of the reference cubes) in np.array
    reference_frames = collect_data(all_files) 

    start_and_end(False)

