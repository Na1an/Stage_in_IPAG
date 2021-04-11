import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from matplotlib.widgets import Slider

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
    return hd[0].data

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

# travesal the SPHERE_DC_DATA and get all the reference master cubes
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
    # argv1 : science object
    science_frames = read_file(str(sys.argv[1]))
    print(type(science_frames))

    # argv2 : the SPHERE_DC/repository path
    all_files = get_reference_cubes(str(sys.argv[2]), "MASTER_CUBE-center")
    for f in all_files:
        print(f)
