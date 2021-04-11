import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from matplotlib.widgets import Slider

# read one file and return its data
def read_file(file_path):
    hd = fits.open(file_path)
    # print("filename =", hd.fileinfo(0)["filename"].split("/")[-1])
    hd.info()
    return hd[0].data

# display the list of rotation angles
def display_rotaion_angles(data):
    print("frame numbers = ",len(data))
    for i in range(len(data)):
        print(str(i)+"th rotation =", data[i])

# travesal the SPHERE_DC_DATA and get all the reference master cubes
def get_reference_cubes(repository_path, keyword):
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

if __name__ == "__main__":
    '''
    data = read_file(str(sys.argv[1]))
    display_rotaion_angles(data)
    '''
    science_frames = read_file(str(sys.argv[1]))
    print(type(science_frames))

    all_files = get_reference_cubes(str(sys.argv[2]), "MASTER_CUBE-center")
    for f in all_files:
        print(f)
