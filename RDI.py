import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from matplotlib.widgets import Slider

def read_file(file_path):
    hd = fits.open(file_path)
    return hd[0].data

def display_rotaion_angles(data):
    print("frame numbers = ",len(data))
    for i in range(len(data)):
        print(str(i)+"th rotation =", data[i])

if __name__ == "__main__":
    data = read_file(str(sys.argv[1]))
    display_rotaion_angles(data)
    
