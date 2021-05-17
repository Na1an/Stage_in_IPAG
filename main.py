import os
import cv2
import time
import numpy as np
import skimage
import vip_hci as vip
import matplotlib.pyplot as plt

from hciplot import plot_frames, plot_cubes

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

if __name__ == "__main__":
    start_and_end_program(True)
    print("vip.version :", vip.__version__)

    start_and_end_program(False)
