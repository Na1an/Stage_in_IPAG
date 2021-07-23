"""
Reduce a IRD_SCIENCE_REDUCED_MASTER_CUBE

@Author : Yuchen BAI
@Date   : 17/07/2021
@Contact: yuchenbai@hotmail.com
"""

import argparse
import warnings
import numpy as np
import vip_hci as vip
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning

###########
# setting #
###########

warnings.simplefilter('ignore', category=AstropyWarning)

############
# function #
############

# frame based version selection but with score system
def selection_frame_based_score(target, nb_best_frame, ref_frames, ref_cube_nb_frames, score, wave_length):
    '''
    Args:
        target : a numpy.ndarray, 4 dims. The science target cube, (wavelengths, nb_frames, x, y).
        nb_best : a integer. How many best frames fo the references stars array we want for each target frame.
        ref_frames : a numpy.ndarry, 4 dims. The reference stars data we have.
        ref_cube_nb_frames : a list of integer. Each element is the frame number of a reference star.
        score : a integer. We will pick all the reference stars which has higher or equal score.
        wave_length : a integer. Wave length of the reference cube.
    Rrturn:
        res : a ndarray, 3 dimensions. Return (nb_frames, x, y).
    '''
    # target shape
    wl_t, nb_fr_t, w, h = target.shape
    wl_ref, nb_fr_ref, w_ref, h_ref = ref_frames.shape

    # score_system
    ref_scores = np.zeros((nb_fr_ref))

    for i in range(nb_fr_t):
        tmp = {}
        for j in range(nb_fr_ref):
            # tmp[j] = np.corrcoef(np.reshape(target[wave_length, i], w*h), np.reshape(ref_frames[wave_length, j], w*h))[0,1]
            tmp[j] = np.corrcoef(np.reshape(target[wave_length, i], w*h), np.reshape(ref_frames[wave_length, j], w*h))[0,1]
        
        if nb_best_frame > len(tmp):
            raise Exception("!!! inside the function selection_frame_based, tmp", len(tmp),"is samller than nb_best_frame", nb_best_frame)
        
        res_tmp = sorted(tmp.items(),key = lambda r:(r[1],r[0]), reverse=True)[0:nb_best_frame]
        
        for (ind, pcc) in res_tmp:
            ref_scores[ind] = ref_scores[ind] + 1

    res_coords = np.where(ref_scores>=score)
    print("res_coords.shape =", res_coords[0].shape, "res_coords.type = ", type(res_coords), " res_coords =", res_coords)
    res = ref_frames[wave_length][res_coords]
    print("res.shape =", res.shape)
    
    return res

#############
# main code #
#############
print("######### Start program : ird_rdi_reduce.py #########")
parser = argparse.ArgumentParser(description="For build the Pearson Correlation Coefficient matrix for the science target and the reference master cubes, we need the following parameters.")
# file .sof whille contain the CORRELATION_MATRIX, SCIENCE TARGET, PARALLACTIC ANGLE
parser.add_argument("sof", help="file name of the sof file", type=str)
parser.add_argument("--score", help="which decide how we choose the reference frame (>=1)", type=int, default=1)
parser.add_argument("--n_corr", help="the number of best correalted frames for each frame of science target", type=int, default=150)
parser.add_argument("--ncomp",help="number of principal components to remove (5 by default)", type=int, default=5)
parser.add_argument("--wl_channels", help="Spectral channel to use (to choose between 0 for channel 0, 1 for channel 1, 2 for both channels)", type=int, choices=[0,1,2], default=0)
parser.add_argument("--scaling", help="scaling for the PCA (to choose between 0 for spat-mean, 1 for spat-standard, 2 for temp-mean, 3 for temp-standard or 4 for None)",\
                    type=int, choices=[0,1,2,3,4], default=0)
# handle args
args = parser.parse_args()

# sof
sofname=args.sof

# --score
score = args.score

# --n_corr
n_corr = args.n_corr

# --wl_channels
dico_conversion_wl_channels = {0 : [0], 1 : [1], 2 : [0,1]}
wl_channels = dico_conversion_wl_channels[args.wl_channels]
nb_wl = len(wl_channels)

# start the program
if type(crop_size) not in [np.int64,np.int,int]:
    crop_size = int(crop_size)
if crop_size>1024:
    crop_size = 1024
elif crop_size<=21:
    crop_size=21
    print('Warning cropsize<=21 ! Value set to {0:d}'.format(crop_size))

# Step-1 Reading the sof file
data=np.loadtxt(sofname,dtype=str)
filenames=data[:,0]
datatypes=data[:,1]

corr_matrix_path = filenames[np.where(datatypes == "IRD_CORR_MATRIX")[0]]
if len(corr_matrix_path) < 1:
    raise Exception("The sof file must contain exactly one IRD_CORR_MATRIX file")

anglenames = filenames[np.where(datatypes == 'IRD_SCIENCE_PARA_ROTATION_CUBE')[0]]
if len(anglenames) != 1: 
    raise Exception('The sof file must contain exactly one IRD_SCIENCE_PARA_ROTATION_CUBE file')

# Step-2 take science cube
corr_matrix = fits.getdata(corr_matrix_path)
corr_matrix_header = fits.getheader(corr_matrix_path)
science_cube = fits.getdata(corr_matrix_header["PATH_TAR"])
science_header = fits.getheader(corr_matrix_header["PATH_TAR"])

nb_science_wl, nb_science_frames, ny, nx = science_cube.shape

# the number of reference cube we have
nb_ref_cubes = int(corr_matrix_header["NB_REF_CUBES"])


print("> science cube :", science_cube_name)
print("> reference cuebs :", reference_cube_names)



'''
derotation_angles = fits.getdata(anglenames[0])
if len(derotation_angles) != nb_science_frames:
    raise Exception('The science cube IRD_SCIENCE_REDUCED_MASTER_CUBE contains {0:d} frames while the list IRD_SCIENCE_PARA_ROTATION_CUBE contains {1:d} angles'.format(nb_science_frames,len(derotation_angles)))
'''
# sort reference cube names
reference_cube_names.sort()
tmp_cube = fits.getdata(reference_cube_names[0])

# collect data, then we have reference frames
border_l = ny//2 - crop_size//2
border_r = ny//2 + crop_size//2 + 1
ref_frames = tmp_cube[..., border_l:border_r, border_l:border_r]
ref_nb_frames = []
ref_nb_frames.append(len(tmp_cube[0]))

for name in reference_cube_names[1:]:
    tmp_cube = fits.getdata(name)
    ref_nb_frames.append(len(tmp_cube[0]))
    ref_frames = np.append(ref_frames, tmp_cube[..., border_l:border_r, border_l:border_r], axis=1)

print("> ref_frames.shape =", ref_frames.shape)
print("> ref_nb_frames =", ref_nb_frames)
wl_ref, nb_ref_frames, ref_x, ref_y = ref_frames.shape

# correlation matrix
res = np.zeros((nb_wl, nb_science_frames, nb_ref_frames))
science_cube_croped = science_cube[..., border_l:border_r, border_l:border_r]
for w in range(nb_wl):
    wl = wl_channels[w]
    for i in range(nb_science_frames):
        for j in range(nb_ref_frames):
            res[wl, i, j] = np.corrcoef(np.reshape(science_cube_croped[wl, i], ref_x*ref_y), np.reshape(ref_frames[wl, j], ref_x*ref_y))[0,1]

file_name = "pcc_matrix.fits"
print("> The result will be stored in :", file_name)
hdu = fits.PrimaryHDU(data=res, header=science_header)
hdu.writeto(file_name)

print("######### End #########")