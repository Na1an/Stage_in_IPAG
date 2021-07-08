import os
import cv2
import sys
import time
import copy
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import skimage
import vip_hci as vip
import matplotlib.pyplot as plt
from astropy.io import fits
from hciplot import plot_frames, plot_cubes

# private module
from utility import *

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
        return the data of hd[0],hd type HDUList. Should be type nparray, 4 dimensions (wave_length, sc_fr_nb, x, y).
    '''
    print("path of target =",get_reference_cubes(file_path, keyword)[0])
    return fits.getdata(get_reference_cubes(file_path, keyword)[0])

# read one file and return its data
def read_wdh(file_path, keyword):
    '''
    Args:
        file_path : a string. The file path!
        keyword : a string. The keyword contains in the file name.
    Return:
        return the list of wind Drection_image, type float. Should be type nparray, 1 dimensions.
    '''

    # now, we have the csv file
    file = None
    files = get_reference_cubes(file_path, keyword) 
    for f in files:
        if f.split('.')[-1] == "csv":
            file = f
            break
    
    data = np.genfromtxt(file, delimiter=',', dtype=str)
    return np.char.replace(data[1:,5], '"','').astype(float)

# 2. Collect the data from SPHERE_DC_DATA
def collect_data(files_path, scale=0.25):
    '''
    Args:
        files_path : a list of string. files path contain keyword.
        scale : a float. The scale in center that we want process, is equal to 1/4 by default.
    Rrturn:
        ref_frames : ndarray, 4 dimensions. Return (wavelengths, nb_frames, x, y)
        ref_wdh : nbarray, 1 dims. Return the image directions for all related frames.
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

def collect_data_wdh(files_path, scale=0.25):
    '''
    Args:
        files_path : a list of string. files path contain keyword.
        scale : a float. The scale in center that we want process, is equal to 1/4 by default.
    Rrturn:
        ref_frames : ndarray, 4 dimensions. Return (wavelengths, nb_frames, x, y)
        ref_wdh : nbarray, 1 dims. Return the image directions for all related frames.
    '''
    hd = fits.getdata(files_path[0])
    ref_wdh = read_wdh(files_path[0], "Analysis_wdh_")

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
        ref_wdh = np.append(ref_wdh, read_wdh(files_path[0], "Analysis_wdh_"))
    return ref_frames, ref_wdh

def collect_frames(files_path, scale=0.25, full_output=True):
    '''
    Args:
        files_path : a list of string. files path contain keyword.
        scale : a float. The scale in center that we want process, is equal to 1/4 by default.
        full_output : a boolean. Default value is False. If it is true, we will return 3 result, if not, we inly return ref_frames. 
    Rrturn:
        ref_frames : ndarray, 4 dimensions. Return (wavelengths, nb_frames, x, y)
        ref_frames_coords : a list of tuple. [(0,0), (0,1), ...]
        ref_cube_nb_frames : a list of integer. The list contains all frame numbers of the reference cube. 
    '''

    ref_frames_coords = []
    hd = fits.getdata(files_path[0])

    # frames in the first wavelength and second wavelength
    # K1/K2, H2/H3, etc...
    wl, nb_fr, w, h = hd.shape
    tmp = (1-scale)*0.5
    size = w
    start = int(size*tmp)
    end = int(size*(1-tmp))

    ref_frames = hd[..., start:end, start:end]
    ref_frames_coords = ref_frames_coords + get_coords_of_ref_frames(0, nb_fr)
    ref_cube_nb_frames = []
    ref_cube_nb_frames.append(nb_fr)

    for i in range(1,len(files_path)):
        hd = fits.getdata(files_path[i])
        wl, nb_fr, w, h = hd.shape
        ref_frames =np.append(ref_frames, hd[..., start:end, start:end], axis=1)
        ref_frames_coords = ref_frames_coords + get_coords_of_ref_frames(i, nb_fr)
        ref_cube_nb_frames.append(nb_fr)

    if full_output is False:
        return ref_frames

    return ref_frames, ref_frames_coords, ref_cube_nb_frames

# chose the best correlated reference stars, not all
def selection(nb_best, target, refs, scale, wave_length=0):
    '''
    Args:
        nb_best : a integer. How many best ref stars we want.
        target : a numpy.ndarray. (wavelengths, nb_frames, x, y).
        refs : a list of string. All stars data we have.
        scale : a float. The scale in center region that we want process, is equal to 1/4 by default.
        wave_length : a integer. Wave length of the cube.
    Rrturn:
        res : a list of string. The int(nb_best) best chosen ref stars.
    '''
    res = {}
    # target_median is 2 dims. (256, 256)
    target_median = median_of_cube(target, wave_length)
    w, h = target_median.shape
    # create mask
    m, pxs_center = create_inner_mask(w,h,MASK_RADIUS)
    target_median_vector = np.reshape(target_median*m,(w*h))

    for i in range(len(refs)):
        # hd is 4 dims: (wl, nb frmes, x, y)
        hd = fits.getdata(refs[i])
        # ref_median is 2 dims. (256, 256)
        ref_meidan = median_of_cube(crop_frame(hd,len(hd[wave_length,i,0]),scale), wave_length)
        ref_meidan_vector = np.reshape(ref_meidan*m, (w*h))

        # maby should try cosine similarity, structural simimarity(SSIM)
        coef_corr = np.corrcoef(target_median_vector, ref_meidan_vector)
        #print(refs[i],"=",coef_corr[0,1])
        res[refs[i]] = coef_corr[0,1]

    tmp = sorted(res.items(),key = lambda r:(r[1],r[0]), reverse=True)

    print(">> There are", len(tmp), "reference stars in the library")
    print(">> we will chose", nb_best, "correlated cube to do PCA on RDI")

    res_bis = []
    for k in range(nb_best):
        (x,y) = tmp[k]
        res_bis.append(x)
        print(k,"- corrcoef value =", y)

    return res_bis

# chose the best correlated reference stars, not all
def selection_n_best(nb_best, target, refs, scale, wave_length=0):
    '''
    Args:
        nb_best : a list of integer. How many best ref stars we want.
        target : a numpy.ndarray. (wavelengths, nb_frames, x, y)
        refs : a list of string. All stars data we have.
        scale : a float. The scale in center region that we want process, is equal to 1/4 by default.
        wave_length : a integer. Wave length of the cube.
    Rrturn:
        res : a list of list of string. The int(nb_best) best chosen ref stars.
    '''
    res = {}
    # target_median is 2 dims. (256, 256)
    target_median = median_of_cube(target, wave_length)
    w, h = target_median.shape
    # create mask
    m, pxs_center = create_inner_mask(w,h,MASK_RADIUS)
    target_median_vector = np.reshape(target_median*m,(w*h))

    for i in range(len(refs)):
        # hd is 4 dims: (wl, nb frmes, x, y)
        hd = fits.getdata(refs[i])
        # ref_median is 2 dims. (256, 256)
        ref_meidan = median_of_cube(crop_frame(hd,len(hd[wave_length,i,0]),scale), wave_length)
        ref_meidan_vector = np.reshape(ref_meidan*m, (w*h))

        # maby should try cosine similarity, structural simimarity(SSIM)
        coef_corr = np.corrcoef(target_median_vector, ref_meidan_vector)
        #print(refs[i],"=",coef_corr[0,1])
        res[refs[i]] = coef_corr[0,1]

    tmp = sorted(res.items(),key = lambda r:(r[1],r[0]), reverse=True)

    print(">> There are", len(tmp), "reference stars in the library")
    res_bis = [None]*len(nb_best)
    for n in range(len(nb_best)):
        print(">> we will chose", nb_best[n], "correlated cube to do PCA on RDI")
        res_temp = []
        for k in range(nb_best[n]):
            (x,y) = tmp[k]
            res_temp.append(x)
            print(k,"- corrcoef value =", y)
        res_bis[n] = res_temp

    return res_bis

# frame based version selection the best correalted data
def selection_frame_based(target, nb_best_frame, ref_frames, wave_length=0):
    '''
    Args:
        target : a numpy.ndarray, 4 dims. The science target cube, (wavelengths, nb_frames, x, y).
        nb_best : a integer. How many best frames fo the references stars array we want for each target frame.
        ref_frames : a numpy.ndarry, 4 dims. The reference stars data we have.
        ref_frames_coords : a list of tuple. [(nth_star, nth_frame_of_star), ...].
        wave_length : a integer. Wave length of the cube.
    Rrturn:
        res : a ndarray, 3 dimensions. Return (nb_frames, x, y).
        res_coords : a 2-dims ndarray. (nb_fr_t, nb_best_frame).
    '''
    start_time = datetime.datetime.now()
    # target shape
    wl_t, nb_fr_t, w, h = target.shape
    wl_ref, nb_fr_ref, w_ref, h_ref = ref_frames.shape

    # 2 tarverses, for building the ref_frames
    cursor = 0
    res_coords = np.zeros((nb_fr_t, nb_best_frame))
    for i in range(nb_fr_t):
        #frames_coords_tmp = ref_frames_coords[cursor, cursor + ]
        tmp = {}
        for j in range(nb_fr_ref):
            # tmp[j] = np.corrcoef(np.reshape(target[wave_length, i], w*h), np.reshape(ref_frames[wave_length, j], w*h))[0,1]
            tmp[j] = np.corrcoef(np.reshape(target[wave_length, i], w*h), np.reshape(ref_frames[wave_length, j], w*h))[0,1]
        if nb_best_frame > len(tmp):
            raise Exception("!!! inside the function selection_frame_based, tmp", len(tmp),"is samller than nb_best_frame", nb_best_frame)
        res_tmp = sorted(tmp.items(),key = lambda r:(r[1],r[0]), reverse=True)[0:nb_best_frame]
        res_tmp_bis = []
        for (ind, pcc) in res_tmp:
            res_tmp_bis.append(ind)
        res_coords[i] = np.array(res_tmp_bis)

    # we will only take the unique frame, what makes our reference library smaller
    # but it should be ok considering the direction of frame (peanut..) 
    all_inds = np.unique(np.reshape(res_coords, nb_fr_t*nb_best_frame))
    
    res = np.zeros((len(all_inds), w_ref, h_ref))
    for k in range(len(all_inds)):
        res[k] = ref_frames[wave_length, int(all_inds[k])]

    end_time = datetime.datetime.now()
    print(">> frame based selection take:", end_time - start_time)
    
    return res, res_coords

# frame based version selection but with score system
def selection_frame_based_score(target, nb_best_frame, ref_frames, ref_cube_nb_frames, score, wave_length, wave_length_target):
    '''
    Args:
        target : a numpy.ndarray, 4 dims. The science target cube, (wavelengths, nb_frames, x, y).
        nb_best : a integer. How many best frames fo the references stars array we want for each target frame.
        ref_frames : a numpy.ndarry, 4 dims. The reference stars data we have.
        ref_cube_nb_frames : a list of integer. Each element is the frame number of a reference star.
        score : a integer. We will pick all the reference stars which has higher or equal score.
        wave_length : a integer. Wave length of the reference cube.
        wave_length_target : a integer. Wave length of the target cube.
    Rrturn:
        res : a ndarray, 3 dimensions. Return (nb_frames, x, y).
        res_coords : a 2-dims ndarray. (nb_fr_t, nb_best_frame)
    '''
    start_time = datetime.datetime.now()
    # target shape
    wl_t, nb_fr_t, w, h = target.shape
    wl_ref, nb_fr_ref, w_ref, h_ref = ref_frames.shape

    # score_system
    ref_scores = np.zeros((nb_fr_ref))

    for i in range(nb_fr_t):
        tmp = {}
        for j in range(nb_fr_ref):
            # tmp[j] = np.corrcoef(np.reshape(target[wave_length, i], w*h), np.reshape(ref_frames[wave_length, j], w*h))[0,1]
            tmp[j] = np.corrcoef(np.reshape(target[wave_length_target, i], w*h), np.reshape(ref_frames[wave_length, j], w*h))[0,1]
        
        if nb_best_frame > len(tmp):
            raise Exception("!!! inside the function selection_frame_based, tmp", len(tmp),"is samller than nb_best_frame", nb_best_frame)
        
        res_tmp = sorted(tmp.items(),key = lambda r:(r[1],r[0]), reverse=True)[0:nb_best_frame]
        
        for (ind, pcc) in res_tmp:
            ref_scores[ind] = ref_scores[ind] + 1

    res_coords = np.where(ref_scores>=score)
    print("res_coords.shape =", res_coords[0].shape, "res_coords.type = ", type(res_coords), " res_coords =", res_coords)
    res = ref_frames[wave_length][res_coords]
    print("res.shape =", res.shape)
    end_time = datetime.datetime.now()
    print(">> frame based selection take:", end_time - start_time)
    
    return res, get_histogram_of_ref_stars_score(res_coords[0], ref_cube_nb_frames)

# option for main : RDI
def RDI(argv, scale):
    print(">> Algo PCA is working! ")
    start_time = datetime.datetime.now()
    if(len(argv) >4):
        scale = float(argv[4])

    # 1. get target
    target_path = str(argv[2])
    #science_target = read_file(target_path, "MASTER_CUBE-center")
    science_target = read_file(target_path, "fake_disk_close")
    science_target_croped = crop_frame(science_target, len(science_target[0,0,0]), scale)
    print("Scale =", scale, "\n science target shape =", science_target_croped.shape)
    
    # 2. get the list of files in library
    ref_files = get_reference_cubes(str(argv[3]), "MASTER_CUBE-center")
    
    # Check if the taget is in the ref files, if true remove it
    ref_files = remove_target(str(argv[2]),ref_files)
    ref_files = chose_reference_files(ref_files, "H23", "IRD")
    print(">> what we have in ref_res")
    for s in ref_files:
        print(s)
    print(">> so we have",len(ref_files),"reference stars in total, all of them are on wave_length H23, type IRDIS, we will focus on wave length - H2 for instance")
    
    # Select the best correlated targets
    # count is the number of we want to chose
    count = int(argv[5])
    #ref_files = selection(count, science_target_croped, ref_files, scale, 0) # 0 is the default wave length
    ref_files = selection_n_best([3,5,7,9,11], science_target_croped, ref_files, scale, 0) # 0 is the default wave length
    
    # 3. put the related data (all frames of the reference cubes) in np.array
    ref_frames_3 = collect_data(ref_files[0], scale)
    print("ref_frames_3 shape =", ref_frames_3.shape)
    ref_frames_5 = collect_data(ref_files[1], scale)
    print("ref_frames_5 shape =", ref_frames_5.shape)
    ref_frames_7 = collect_data(ref_files[2], scale)
    print("ref_frames_7 shape =", ref_frames_7.shape)
    ref_frames_9 = collect_data(ref_files[3], scale)
    print("ref_frames_9 shape =", ref_frames_9.shape)
    ref_frames_11 = collect_data(ref_files[4], scale)
    print("ref_frames_11 shape =", ref_frames_11.shape)
    
    # get angles
    angles = read_file(str(argv[2]), "ROTATION")
    
    # get ref shape
    wl_ref, nb_fr_ref, w, h = ref_frames_11.shape
    wl = 1
    n = nb_fr_ref
    
    # create outer mask
    r_in = 15
    r_out = 125
    outer_mask, n_pxls = create_outer_mask(w,h,r_out)
    science_target_croped[wl] = science_target_croped[wl]*outer_mask
    
    number_klips = []
    for i in range(0,101,5):
        number_klips.append(i)
    number_klips[0] = 1

    '''
    # 3
    for i in number_klips:
        res_tmp = vip.pca.pca_fullfr.pca(science_target_croped[wl], -angles, ncomp= i, mask_center_px=r_in, cube_ref=ref_frames_3[wl]*outer_mask, scaling='spat-mean')
        #res_tmp = vip.pca.pca_local.pca_annular(science_target_croped[wl], -angles, cube_ref=ref_frames[wl], radius_int=r_in, asize=96, ncomp=i, scaling='spat-mean')
        path = "./K_kilp_ADI_RDI/spat_mean/fake_disk_far/3_best/"+"{0:05d}".format(i) + "_spat_mean.fits"            
        hdu = fits.PrimaryHDU(res_tmp)
        hdu.writeto(path)
        print(">>===", i, "of 3_best === fits writed to === path:", path)
    
    # 5
    for i in number_klips:
        res_tmp = vip.pca.pca_fullfr.pca(science_target_croped[wl], -angles, ncomp= i, mask_center_px=r_in, cube_ref=ref_frames_5[wl]*outer_mask, scaling='spat-mean')
        #res_tmp = vip.pca.pca_local.pca_annular(science_target_croped[wl], -angles, cube_ref=ref_frames[wl], radius_int=r_in, asize=96, ncomp=i, scaling='spat-mean')
        path = "./K_kilp_ADI_RDI/spat_mean/fake_disk_far/5_best/"+"{0:05d}".format(i) + "_spat_mean.fits"            
        hdu = fits.PrimaryHDU(res_tmp)
        hdu.writeto(path)
        print(">>===", i, "of 5_best === fits writed to === path:", path)
    '''
    # 7
    for i in number_klips:
        res_tmp = vip.pca.pca_fullfr.pca(science_target_croped[wl], -angles, ncomp= i, mask_center_px=r_in, cube_ref=ref_frames_7[wl]*outer_mask, scaling='spat-mean')
        #res_tmp = vip.pca.pca_local.pca_annular(science_target_croped[wl], -angles, cube_ref=ref_frames[wl], radius_int=r_in, asize=96, ncomp=i, scaling='spat-mean')
        path = "./K_kilp_ADI_RDI/spat_mean/fake_disk_close/7_best_bis/"+"{0:05d}".format(i) + "_spat_mean.fits"            
        hdu = fits.PrimaryHDU(res_tmp)
        hdu.writeto(path)
        print(">>===", i, "of 7_best === fits writed to === path:", path)
    
    '''
    # 9
    for i in number_klips:
        res_tmp = vip.pca.pca_fullfr.pca(science_target_croped[wl], -angles, ncomp= i, mask_center_px=r_in, cube_ref=ref_frames_9[wl]*outer_mask, scaling='spat-mean')
        #res_tmp = vip.pca.pca_local.pca_annular(science_target_croped[wl], -angles, cube_ref=ref_frames[wl], radius_int=r_in, asize=96, ncomp=i, scaling='spat-mean')
        path = "./K_kilp_ADI_RDI/spat_mean/fake_disk_far/9_best/"+"{0:05d}".format(i) + "_spat_mean.fits"            
        hdu = fits.PrimaryHDU(res_tmp)
        hdu.writeto(path)
        print(">>===", i, "of 9_best === fits writed to === path:", path)
    
    # 11
    for i in number_klips:
        res_tmp = vip.pca.pca_fullfr.pca(science_target_croped[wl], -angles, ncomp= i, mask_center_px=r_in, cube_ref=ref_frames_11[wl]*outer_mask, scaling='spat-mean')
        #res_tmp = vip.pca.pca_local.pca_annular(science_target_croped[wl], -angles, cube_ref=ref_frames[wl], radius_int=r_in, asize=96, ncomp=i, scaling='spat-mean')
        path = "./K_kilp_ADI_RDI/spat_mean/fake_disk_far/11_best/"+"{0:05d}".format(i) + "_spat_mean.fits"            
        hdu = fits.PrimaryHDU(res_tmp)
        hdu.writeto(path)
        print(">>===", i, "of 11_best === fits writed to === path:", path)
    '''
    end_time = datetime.datetime.now()
    print("PCA on RDI ", n," take", end_time - start_time)

# option for main : ADI
def ADI(argv, scale):
    print(">> Algo PCA ADI is working! ")
    start_time = datetime.datetime.now()
    if(len(sys.argv) >4):
        scale = float(sys.argv[4])

    # 1. get target
    target_path = str(sys.argv[2])
    science_target = read_file(target_path, "fake_disk_close")
    science_target_croped = crop_frame(science_target, len(science_target[0,0,0]), scale)
    print("Scale =", scale, "\n science target shape =", science_target_croped.shape)
    
    # get angles
    angles = read_file(str(sys.argv[2]), "ROTATION")
    
    # get science target shape
    wl_ref, nb_fr_ref, w, h = science_target_croped.shape
    wl = 1
    n = nb_fr_ref
    
    # create outer mask
    r_in = 15 
    r_out = 125
    outer_mask, n_pxls = create_outer_mask(w,h,r_out)
    science_target_croped[wl] = science_target_croped[wl]*outer_mask

    for i in range(1,n+1):
        #res_tmp = vip.pca.pca_fullfr.pca(science_target_croped[wl], -angles, ncomp= i, mask_center_px=MASK_RADIUS, cube_ref=ref_frames[wl], scaling='temp-mean')
        #res_tmp = vip.pca.pca_local.pca_annular(science_target_croped[wl], -angles, radius_int=118, asize=7, ncomp=i)
        #res_tmp = vip.pca.pca_annular(science_target_croped[wl], -angles, cube_ref=science_target_croped[wl],radius_int=118, asize=7, ncomp=i, scaling='temp-mean')
        res_tmp = vip.pca.pca_fullfr.pca(science_target_croped[wl], -angles, ncomp= i, mask_center_px=r_in, scaling='temp-mean')

        #path = "./K_kilp_ADI_RDI/Test_ADI/ADI_Masked" + "{0:05d}".format(i) + ".fits"
        path = "./K_kilp_ADI_RDI/ADI/fake_disk_close/" + "{0:05d}".format(i) + "_bright.fits"
        hdu = fits.PrimaryHDU(res_tmp)
        hdu.writeto(path)
        print(">>===", i, "of", n,"=== fits writed ===")

    end_time = datetime.datetime.now()
    
    print("PCA on ADI ", n," take", end_time - start_time)

# option for main : INJECTION
def INJECTION(argv, scale):
    print(">> Inject a fake companion or a disk")
        
    # 1. get target
    target_path = str(argv[2])
    science_target = read_file(target_path, "MASTER_CUBE-center")
    obj = "abc"
    if(len(argv) >4):
        scale = float(argv[4])
    if(len(argv) >5):
        obj = str(argv[5]).upper()
        print("obj =", obj)
    print("science target shape =", science_target.shape)
    
    # 2. prepare these parameters
    # get angles
    angles = read_file(target_path, "ROTATION")
    psf = read_file(target_path, "PSF_MASTER_CUBE") 
    
    # get science target shape
    wl_ref, nb_fr_ref, w, h = science_target.shape
    wl = 0
    
    # fwhm psfn
    fwhm = get_fwhm_from_psf(psf[wl])
    psfn = vip.metrics.normalize_psf(psf[wl], fwhm, size=17)
    print("psfn =", psfn.shape, "psfn.ndim =", psfn.ndim)

    # pxscale of IRDIS
    pxscale = get_pxscale()

    # make fake companion
    if obj == "PLANETE":
        fake_comp_0 = vip.metrics.cube_inject_companions(science_target[wl], psf_template=psfn, angle_list=-angles, flevel=0, plsc=pxscale, rad_dists=[100], theta=160, n_branches = 4)
        fake_comp_1 = vip.metrics.cube_inject_companions(science_target[1], psf_template=psfn, angle_list=-angles, flevel= 40, plsc=pxscale, rad_dists=[100], theta=160, n_branches = 4)
        fake_comp_2 = vip.metrics.cube_inject_companions(science_target[1], psf_template=psfn, angle_list=-angles, flevel= 1500, plsc=pxscale, rad_dists=[100], theta=160, n_branches = 4)
        print("fake companion 0 shape = ", fake_comp_0.shape)
        fake_comp = np.zeros((3, nb_fr_ref, w, h))
        fake_comp[0] = fake_comp_0
        fake_comp[1] = fake_comp_1
        fake_comp[2] = fake_comp_2
        path_fake_comp = "./K_kilp_ADI_RDI/fake_planet/fake_comp_100px.fits"

        hdu = fits.PrimaryHDU(fake_comp)
        hdu.writeto(path_fake_comp) 
    
    elif obj == "DISK":
        # far 100 pxs = 32, close 20 pxs = 145, close2 27 pxs = 115
        dstar = 32 # distance to the star in pc, the bigger the disk if more small and more close to star
        nx = 256 # number of pixels of your image in X
        ny = 256 # number of pixels of your image in Y
        # itilt = 60, ok, 0 pole-on
        itilt = 0 # inclination of your disk in degreess (0 means pole-on -> can see the full plate, 90 means edge on -> only see a line)
        # pa=70 pos1, 160 pos2 two side of disk are both on the dark side
        pa = 160 # position angle of the disk in degrees (0 means north, 90 means east)
        a = 40 # semimajoraxis of the disk in au / semimajor axis in arcsec is 80 au/80px = 1 arcsec
        fake_disk1 = vip.metrics.scattered_light_disk.ScatteredLightDisk(\
                    nx=nx,ny=ny,distance=dstar,\
                    itilt=itilt,omega=0,pxInArcsec=pxscale,pa=pa,\
                    density_dico={'name':'2PowerLaws','ain':4,'aout':-4,\
                    'a':a,'e':0.0,'ksi0':1.,'gamma':2.,'beta':1.},\
                    spf_dico={'name':'HG', 'g':0., 'polar':False})
        fake_disk1_map = fake_disk1.compute_scattered_light()
        fake_disk1_map = fake_disk1_map/np.max(fake_disk1_map)
        ds9 = vip.Ds9Window()
        ds9.display(fake_disk1_map)
        
        # add fake disk to science target
        scaling_factor = float(argv[7])
        cube_fakeddisk = vip.metrics.cube_inject_fakedisk(fake_disk1_map*scaling_factor ,angle_list=angles,psf=psfn)
        ds9 = vip.Ds9Window()
        ds9.display(cube_fakeddisk[0])
        #print(">>>> cube fake disk :", cube_fakeddisk[0:50, 0:50])
        # only want center
        start = int(w*(1-scale)/2)
        end = int(start+w*scale)

        w_t, nb_fr_t, w_t, h_t = science_target.shape
        fake_comp_differents = np.zeros((3, nb_fr_t, w_t, h_t))
        for i in range(3):
            fake_comp_differents[i] = science_target[0]

        #science_target[0,:,start:end,start:end] = science_target[0,:,start:end,start:end] + cube_fakeddisk*5
        #cube_fakeddisk = vip.metrics.cube_inject_fakedisk(fake_disk1_map,angle_list=angles,psf=psfn)
        #science_target[1,:,start:end,start:end] = science_target[1,:,start:end,start:end] + cube_fakeddisk*10000
        fake_comp_differents[0,:,start:end,start:end] = science_target[0,:,start:end,start:end] + cube_fakeddisk*0
        fake_comp_differents[1,:,start:end,start:end] = science_target[0,:,start:end,start:end] + cube_fakeddisk*10
        fake_comp_differents[2,:,start:end,start:end] = science_target[0,:,start:end,start:end] + cube_fakeddisk*120

        path_fake_disk = "./K_kilp_ADI_RDI/fake_planet/"+str(argv[6])
        hdu = fits.PrimaryHDU(fake_comp_differents)
        hdu.writeto(path_fake_disk)
    else:
        print("No such object inject option")
    
    # display
    #ds9 = vip.Ds9Window()
    #ds9.display(fake_comp[0])

# option for main : SCAL
def SCAL(argv, scale):
    print(">> Analysis scalings effect! ")
    start_time = datetime.datetime.now()
    
    if(len(argv) >4):
        scale = float(argv[4])

    scalings = ["temp-mean", "spat-mean", "temp-standard","spat-standard"]
    
    # 1. get target
    target_path = str(argv[2])
    science_target = read_file(target_path, "MASTER_CUBE-center")
    science_target_croped = crop_frame(science_target, len(science_target[0,0,0]), scale)
    print("Scale =", scale, "\n science target shape =", science_target_croped.shape)
   
    # 2. get the list of files in library
    ref_files = get_reference_cubes(str(argv[3]), "MASTER_CUBE-center")
    
    # Check if the taget is in the ref files, if true remove it
    ref_files = remove_target(str(argv[2]),ref_files)
    print(">> what we have in ref_res")
    for s in ref_files:
        print(s)

    # Select the best correlated targets
    # count is the number of we want to chose
    count = 3
    ref_files = selection(count, science_target_croped, ref_files, scale, 0) # 0 is the default wave length

    # 3. put the related data (all frames of the reference cubes) in np.array
    ref_frames = collect_data(ref_files, scale)
    print("ref_frames shape =", ref_frames.shape)

    # get angles
    angles = read_file(str(argv[2]), "ROTATION")
    
    # get science target shape
    wl_ref, nb_fr_ref, w, h = ref_frames.shape
    wl = 0
    n = nb_fr_ref       
    
    # create outer mask
    r_in = 118
    r_out = (w/2)
    
    outer_mask, n_pxls = create_outer_mask(w,h,r_out)
    #science_target_croped[wl] = science_target_croped[wl] * outer_mask
    
    for s in scalings:
        for i in range(1, n+1):
            res_tmp = vip.pca.pca_fullfr.pca(science_target_croped[wl], -angles, ncomp= i, mask_center_px=r_in, cube_ref=ref_frames[wl], scaling=s)
            path = "./K_kilp_ADI_RDI/RDI_only_big_inner/" +s+"/{0:05d}".format(i) + "_with_mask_on_ref.fits"
            hdu = fits.PrimaryHDU(res_tmp)
            hdu.writeto(path)
            print(">>===", i, "of", n,"=== fits writed ===")

    end_time = datetime.datetime.now()
    
    print("PCA Scals ", n," take", end_time - start_time)

# option for main : SAM
def SAM(argv, scale):
    print(">> Analysis spat-mean vs spat-annular-mean! ")
    start_time = datetime.datetime.now()
    
    if(len(argv) >4):
        scale = float(argv[4])

    # 1. get target
    target_path = str(argv[2])
    #science_target_origin = read_file(target_path, "MASTER_CUBE-center")
    #science_target_origin_croped = crop_frame(science_target_origin, len(science_target_origin[0,0,0]), scale)

    science_target = read_file(target_path, "fake_disk_close")
    science_target_croped = crop_frame(science_target, len(science_target[0,0,0]), scale)
    #store_to_fits(science_target_croped[0]-science_target_origin_croped[0], "./K_kilp_ADI_RDI/1206_origin_minus_fake_disk_01.fits")
    #exit()
    # science target scale 
    wl_tar, nb_fr_tar, w_tar, h_tar = science_target_croped.shape
    print("Scale =", scale, "\n science target shape =", science_target_croped.shape)
    # 2. get the list of files in library
    ref_files = get_reference_cubes(str(argv[3]), "MASTER_CUBE-center")

    # Check if the taget is in the ref files, if true remove it
    ref_files = remove_target(str(argv[2]), ref_files)
    ref_files = chose_reference_files(ref_files, "H23", "IRD")
    print(">> what we have in ref_res")
    for s in ref_files:
        print(s)
    print(">>> we have", len(ref_files), "reference stars in our library")

    # Select the best correlated targets
    # count is the number of we want to chose
    count = int(argv[5])
    #ref_files = selection(count, science_target_croped, ref_files, scale, 0) # 0 is the default wave length
    print("Scale =", scale, "\n science target croped shape =", science_target_croped.shape)
    ref_files = selection_n_best([3,5,7,9,11], science_target_croped, ref_files, scale, 0)

    # 3. put the related data (all frames of the reference cubes) in np.array
    #ref_frames = collect_data(ref_files, scale)
    #print("ref_frames shape =", ref_frames.shape)
    ref_frames_3 = collect_data(ref_files[0], scale)
    print("ref_frames_3 shape =", ref_frames_3.shape)
    ref_frames_5 = collect_data(ref_files[1], scale)
    print("ref_frames_5 shape =", ref_frames_5.shape)
    ref_frames_7 = collect_data(ref_files[2], scale)
    print("ref_frames_7 shape =", ref_frames_7.shape)
    ref_frames_9 = collect_data(ref_files[3], scale)
    print("ref_frames_9 shape =", ref_frames_9.shape)
    ref_frames_11 = collect_data(ref_files[4], scale)
    print("ref_frames_11 shape =", ref_frames_11.shape)

    # get angles
    angles = read_file(str(argv[2]), "ROTATION")
    
    # get science target shape
    wl_ref, nb_fr_ref, w, h = science_target_croped.shape
    wl = 0
    n = 100
    
    # create outer mask
    r_in = 15 
    r_out = 125 
    
    outer_mask, n_pxls = create_outer_mask(w,h,r_out)
    science_target_croped[wl] = science_target_croped[wl] * outer_mask
    #store_to_fits(science_target_croped[wl], "./K_kilp_ADI_RDI/1306_fake_disk_10_before_sep_mean.fits")

    print("start remove separation mean from science_target")
    sep_mean = remove_separation_mean_from_cube(science_target_croped[wl])
    #store_to_fits(sep_mean, "./K_kilp_ADI_RDI/1306_fake_disk_10_sep_mean.fits")
    #store_to_fits(science_target_croped[wl], "./K_kilp_ADI_RDI/1306_fake_disk_10_after_sep_mean.fits")
    
    #exit()

    print("star remove separation mean from ref_frames")
    remove_separation_mean_from_cube(ref_frames_3[wl])
    remove_separation_mean_from_cube(ref_frames_5[wl])
    remove_separation_mean_from_cube(ref_frames_7[wl])
    remove_separation_mean_from_cube(ref_frames_9[wl])
    remove_separation_mean_from_cube(ref_frames_11[wl])

    number_klips = []
    for i in range(0,101,5):
        number_klips.append(i)
    number_klips[0] = 1

    # 3
    for i in number_klips:
        res_tmp = vip.pca.pca_fullfr.pca(science_target_croped[wl], -angles, ncomp= i, mask_center_px=r_in, cube_ref=ref_frames_3[wl]*outer_mask, scaling=None)
        path = "./K_kilp_ADI_RDI/spat_annular_mean/fake_disk_close/3_best/"+"{0:05d}".format(i) + "_spat_annular.fits"
        hdu = fits.PrimaryHDU(res_tmp)
        hdu.writeto(path)
        print(">>===", i, "of 3 best=== fits writed === path :", path)
    
    # 5
    for i in number_klips:
        res_tmp = vip.pca.pca_fullfr.pca(science_target_croped[wl], -angles, ncomp= i, mask_center_px=r_in, cube_ref=ref_frames_5[wl]*outer_mask, scaling=None)
        path = "./K_kilp_ADI_RDI/spat_annular_mean/fake_disk_close/5_best/"+"{0:05d}".format(i) + "_spat_annular.fits"
        hdu = fits.PrimaryHDU(res_tmp)
        hdu.writeto(path)
        print(">>===", i, "of 5 best=== fits writed === path :", path)
    
    # 7
    for i in number_klips:
        res_tmp = vip.pca.pca_fullfr.pca(science_target_croped[wl], -angles, ncomp= i, mask_center_px=r_in, cube_ref=ref_frames_7[wl]*outer_mask, scaling=None)
        path = "./K_kilp_ADI_RDI/spat_annular_mean/fake_disk_close/7_best/"+"{0:05d}".format(i) + "_spat_annular.fits"
        hdu = fits.PrimaryHDU(res_tmp)
        hdu.writeto(path)
        print(">>===", i, "of 7 best=== fits writed === path :", path)

    # 9
    for i in number_klips:
        res_tmp = vip.pca.pca_fullfr.pca(science_target_croped[wl], -angles, ncomp= i, mask_center_px=r_in, cube_ref=ref_frames_9[wl]*outer_mask, scaling=None)
        path = "./K_kilp_ADI_RDI/spat_annular_mean/fake_disk_close/9_best/"+"{0:05d}".format(i) + "_spat_annular.fits"
        hdu = fits.PrimaryHDU(res_tmp)
        hdu.writeto(path)
        print(">>===", i, "of 9 best=== fits writed === path :", path)

    # 11
    for i in number_klips:
        res_tmp = vip.pca.pca_fullfr.pca(science_target_croped[wl], -angles, ncomp=i, mask_center_px=r_in, cube_ref=ref_frames_11[wl]*outer_mask, scaling=None)
        path = "./K_kilp_ADI_RDI/spat_annular_mean/fake_disk_close/11_best/"+"{0:05d}".format(i) + "_spat_annular.fits"
        hdu = fits.PrimaryHDU(res_tmp)
        hdu.writeto(path)
        print(">>===", i, "of 11 best=== fits writed === path :", path)
    
    end_time = datetime.datetime.now()
    
    print("PCA SAM ", n," take", end_time - start_time)

# option for main : WDH - wind driven halo
def WDH(argv, scale):
    print(">> Analysis spat-mean vs spat-annular-mean! ")
    start_time = datetime.datetime.now()
    
    if(len(argv) >4):
        scale = float(argv[4])

    # 1. get target
    target_path = str(argv[2])
    #science_target = read_file(target_path, "MASTER_CUBE-center")
    science_target = read_file(target_path, "fake_comp_added_disk_01")
    science_target_croped = crop_frame(science_target, len(science_target[0,0,0]), scale)
    # science target scale 
    wl_tar, nb_fr_tar, w_tar, h_tar = science_target_croped.shape
    print("Scale =", scale, "\n science target shape =", science_target_croped.shape)
    # 2. get the list of files in library
    ref_files = get_reference_cubes(str(argv[3]), "MASTER_CUBE-center")
    
    # Check if the taget is in the ref files, if true remove it
    ref_files = remove_target(str(argv[2]), ref_files)
    print(">> what we have in ref_res")
    for s in ref_files:
        print(s)

    # Select the best correlated targets
    # count is the number of we want to chose
    count = int(argv[5])
    #ref_files = selection(count, science_target_croped, ref_files, scale, 0) # 0 is the default wave length

    # 3. put the related data (all frames of the reference cubes) in np.array
    #ref_frames, ref_wdh = collect_data(ref_files, scale)
    #print("ref_frames shape =", ref_frames.shape)

    # get angles
    angles = read_file(str(argv[2]), "ROTATION")
    
    # get science target shape
    wl_ref, nb_fr_ref, w, h = science_target_croped.shape
    wl = 0
    n = 50 
    
    # 4. create outer mask, inner mask with VIP
    r_in = 15 
    r_out = 125 
    
    outer_mask, n_pxls = create_outer_mask(w,h,r_out)
    science_target_croped[wl] = science_target_croped[wl] * outer_mask
   
    # 5. Wind driven halo
    # get the image directions : penault angle for each frame of science target
    target_wind_angle = read_wdh(argv[2], "Analysis_wdh_")
    
    print("start attenuate the wdh influence")
    
    path = "./K_kilp_ADI_RDI/wdh/origin.fits"
    hdu = fits.PrimaryHDU(science_target_croped[wl])
    hdu.writeto(path)
    
    print("start remove separation mean from science_target")
    #remove_separation_mean_from_cube(science_target_croped[0])
    wdh_influence = attenuate_wdh_influence_from_cube(science_target_croped[0], target_wind_angle, detail=True)
    
    path = "./K_kilp_ADI_RDI/wdh/wdh_influence.fits"
    hdu = fits.PrimaryHDU(wdh_influence)
    hdu.writeto(path)

    path = "./K_kilp_ADI_RDI/wdh/after_attenuate.fits"
    hdu = fits.PrimaryHDU(science_target_croped[wl])
    hdu.writeto(path)
    print("we end here for now")
    exit()

    '''
    # 5. remove SAM in the option SAM, but we don't do this in WDH
    print("start remove separation mean from science_target")
    remove_separation_mean_from_cube(science_target_croped[0])
    print("star remove separation mean from ref_frames")
    remove_separation_mean_from_cube(ref_frames[0])
    '''

    # 5. WDH process
    
    # 6. start the process PCA
    for i in range(1, 51):
        res_tmp = vip.pca.pca_fullfr.pca(science_target_croped[wl], -angles, ncomp= i, mask_center_px=r_in, cube_ref=ref_frames[wl]*outer_mask, scaling=None)
        path = "./K_kilp_ADI_RDI/spat-annular-bis/" +str(count)+"_best/{0:05d}".format(i) + "spat_annular.fits"
        hdu = fits.PrimaryHDU(res_tmp)
        hdu.writeto(path)
        print(">>===", i, "of", n,"=== fits writed === path :", path)

    end_time = datetime.datetime.now()
    
    print("PCA WDH", n," take", end_time - start_time)

# option for main : frame based RDI
def RDI_frame(argv, scale):
    print(">> Algo PCA is working! ")
    start_time = datetime.datetime.now()
    if(len(argv) >4):
        scale = float(argv[4])

    key_word_target = "MASTER_CUBE-center"
    if(len(argv) >6):
        key_word_target = str(argv[6])
    print(">>> key word of target is:", key_word_target)
    
    # 1. get target
    target_path = str(argv[2])
    print(">>> target_path =", target_path)
    science_target = read_file(target_path, key_word_target)
    #science_target = read_file(target_path, "fake_disk_close_5_bis")
    science_target_croped = crop_frame(science_target, len(science_target[0,0,0]), scale)
    print("Scale =", scale, "\n science target shape =", science_target_croped.shape)
    
    # 2. get the list of files in library
    ref_files = get_reference_cubes(str(argv[3]), "MASTER_CUBE-center")
    
    # Check if the taget is in the ref files, if true remove it
    ref_files = remove_target(str(argv[2]),ref_files)
    ref_files = chose_reference_files(ref_files, "H23", "IRD")
    print(">> what we have in ref_res")
    for s in ref_files:
        print(s)
    print(">> so we have",len(ref_files),"reference stars in total, all of them are on wave_length H23, type IRDIS")
    
    # take all reference cubes
    ref_frames, ref_frames_coords, ref_cube_nb_frames = collect_frames(ref_files, scale)
    print(">> in total, there are", len(ref_frames_coords), "frames in our reference librarys")
    
    #print(">> ref frames_coords =", ref_frames_coords)
    # 20 89
    # nb_best_frame = 100, we have 281 frames in our reference library
    # nb_best_frame = 200, we have 440 frames in our reference library
    nb_best_frame = 200

    if(len(argv)>5):
        nb_best_frame = int(argv[5])

    ref_frames_selected, target_ref_coords = selection_frame_based_score(science_target_croped, nb_best_frame, ref_frames, ref_cube_nb_frames, 0, wave_length=0, wave_length_target=1)

    print("ref_frames.shape =", ref_frames.shape)
    print("target_ref_coords.shape =", target_ref_coords.shape)
    print("target_ref_coords =", target_ref_coords, " sum=", target_ref_coords.sum())
    
    # take ref_files and target_ref_coords, produce a dictionary
    dict_ref_in_target = get_dict(ref_files, target_ref_coords)
    print(dict_ref_in_target)

    plt.bar(dict_ref_in_target.keys(), dict_ref_in_target.values())
    plt.xticks(rotation=30)
    plt.title("How many frames are used for the reference stars", fontsize="18")
    plt.xlabel("Name of reference star used", fontsize="16")
    plt.ylabel("Number of frames", fontsize="16")
    plt.savefig("./K_kilp_ADI_RDI/ref_frames_histogram"+datetime.datetime.now().strftime('%m-%d_%H_%M_%S')+".png")
    #plt.show()

    # get angles
    angles = read_file(str(argv[2]), "ROTATION")

    # get ref shape
    nb_fr_ref, w, h = ref_frames_selected.shape
    wl = 0
    #wl_target = wl
    wl_target = 1
    n = nb_fr_ref
    
    # create outer mask
    r_in = 15
    r_out = 125
    outer_mask, n_pxls = create_outer_mask(w,h,r_out)
    science_target_vip = science_target_croped[wl_target]*outer_mask
    #science_target_croped[3] = science_target_croped[3]*outer_mask
    
    number_klips = []
    for i in range(0,266,20):
        number_klips.append(i)
    number_klips[0] = 1

    res_path = "./K_kilp_ADI_RDI/res_0907_presentation/spat_mean/disk_20pxs/pos1/"
    if(len(argv) >7):
        res_path = str(argv[7]) + "/"
    print(">>> We will put our result here:", res_path)
    
    scal = None
    #scal = "spat-mean"
    print(">> scaling =", scal)

    for i in number_klips:
        res_tmp = vip.pca.pca_fullfr.pca(science_target_vip, -angles, ncomp=i, mask_center_px=r_in, cube_ref=ref_frames_selected*outer_mask, scaling=scal)
        #res_tmp = vip.pca.pca_local.pca_annular(science_target_croped[wl], -angles, cube_ref=ref_frames[wl], radius_int=r_in, asize=96, ncomp=i, scaling='spat-mean')
        path = res_path+"{0:05d}".format(i) + "_disk_20pxs.fits"            
        hdu = fits.PrimaryHDU(res_tmp)
        hdu.writeto(path)
        print(">>===", i, "of ", number_klips[-1],"RDI  === fits writed to === path:", path)
    
    end_time = datetime.datetime.now()
    print("PCA on RDI frame based ", n," take", end_time - start_time)

def RDI_frame_bis(argv, scale):
    print(">> Algo PCA is working! ")
    start_time = datetime.datetime.now()
    if(len(argv) >4):
        scale = float(argv[4])

    key_word_target = "MASTER_CUBE-center"
    if(len(argv) >6):
        key_word_target = str(argv[6])
    print(">>> key word of target is:", key_word_target)
    
    # 1. get target
    target_path = str(argv[2])
    print(">>> target_path =", target_path)
    science_target = read_file(target_path, key_word_target)
    #science_target = read_file(target_path, "fake_disk_close_5_bis")
    science_target_croped = crop_frame(science_target, len(science_target[0,0,0]), scale)
    print("Scale =", scale, "\n science target shape =", science_target_croped.shape)
    
    # 2. get the list of files in library
    ref_files = get_reference_cubes(str(argv[3]), "MASTER_CUBE-center")
    
    # Check if the taget is in the ref files, if true remove it
    ref_files = remove_target(str(argv[2]),ref_files)
    ref_files = chose_reference_files(ref_files, "H23", "IRD")
    print(">> what we have in ref_res")
    for s in ref_files:
        print(s)
    print(">> so we have",len(ref_files),"reference stars in total, all of them are on wave_length H23, type IRDIS")
    
    # take all reference cubes
    ref_frames, ref_frames_coords, ref_cube_nb_frames = collect_frames(ref_files, scale)
    print(">> in total, there are", len(ref_frames_coords), "frames in our reference librarys")
    
    #print(">> ref frames_coords =", ref_frames_coords)
    # 20 89
    # nb_best_frame = 100, we have 281 frames in our reference library
    # nb_best_frame = 200, we have 440 frames in our reference library
    #nb_best_frame = [50, 100, 150, 200, 250]
    nb_best_frame = [500, 1000]

    # store the results
    res_path_fichier = "./K_kilp_ADI_RDI/res_0907_presentation/"
    print(">> We will put our result here:", res_path_fichier)
    res_path_fichier_real = "./K_kilp_ADI_RDI/res_0907_presentation_real/"
    
    sc_target_for_sam = []

    for nb in range(len(nb_best_frame)):
        print(">>> we will chose " + str(nb_best_frame[nb]) + " best correlated frames for each frame")
        ref_frames_selected, target_ref_coords = selection_frame_based_score(science_target_croped, nb_best_frame[nb], ref_frames, ref_cube_nb_frames, 0, wave_length=0, wave_length_target=1)

        print("ref_frames_selected.shape =", ref_frames_selected.shape)
        print("target_ref_coords.shape =", target_ref_coords.shape)
        print("target_ref_coords =", target_ref_coords, " sum=", target_ref_coords.sum())
        
        # take ref_files and target_ref_coords, produce a dictionary
        dict_ref_in_target = get_dict(ref_files, target_ref_coords)
        print(dict_ref_in_target)
        d_keys, d_values = list_of_tuple_to_2_list(sorted(dict_ref_in_target.items(),key = lambda r:(r[1],r[0]), reverse=True))
        plt.bar(d_keys, d_values)
        plt.xticks(rotation=25)
        plt.title("How many frames are used for the reference stars " + str(target_ref_coords.sum()), fontsize="18")
        plt.xlabel("Name of reference star used", fontsize="16")
        plt.ylabel("Number of frames", fontsize="16")
        plt.savefig("./K_kilp_ADI_RDI/ref_frames_histogram"+datetime.datetime.now().strftime('%m-%d_%H_%M_%S')+".png")
        #plt.show()
        # get angles
        angles = read_file(str(argv[2]), "ROTATION")

        # get ref shape
        nb_fr_ref, w, h = ref_frames_selected.shape
        wl = 0
        # wl_target = 0 -> deal with the raw data
        # wl_target = 1 -> deal with the fake data
        #wl_target = wl
        wl_target = 1
        n = nb_fr_ref
        
        # create outer mask
        r_in = 15
        r_out = 125
        outer_mask, n_pxls = create_outer_mask(w,h,r_out)
        science_target_vip_raw = science_target_croped[0]*outer_mask
        science_target_vip = science_target_croped[wl_target]*outer_mask
        #science_target_croped[3] = science_target_croped[3]*outer_mask
        
        number_klips = []
        for i in range(0, nb_fr_ref, 20):
            number_klips.append(i)
        number_klips[0] = 1
        print(">>> nb_best_frames =", nb_best_frame[nb], "number_klips =", number_klips)

        res_path = res_path_fichier + "frame_" + "{0:03d}".format(nb_best_frame[nb]) + "/"
        print(">>> We will put our result here:", res_path)
        res_path_real = res_path_fichier_real + "frame_" + "{0:03d}".format(nb_best_frame[nb]) + "/"
        print(">>> We will put our result here, real copy:", res_path_real)

        where_to_store = "/disk_close_27pxs/pos2/"

        for i in number_klips:
            
            ###############
            # fake target #
            ###############

            # non scale
            res_tmp = vip.pca.pca_fullfr.pca(science_target_vip, -angles, ncomp=i, mask_center_px=r_in, cube_ref=ref_frames_selected*outer_mask, scaling=None)
            path = res_path+"no_scale" + where_to_store +"{0:05d}".format(i) + "_fake.fits"            
            hdu = fits.PrimaryHDU(res_tmp)
            hdu.writeto(path)
            print(">>> = scaling is None ===", i, "of ", number_klips[-1],"RDI  === fits writed to === path:", path)

            '''
            # spat-mean
            res_tmp = vip.pca.pca_fullfr.pca(science_target_vip, -angles, ncomp=i, mask_center_px=r_in, cube_ref=ref_frames_selected*outer_mask, scaling='spat-mean')
            path = res_path+ "spat_mean" + where_to_store +"{0:05d}".format(i) + "_fake.fits"            
            hdu = fits.PrimaryHDU(res_tmp)
            hdu.writeto(path)
            print(">>> = scaling is spat-mean === ", i, "of ", number_klips[-1],"RDI  === fits writed to === path:", path)
            '''
            ###############
            # raw target #
            ###############

            # non scale
            res_tmp = vip.pca.pca_fullfr.pca(science_target_vip_raw, -angles, ncomp=i, mask_center_px=r_in, cube_ref=ref_frames_selected*outer_mask, scaling=None)
            path = res_path_real+"no_scale" + where_to_store +"{0:05d}".format(i) + "_real.fits"            
            hdu = fits.PrimaryHDU(res_tmp)
            hdu.writeto(path)
            print(">>> = scaling is None ===", i, "of ", number_klips[-1],"RDI  === fits writed to === path:", path)

            '''
            # spat-mean
            res_tmp = vip.pca.pca_fullfr.pca(science_target_vip_raw, -angles, ncomp=i, mask_center_px=r_in, cube_ref=ref_frames_selected*outer_mask, scaling='spat-mean')
            path = res_path_real+ "spat_mean" + where_to_store +"{0:05d}".format(i) + "_raw.fits"            
            hdu = fits.PrimaryHDU(res_tmp)
            hdu.writeto(path)
            print(">>> = scaling is spat-mean === ", i, "of ", number_klips[-1],"RDI  === fits writed to === path:", path)
            '''
        """
        sc_target_for_sam = copy.deepcopy(science_target_vip)
        sc_target_for_sam_raw = copy.deepcopy(science_target_vip_raw)
        print("start remove separation mean from sc_target_for_sam")
        remove_separation_mean_from_cube(sc_target_for_sam)
        remove_separation_mean_from_cube(sc_target_for_sam_raw)

        print("star remove separation mean from ref_frames")
        remove_separation_mean_from_cube(ref_frames_selected)

        
        for i in number_klips:

            ###############
            # fake target #
            ###############

            # non scale
            res_tmp = vip.pca.pca_fullfr.pca(sc_target_for_sam, -angles, ncomp=i, mask_center_px=r_in, cube_ref=ref_frames_selected*outer_mask, scaling=None)
            path = res_path+"spat_annular_mean" + where_to_store +"{0:05d}".format(i) + "_fake.fits"            
            hdu = fits.PrimaryHDU(res_tmp)
            hdu.writeto(path)
            print(">>> = scaling is spat-annumar-mean ===", i, "of ", number_klips[-1],"RDI  === fits writed to === path:", path)

            ###############
            # raw target #
            ###############

            # non scale
            res_tmp = vip.pca.pca_fullfr.pca(sc_target_for_sam_raw, -angles, ncomp=i, mask_center_px=r_in, cube_ref=ref_frames_selected*outer_mask, scaling=None)
            path = res_path_real+"spat_annular_mean" + where_to_store +"{0:05d}".format(i) + "_raw.fits"            
            hdu = fits.PrimaryHDU(res_tmp)
            hdu.writeto(path)
            print(">>> = scaling is spat-annumar-mean ===", i, "of ", number_klips[-1],"RDI  === fits writed to === path:", path)
        """

    end_time = datetime.datetime.now()
    print("PCA on RDI frame based ", n," take", end_time - start_time)

# option for main : RDI_SCORES
def RDI_scores(argv, scale):
    print(">> Algo PCA is working! ")
    start_time = datetime.datetime.now()
    if(len(argv) >4):
        scale = float(argv[4])

    key_word_target = "MASTER_CUBE-center"
    if(len(argv) >6):
        key_word_target = str(argv[6])
    print(">>> key word of target is:", key_word_target)
    
    # 1. get target
    target_path = str(argv[2])
    print(">>> target_path =", target_path)
    science_target = read_file(target_path, key_word_target)
    #science_target = read_file(target_path, "fake_disk_close_5_bis")
    science_target_croped = crop_frame(science_target, len(science_target[0,0,0]), scale)
    print("Scale =", scale, "\n science target shape =", science_target_croped.shape)
    
    # 2. get the list of files in library
    ref_files = get_reference_cubes(str(argv[3]), "MASTER_CUBE-center")
    
    # Check if the taget is in the ref files, if true remove it
    ref_files = remove_target(str(argv[2]),ref_files)
    ref_files = chose_reference_files(ref_files, "H23", "IRD")
    print(">> what we have in ref_res")
    for s in ref_files:
        print(s)
    print(">> so we have",len(ref_files),"reference stars in total, all of them are on wave_length H23, type IRDIS")
    
    # take all reference cubes
    ref_frames, ref_frames_coords, ref_cube_nb_frames = collect_frames(ref_files, scale)
    print(">> in total, there are", len(ref_frames_coords), "frames in our reference librarys")

    nb_best_frame = 150

    # store the results
    res_path_fichier = "./K_kilp_ADI_RDI/res_0707_fake/"
    print(">> We will put our result here:", res_path_fichier)
    res_path_fichier_real = "./K_kilp_ADI_RDI/res_0707_real/"

    scores = [1, 5, 10]

    for score in scores:

        #sc_target_for_sam = []

        print(">>> we will chose " + str(nb_best_frame) + " best correlated frames for each frame")
        print(">>> we will use score =" + str(score) + " for selecting the reference frames")
        ref_frames_selected, target_ref_coords = selection_frame_based_score(science_target_croped, nb_best_frame, ref_frames, ref_cube_nb_frames, score=score, wave_length=0, wave_length_target=1)

        print("ref_frames_selected.shape =", ref_frames_selected.shape)
        print("target_ref_coords.shape =", target_ref_coords.shape)
        print("target_ref_coords =", target_ref_coords, " sum=", target_ref_coords.sum())
        
        '''
        # take ref_files and target_ref_coords, produce a dictionary
        dict_ref_in_target = get_dict(ref_files, target_ref_coords)
        print(dict_ref_in_target)
        d_keys, d_values = list_of_tuple_to_2_list(sorted(dict_ref_in_target.items(),key = lambda r:(r[1],r[0]), reverse=True))
        plt.bar(d_keys, d_values)
        plt.xticks(rotation=25)
        plt.title("How many frames are used for the reference stars " + str(target_ref_coords.sum()), fontsize="18")
        plt.xlabel("Name of reference star used", fontsize="16")
        plt.ylabel("Number of frames", fontsize="16")
        plt.savefig("./K_kilp_ADI_RDI/ref_frames_histogram"+datetime.datetime.now().strftime('%m-%d_%H_%M_%S')+".png")
        '''

        # get angles
        angles = read_file(str(argv[2]), "ROTATION")

        # get ref shape
        nb_fr_ref, w, h = ref_frames_selected.shape
        wl = 0
        # wl_target = 0 -> deal with the raw data
        # wl_target = 1 -> deal with the fake data
        #wl_target = wl
        wl_target = 1
        n = nb_fr_ref
        
        # create outer mask
        r_in = 15
        r_out = 125
        outer_mask, n_pxls = create_outer_mask(w,h,r_out)
        science_target_vip_raw = science_target_croped[0]*outer_mask
        science_target_vip = science_target_croped[wl_target]*outer_mask
        #science_target_croped[3] = science_target_croped[3]*outer_mask
        
        number_klips = []
        for i in range(0, nb_fr_ref, 20):
            number_klips.append(i)
        number_klips[0] = 1
        print(">>> nb_best_frames =", nb_best_frame, "number_klips =", number_klips)
        where_to_store = "disk_close_27pxs/"
        res_path = res_path_fichier + where_to_store + "score_" + "{0:03d}".format(score) + "/pos2/"
        print(">>> We will put our result here:", res_path)
        res_path_real = res_path_fichier_real + where_to_store + "score_" + "{0:03d}".format(score) + "/pos2/"
        print(">>> We will put our result here:", res_path_real)

        for i in number_klips:
            
            ###############
            # fake target #
            ###############

            # non scale
            res_tmp = vip.pca.pca_fullfr.pca(science_target_vip, -angles, ncomp=i, mask_center_px=r_in, cube_ref=ref_frames_selected*outer_mask, scaling=None)
            path = res_path + "{0:05d}".format(i) + "_fake.fits"            
            hdu = fits.PrimaryHDU(res_tmp)
            hdu.writeto(path)
            print(">>> = scaling is None ===", i, "of ", number_klips[-1],"RDI  === fits writed to === path:", path)

            ###############
            # raw target #
            ###############

            # non scale
            res_tmp = vip.pca.pca_fullfr.pca(science_target_vip_raw, -angles, ncomp=i, mask_center_px=r_in, cube_ref=ref_frames_selected*outer_mask, scaling=None)
            path = res_path_real + "{0:05d}".format(i) + "_real.fits"            
            hdu = fits.PrimaryHDU(res_tmp)
            hdu.writeto(path)
            print(">>> = scaling is None ===", i, "of ", number_klips[-1],"RDI  === fits writed to === path:", path)

    end_time = datetime.datetime.now()
    print("PCA on RDI frame based ", n," take", end_time - start_time)

# Algo_RDI, the algorithm we may apply on the cobrex server
def Algo_RDI(target_path, ref_path, scale, wl=0, n_corr=150, score=1, scaling=None, res_path=None):
    '''
    Args:
        target_path : a string. The directory path where to tarversal is for getting the science target.
        ref_path : a string. The directory path of the reference library.
        scale : a float. The scale in center region that we want process, is equal to 1/4 by default.
        wl : a integer. The wave length we will focus on.
        n_corr : a integer. The best correalted frames we will pick from the reference library for each frame of the science target.
        score : a integer, shouble be positive (>=1). Use the score to chose reference frames in the library.
        scaling : a string. The optiong for vip function. None/spat-mean/temp-mean...
        res_path : a string. The directory path where we will store the result. 
    Rrturn:
        None.
    '''
    print(">> Algo PCA is working! ")
    start_time = datetime.datetime.now()

    key_word_target = "MASTER_CUBE-center"
    print(">>> key word of target is:", key_word_target)
    
    # 1. get target
    print(">>> target_path =", target_path)
    science_target = read_file(target_path, key_word_target)
    science_target_croped = crop_frame(science_target, len(science_target[0,0,0]), scale)
    print("Scale =", scale, "\n science target shape =", science_target_croped.shape)
    
    # 2. get the list of files in library
    ref_files = get_reference_cubes(ref_path, "MASTER_CUBE-center")
    
    # Check if the taget is in the ref files, if true remove it
    ref_files = remove_target(target_path,ref_files)
    ref_files = chose_reference_files(ref_files, "H23", "IRD")
    print(">> what we have in ref_res")
    for s in ref_files:
        print(s)
    print(">> so we have",len(ref_files),"reference stars in total, all of them are on wave_length H23, type IRDIS")
    
    # take all reference cubes
    ref_frames, ref_frames_coords, ref_cube_nb_frames = collect_frames(ref_files, scale)
    print(">> in total, there are", len(ref_frames_coords), "frames in our reference librarys")

    nb_best_frame = n_corr

    print(">>> we will chose " + str(nb_best_frame) + " best correlated frames for each frame")
    print(">>> we will use score =" + str(score) + " for selecting the reference frames")
    ref_frames_selected, target_ref_coords = selection_frame_based_score(science_target_croped, nb_best_frame, ref_frames, ref_cube_nb_frames, score=score, wave_length=wl, wave_length_target=wl)

    print("ref_frames_selected.shape =", ref_frames_selected.shape)
    print("target_ref_coords.shape =", target_ref_coords.shape)
    print("target_ref_coords =", target_ref_coords, " sum=", target_ref_coords.sum())
    
    # take ref_files and target_ref_coords, produce a dictionary
    dict_ref_in_target = get_dict(ref_files, target_ref_coords)
    print(dict_ref_in_target)
    d_keys, d_values = list_of_tuple_to_2_list(sorted(dict_ref_in_target.items(),key = lambda r:(r[1],r[0]), reverse=True))
    plt.bar(d_keys, d_values)
    plt.xticks(rotation=25)
    plt.title("How many frames are used for the reference stars " + str(target_ref_coords.sum()), fontsize="18")
    plt.xlabel("Name of reference star used", fontsize="16")
    plt.ylabel("Number of frames", fontsize="16")
    plt.savefig("./K_kilp_ADI_RDI/ref_frames_histogram"+datetime.datetime.now().strftime('%m-%d_%H_%M_%S')+".png")
    #plt.show()
    # get angles
    angles = read_file(target_path, "ROTATION")

    # get ref shape
    nb_fr_ref, w, h = ref_frames_selected.shape
    # wl_target = 0 -> deal with the raw data
    # wl_target = 1 -> deal with the fake data
    #wl_target = wl
    wl_target = wl
    number_klips = int((nb_fr_ref//10)*10)
    
    # create outer mask
    r_in = 15
    r_out = 125
    outer_mask, n_pxls = create_outer_mask(w,h,r_out)
    science_target_vip_raw = science_target_croped[0]*outer_mask
    science_target_vip = science_target_croped[wl_target]*outer_mask
    
    print(">>> nb_best_frames =", nb_best_frame, "number_klips =", number_klips)
    print(">>> We will put our result here:", res_path)

    # non scaling = None
    res_tmp = vip.pca.pca_fullfr.pca(science_target_vip, -angles, ncomp=number_klips, mask_center_px=r_in, cube_ref=ref_frames_selected*outer_mask, scaling=None)
    path = res_path + "/" + "{0:05d}".format(number_klips) + "_fake.fits"            
    hdu = fits.PrimaryHDU(res_tmp)
    hdu.writeto(path)
    print(">>> = vip scaling is None === fits writed to === path:", path)

    end_time = datetime.datetime.now()
    print("PCA on RDI frame based ", n_corr," take", end_time - start_time)

if __name__ == "__main__":
    start_and_end_program(True)
    print("vip.version :", vip.__version__)
    
    # Default scale for focus area of the frame
    scale = 0.25
    
    # algo option
    opt = str(sys.argv[1]).upper()
    
    if opt == "RDI":
        # product the processed images    
        RDI(sys.argv, scale) 
    
    elif opt == "ADI":
        # product the processed images    
        ADI(sys.argv, scale) 
    
    elif opt == "INJECTION":
        # inject a fake planet or disk
        INJECTION(sys.argv, scale)

    elif opt == "SCAL":
        # SCAL : analysis scalings effects 
        SCAL(sys.argv, scale)        
    
    elif opt == "SAM":
        # SAM : spat-annular-mean
        SAM(sys.argv, scale)

    elif opt == "WDH":
        # WDH : wind driven halo
        #tmp = FrameTempRadian(7)
        #tmp.print_coords()
        WDH(sys.argv, scale)
        #target_wind_angle = read_wdh(sys.argv[2], "Analysis_wdh_")
        #print(target_wind_angle)
        '''
        t1 = np.ones((7,7))
        
        # assignment the values to frame template
        l1 = [(1,5), (2,4), (5,1), (4,2)]
        l2 = [(1,4), (2,5), (4,1), (5,2)]
        l3 = [(1,3), (2,3), (3,1), (3,2), (3,4), (3,5), (4,3), (5,3)]
        l4 = [(2,2), (4,4)]
        give_value(l1, t1, 150)
        give_value(l2, t1, 120)
        give_value(l3, t1, 20)
        give_value(l4, t1, 3)
        print(">> origin frame")
        print(t1)
        print("\n")
        m = tmp.wdh_influence(t1, 45, True)
        print("\n>> wdh influence")
        print(m)
        print(">> origin - wdh influence")
        print(t1-m)

        fig, ax = plt.subplots(1,3)
        ax[0].set_title("origin")
        im1 = ax[0].imshow(t1, origin='lower')
        #fig.colorbar(im1, ax=ax)
        ax[1].set_title("wdh_influence")
        ax[1].imshow(m, origin='lower')
        ax[2].set_title("origin - wdh_influence")
        ax[2].imshow(t1-m, origin='lower')
        #fig.colorbar(images[0], ax=axs, orientation='horizontal', fraction=.1)
        plt.show()
        '''

    elif opt == "CONTRAST":
        print(">> Test raw contrast")
        
        # 1. get target
        argv = sys.argv
        target_path = str(argv[2])
        science_target = read_file(target_path, "MASTER_CUBE-center")
        science_target_croped = crop_frame(science_target, len(science_target[0,0,0]), scale)
        obj = "abc"
        if(len(argv) >4):
            scale = float(argv[4])
        if(len(argv) >5):
            obj = str(argv[5]).upper()
            print("obj =", obj)
        print("science target shape =", science_target.shape)
        
        # 2. prepare these parameters
        # get angles
        angles = read_file(target_path, "ROTATION")
        psf = read_file(target_path, "PSF_MASTER_CUBE") 
        
        # get science target shape
        wl_ref, nb_fr_ref, w, h = science_target.shape
        wl = 0
        
        # fwhm psfn
        fwhm = get_fwhm_from_psf(psf[wl])
        
        psfn, fwhm_flux, fwhm = vip.metrics.normalize_psf(psf[wl], fwhm, size=17, full_output=True)
        print("psfn.shape =", psfn.shape, "psfn.ndim =", psfn.ndim)
        raw_contrast = get_raw_contrast(fwhm_flux[0], median_of_cube(science_target_croped, wl=0))

        plt.title("The contrast of the science cube (median)", fontsize="20")
        plt.plot(raw_contrast)
        plt.ylabel("raw contrast (mean)", fontsize="18")
        plt.xlabel("separation (pxs)", fontsize="18")
        plt.show()
        '''
        df = pd.DataFrame(raw_contrast, columns=["Contrast"])
        df["Radius"] = df.index
        #print(df.head())
        #exit()
        for i in range(1,nb_fr_ref,1):
            raw_contrast = get_raw_contrast(fwhm, science_target_croped[wl,i])
            df_temp = pd.DataFrame(raw_contrast, columns=["Contrast"])
            df_temp["Radius"] = df_temp.index
            df = pd.concat([df,df_temp], axis=0, ignore_index=False)
        
        print(df)
        #df.index.name = "Frames"
        data = pd.DataFrame(raw_contrast)
        #data.columns.name = "Radius"
        data.index.name = "Frames"
        print(data.head())
        data.index = data.index * 0.01227
        ax = sns.relplot(kind='line', data=data)
        plt.show()
        '''
    
    elif opt == "RDI_FRAME":
        RDI_frame(sys.argv, scale)

    elif opt == "RDI_FRAME_BIS":
        RDI_frame_bis(sys.argv, scale)

    elif opt == "RDI_SCORES":
        RDI_scores(sys.argv, scale) 
    
    elif opt == "ALGO_RDI":
        Algo_RDI(str(sys.argv[2]), str(sys.argv[3]), float(sys.argv[4]), res_path=str(sys.argv[5]))

    else:
        print("No such option")

    start_and_end_program(False)
