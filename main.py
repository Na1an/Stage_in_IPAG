import os
import cv2
import sys
import time
import datetime
import numpy as np
import skimage
import vip_hci as vip
import matplotlib.pyplot as plt
from astropy.io import fits
from hciplot import plot_frames, plot_cubes

# private module
from utility import *

# Global constant
MASK_RADIUS = 32

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
    return fits.getdata(get_reference_cubes(file_path, keyword)[0])

# chose the best correlated reference stars, not all
def selection(nb_best, target, refs, scale, wave_length=0):
    '''
    Args:
        nb_best : a integer. How many best ref stars we want.
        target : a numpy.ndarray. (wavelengths, nb_frames, x, y)
        refs : a list of string. All stars data we have.
        wave_length : a integer. Wave length of the cube.
    Rrturn:
        res : a list of string. The int(nb_best) best chosen ref stars.
    '''
    res = {}
    # target_median is 2 dims. (256, 256)
    target_median = median_of_cube(target, wave_length)
    w, h = target_median.shape
    # create mask
    m, pxs_center = create_mask(w,h,MASK_RADIUS)
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

# distance between two points
def distance(x1, y1, x2, y2):
    '''
    Args:
        x1 : an integer. object 1 - coordinate X
        y1 : an integer. object 1 - coordinate Y
        x2 : an integer. object 2 - coordinate X
        y2 : an integer. object 2 - coordinate Y
    Return:
        res : an integer. The distance between two points.
    '''

    return ((x1-x2)**2+(y1-y2)**2)**0.5

# A mask, cover the center of image, inner mask
def create_inner_mask(w, h, radius=MASK_RADIUS):
    '''
    Args:
        w : an integer. The weight of image.
        h : an integer. The height of image.
        radius : an integer. The radius of mask.
    Return:
        res : a numpy.ndarray, 2 dimens. Ex. (256, 256) but the center is all 0.
    '''
    count = 0
    res = np.full((w,h),True)
    x = w//2
    y = h//2
    for i in range(w):
        for j in range(h):
            if distance(i, j, x, y) <= radius:
                res[i,j] = False
                count = count + 1
    return res, count

# A mask, cover the center of image, outer mask
def create_outer_mask(w, h, radius):
    '''
    Args:
        w : an integer. The weight of image.
        h : an integer. The height of image.
        radius : an integer. The radius of mask.
    Return:
        res : a numpy.ndarray, 2 dimens. Ex. (256, 256) but the center is all 0.
    '''
    count = 0
    res = np.full((w,h),True)
    x = w//2
    y = h//2
    for i in range(w):
        for j in range(h):
            if distance(i, j, x, y) >= radius:
                res[i,j] = False
                count = count + 1
    return res, count

# rotate the frames in a cube
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

# store the median of the cube and rotate -- Not work
def median_of_cube_test(science_frames, rotations, scale):
    '''
    Args:
        science_frames : a numpy.ndarray. (wavelengths, nb_frames, x, y)
        rotations : a numpy.ndarry. The angles of ratations
        scale : a float. The scale of study area in the center of frames.
    Return:
        res : a numpy.ndarray, 3 dimensions. Ex. (2 wavelengths, 256, 256).
    '''
    wave_length, sc_fr_nb, w, h = science_frames.shape
    f_median = np.zeros((wave_length, w, h))
    res = np.zeros((wave_length, int(w*scale), int(h*scale)))

    for wl in range(wave_length):
        for i in range(w):
            for j in range(h):
                f_median[wl, i, j] = np.median(science_frames[wl, :, i, j])

    for wl in range(wave_length):
        for n in range(sc_fr_nb):
            res[wl] = res[wl] + rotate((science_frames[wl, n] - f_median[wl]), rotations[n])

    return None

# chose the best correlated reference stars, not all
def selection(nb_best, target, refs, scale, wave_length=0):
    '''
    Args:
        nb_best : a integer. How many best ref stars we want.
        target : a numpy.ndarray. (wavelengths, nb_frames, x, y)
        refs : a list of string. All stars data we have.
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

# detection
def rdi_detection():
    return None

if __name__ == "__main__":
    start_and_end_program(True)
    print("vip.version :", vip.__version__)
    
    # Default scale for focus area of the frame
    scale = 0.25
    
    # algo option
    opt = str(sys.argv[1]).upper()

    
    if opt == "RDI":
        # product the processed images    
        print(">> Algo PCA is working! ")
        start_time = datetime.datetime.now()
        if(len(sys.argv) >4):
            scale = float(sys.argv[4])

        # 1. get target
        target_path = str(sys.argv[2])
        science_target = read_file(target_path, "fake_comp")
        science_target_croped = crop_frame(science_target, len(science_target[0,0,0]), scale)
        print("Scale =", scale, "\n science target shape =", science_target_croped.shape)
        
        # 2. get the list of files in library
        ref_files = get_reference_cubes(str(sys.argv[3]), "MASTER_CUBE-center")
        
        # Check if the taget is in the ref files, if true remove it
        ref_files = remove_target(str(sys.argv[2]),ref_files)
        print(">> what we have in ref_res")
        for s in ref_files:
            print(s)

        # Select the best correlated targets
        # count is the number of we want to chose
        count = int(sys.argv[5])
        ref_files = selection(count, science_target_croped, ref_files, scale, 0) # 0 is the default wave length

        # 3. put the related data (all frames of the reference cubes) in np.array
        ref_frames = collect_data(ref_files, scale)
        print("ref_frames shape =", ref_frames.shape)
        
        # get angles
        angles = read_file(str(sys.argv[2]), "ROTATION")
        
        # get ref shape
        wl_ref, nb_fr_ref, w, h = ref_frames.shape
        wl = 0
        n = nb_fr_ref
        
        # create outer mask
        r_in = 115
        r_out = 128
        outer_mask, n_pxls = create_outer_mask(w,h,r_out)
        science_target_croped[wl] = science_target_croped[wl]*outer_mask
        
        for i in range(1,31):
            res_tmp = vip.pca.pca_fullfr.pca(science_target_croped[wl], -angles, ncomp= i, mask_center_px=r_in, cube_ref=ref_frames[wl]*outer_mask, scaling='spat-mean')
            #res_tmp = vip.pca.pca_local.pca_annular(science_target_croped[wl], -angles, cube_ref=ref_frames[wl], radius_int=r_in, asize=96, ncomp=i, scaling='spat-mean')
            #path = "./K_kilp_ADI_RDI/RDI_res_"+str(count)+"/RDI_Masked" + "{0:05d}".format(i) + ".fits"
            path = "./K_kilp_ADI_RDI/outer_mask_115_128/" + "{0:05d}".format(i) + ".fits"
            hdu = fits.PrimaryHDU(res_tmp)
            hdu.writeto(path)
            print(">>===", i, "of", n,"=== fits writed to === path:", path)
        end_time = datetime.datetime.now()
        print("PCA on RDI ", n," take", end_time - start_time)
    
    elif opt == "ADI":
        # product the processed images    
        print(">> Algo PCA ADI is working! ")
        start_time = datetime.datetime.now()
        if(len(sys.argv) >4):
            scale = float(sys.argv[4])

        # 1. get target
        target_path = str(sys.argv[2])
        science_target = read_file(target_path, "MASTER_CUBE-center")
        science_target_croped = crop_frame(science_target, len(science_target[0,0,0]), scale)
        print("Scale =", scale, "\n science target shape =", science_target_croped.shape)
        
        # get angles
        angles = read_file(str(sys.argv[2]), "ROTATION")
        
        # get science target shape
        wl_ref, nb_fr_ref, w, h = science_target_croped.shape
        wl = 0
        n = nb_fr_ref
        
        # create outer mask
        r_in = 118
        r_out = 125
        outer_mask, n_pxls = create_outer_mask(w,h,r_out)
        science_target_croped[wl] = science_target_croped[wl]*outer_mask

        for i in range(1,n+1):
            #res_tmp = vip.pca.pca_fullfr.pca(science_target_croped[wl], -angles, ncomp= i, mask_center_px=MASK_RADIUS, cube_ref=ref_frames[wl], scaling='temp-mean')
            #res_tmp = vip.pca.pca_local.pca_annular(science_target_croped[wl], -angles, radius_int=118, asize=7, ncomp=i)
            #res_tmp = vip.pca.pca_annular(science_target_croped[wl], -angles, cube_ref=science_target_croped[wl],radius_int=118, asize=7, ncomp=i, scaling='temp-mean')
            res_tmp = vip.pca.pca_fullfr.pca(science_target_croped[wl], -angles, ncomp= i, mask_center_px=r_in, scaling='temp-mean')

            #path = "./K_kilp_ADI_RDI/Test_ADI/ADI_Masked" + "{0:05d}".format(i) + ".fits"
            path = "./K_kilp_ADI_RDI/fake_res_adi/ADI_Masked" + "{0:05d}".format(i) + ".fits"
            hdu = fits.PrimaryHDU(res_tmp)
            hdu.writeto(path)
            print(">>===", i, "of", n,"=== fits writed ===")

        end_time = datetime.datetime.now()
        
        print("PCA on ADI ", n," take", end_time - start_time)
    
    elif opt == "INJECTION":
        # inject a fake planet
        print(">> Inject a fake planet")
        
        # 1. get target
        target_path = str(sys.argv[2])
        science_target = read_file(target_path, "MASTER_CUBE-center")
        #science_target_croped = crop_frame(science_target, len(science_target[0,0,0]), scale)
        print("Scale =", scale, "\n science target shape =", science_target.shape)
        
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
        
        # pxscale of IRDIS
        pxscale = get_pxscale()

        # make fake companion
        fake_comp_0 = vip.metrics.cube_inject_companions(science_target[wl], psf_template=psfn, angle_list=-angles, flevel=80, plsc=pxscale, rad_dists=[25, 36, 51, 72, 97], theta=70, n_branches = 1)
        fake_comp_1 = vip.metrics.cube_inject_companions(science_target[1], psf_template=psfn, angle_list=-angles, flevel= 80, plsc=pxscale, rad_dists=[25, 36, 51, 72, 97], theta=70, n_branches = 1)
        print("fake companion 0 shape = ", fake_comp_0.shape)
        
        # display
        #ds9 = vip.Ds9Window()
        #ds9.display(fake_comp[0])
        fake_comp = np.zeros((wl_ref, nb_fr_ref, w, h))
        fake_comp[0] = fake_comp_0
        fake_comp[1] = fake_comp_1
        path_fake_comp = "./fake_planet/fake_comp01.fits"

        hdu = fits.PrimaryHDU(fake_comp)
        hdu.writeto(path_fake_comp)
    
    elif opt == "SCAL":
        
        print(">> Analysis scalings effect! ")
        start_time = datetime.datetime.now()
        
        if(len(sys.argv) >4):
            scale = float(sys.argv[4])

        scalings = ["temp-mean", "spat-mean", "temp-standard","spat-standard"]
        
        # 1. get target
        target_path = str(sys.argv[2])
        science_target = read_file(target_path, "MASTER_CUBE-center")
        science_target_croped = crop_frame(science_target, len(science_target[0,0,0]), scale)
        print("Scale =", scale, "\n science target shape =", science_target_croped.shape)
       
        # 2. get the list of files in library
        ref_files = get_reference_cubes(str(sys.argv[3]), "MASTER_CUBE-center")
        
        # Check if the taget is in the ref files, if true remove it
        ref_files = remove_target(str(sys.argv[2]),ref_files)
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
        angles = read_file(str(sys.argv[2]), "ROTATION")
        
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
    elif opt == "SAM":
        # SAM : spat-annular-mean
        


    else:
        print("No such option")

    start_and_end_program(False)
