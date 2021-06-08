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
    #science_target = read_file(target_path, "MASTER_CUBE-center")
    science_target = read_file(target_path, "fake_comp02")
    science_target_croped = crop_frame(science_target, len(science_target[0,0,0]), scale)
    # science target scale 
    wl_tar, nb_fr_tar, w_tar, h_tar = science_target_croped.shape
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
    count = int(argv[5])
    #ref_files = selection(count, science_target_croped, ref_files, scale, 0) # 0 is the default wave length

    # 3. put the related data (all frames of the reference cubes) in np.array
    #ref_frames = collect_data(ref_files, scale)
    #print("ref_frames shape =", ref_frames.shape)

    # get angles
    angles = read_file(str(argv[2]), "ROTATION")
    
    # get science target shape
    wl_ref, nb_fr_ref, w, h = science_target_croped.shape
    wl = 0
    n = 50 
    
    # create outer mask
    r_in = 15 
    r_out = 125 
    
    outer_mask, n_pxls = create_outer_mask(w,h,r_out)
    science_target_croped[wl] = science_target_croped[wl] * outer_mask
    
    
    print("start remove separation mean from science_target")
    remove_separation_mean_from_cube(science_target_croped[0])
    path = "./K_kilp_ADI_RDI/target_after_substract_mean.fits"
    hdu = fits.PrimaryHDU(science_target_croped[wl])
    hdu.writeto(path)
    print("star remove separation mean from ref_frames")
    remove_separation_mean_from_cube(ref_frames[0])

    for i in range(1, 51):
        res_tmp = vip.pca.pca_fullfr.pca(science_target_croped[wl], -angles, ncomp= i, mask_center_px=r_in, cube_ref=ref_frames[wl]*outer_mask, scaling=None)
        path = "./K_kilp_ADI_RDI/spat-annular-bis/" +str(count)+"_best/{0:05d}".format(i) + "spat_annular.fits"
        hdu = fits.PrimaryHDU(res_tmp)
        hdu.writeto(path)
        print(">>===", i, "of", n,"=== fits writed === path :", path)

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
    science_target = read_file(target_path, "fake_comp02")
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
    ref_files = selection(count, science_target_croped, ref_files, scale, 0) # 0 is the default wave length

    # 3. put the related data (all frames of the reference cubes) in np.array
    ref_frames = collect_data(ref_files, scale)
    print("ref_frames shape =", ref_frames.shape)

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
        science_target = read_file(target_path, "fake_comp_added_disk")
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
        wl = 1
        n = nb_fr_ref
        
        # create outer mask
        r_in = 15
        r_out = 125
        outer_mask, n_pxls = create_outer_mask(w,h,r_out)
        science_target_croped[wl] = science_target_croped[wl]*outer_mask
        
        for i in range(1,51):
            res_tmp = vip.pca.pca_fullfr.pca(science_target_croped[wl], -angles, ncomp= i, mask_center_px=r_in, cube_ref=ref_frames[wl]*outer_mask, scaling='spat-mean')
            #res_tmp = vip.pca.pca_local.pca_annular(science_target_croped[wl], -angles, cube_ref=ref_frames[wl], radius_int=r_in, asize=96, ncomp=i, scaling='spat-mean')
            path = "./K_kilp_ADI_RDI/disk/"+"{0:05d}".format(i) + ".fits"            
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
        science_target = read_file(target_path, "fake")
        science_target_croped = crop_frame(science_target, len(science_target[0,0,0]), scale)
        print("Scale =", scale, "\n science target shape =", science_target_croped.shape)
        
        # get angles
        angles = read_file(str(sys.argv[2]), "ROTATION")
        
        # get science target shape
        wl_ref, nb_fr_ref, w, h = science_target_croped.shape
        wl = 0
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
            path = "./K_kilp_ADI_RDI/fake_planet/ADI_Masked" + "{0:05d}".format(i) + ".fits"
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
        obj = "abc"
        if(len(sys.argv) >4):
            scale = float(sys.argv[4])
        if(len(sys.argv) >5):
            obj = str(sys.argv[5]).upper()
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
            fake_comp_0 = vip.metrics.cube_inject_companions(science_target[wl], psf_template=psfn, angle_list=-angles, flevel=40, plsc=pxscale, rad_dists=[40], theta=160, n_branches = 1)
            fake_comp_1 = vip.metrics.cube_inject_companions(science_target[1], psf_template=psfn, angle_list=-angles, flevel= 1000, plsc=pxscale, rad_dists=[40], theta=160, n_branches = 1)
            print("fake companion 0 shape = ", fake_comp_0.shape)
            fake_comp = np.zeros((wl_ref, nb_fr_ref, w, h))
            fake_comp[0] = fake_comp_0
            fake_comp[1] = fake_comp_1
            path_fake_comp = "./K_kilp_ADI_RDI/fake_planet/fake_comp02.fits"

            hdu = fits.PrimaryHDU(fake_comp)
            hdu.writeto(path_fake_comp) 
        
        elif obj == "DISK":
            dstar = 35 # distance to the star in pc, the bigger the disk if more small and more close to star
            nx = 256 # number of pixels of your image in X
            ny = 256 # number of pixels of your image in Y
            itilt = 65 # inclination of your disk in degreess (0 means pole-on -> can see the full plate, 90 means edge on -> only see a line)
            pa = -50 # position angle of the disk in degrees (0 means north, 90 means east)
            a = 40 # semimajoraxis of the disk in au / semimajor axis in arcsec is 80 au/80px = 1 arcsec
            fake_disk1 = vip.metrics.scattered_light_disk.ScatteredLightDisk(\
                        nx=nx,ny=ny,distance=dstar,\
                        itilt=itilt,omega=0,pxInArcsec=pxscale,pa=pa,\
                        density_dico={'name':'2PowerLaws','ain':4,'aout':-4,\
                        'a':a,'e':0.0,'ksi0':1.,'gamma':2.,'beta':1.},\
                        spf_dico={'name':'HG', 'g':0., 'polar':False})
            fake_disk1_map = fake_disk1.compute_scattered_light()
            fake_disk1_map = fake_disk1_map/np.max(fake_disk1_map)
            #ds9 = vip.Ds9Window()
            #ds9.display(fake_disk1_map)
            
            # add fake disk to science target
            scaling_factor = 0.1
            cube_fakeddisk = vip.metrics.cube_inject_fakedisk(fake_disk1_map*scaling_factor ,angle_list=angles,psf=psfn)
            
            # only want center
            start = int(w*(1-scale)/2)
            end = int(start+w*scale)
            science_target[0,:,start:end,start:end] = science_target[0,:,start:end,start:end] + cube_fakeddisk
            cube_fakeddisk = vip.metrics.cube_inject_fakedisk(fake_disk1_map,angle_list=angles,psf=psfn)
            science_target[1,:,start:end,start:end] = science_target[1,:,start:end,start:end] + cube_fakeddisk*100
            path_fake_disk = "./K_kilp_ADI_RDI/fake_planet/"+str(sys.argv[6])
            hdu = fits.PrimaryHDU(science_target)
            hdu.writeto(path_fake_disk)
        else:
            print("No such object inject option")
        
        # display
        #ds9 = vip.Ds9Window()
        #ds9.display(fake_comp[0])
    elif opt == "SCAL":
        # SCAL : analysis scalings effects 
        SCAL(sys.argv, scale)        
    elif opt == "SAM":
        # SAM : spat-annular-mean
        SAM(sys.argv, scale)

    elif opt == "WDH":
        # WDH : wind driven halo
        WDH(sys.argv, scale)

    else:
        print("No such option")

    start_and_end_program(False)
