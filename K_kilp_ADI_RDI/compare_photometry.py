import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from astropy.io import fits
from astropy.visualization import simple_norm
from photutils.aperture import CircularAperture, aperture_photometry, CircularAnnulus

SHOW_POSITION = False

positions = [(126.05284, 249.11)]
aperture = CircularAperture(positions, r=2)
annulus = CircularAnnulus(positions, r_in=4, r_out=6)

def get_stdev(data):
    radius = int((128-125.99284)**2+(128-248.92338)**2)
    pixels = []
    for i in range(256):
        for j in range(256):
            if int(((125.99284-i)**2+(248.92338-j)**2)**0.5) < 2:
                continue
            if ((128-i)**2+(128-j)**2) < (radius+1) :
                pixels.append(data[i][j])
    return np.std(pixels)

def get_photometry(path):
    files = os.listdir(path)
    files.sort()
    res = np.zeros((len(files)))
    SN = np.zeros(len(files))
    for i in range(len(res)):
        file = path+'/'+files[i]
        print("file =",file)
        data = fits.getdata(file)
        flux_companion = aperture_photometry(data, [aperture, annulus])
        flux_companion['aperture_sum_0','aperture_sum_1'].info.format = '%.8g'
        bkg_mean = flux_companion['aperture_sum_1']/annulus.area
        bkg_sum_in_companion = bkg_mean * aperture.area 
        annulus_stdev = get_stdev(data)
        res[i] = flux_companion['aperture_sum_0'] - bkg_sum_in_companion
        SN[i] = res[i] / annulus_stdev 
        if i==1 and SHOW_POSITION:
            norm = simple_norm(data, 'sqrt', percent=99)
            plt.imshow(data, norm=norm, interpolation='nearest')
            ap_patches = aperture.plot(color='white', lw=2, label='Photometry aperture '+str(res[i]))
            ann_patches = annulus.plot(color='red', lw=2, label='Background annulus bkg_sum_in_companion '+ str(bkg_sum_in_companion))
            handles = (ap_patches[0],ann_patches[0])
            plt.legend(loc=(0.17, 0.05), facecolor='#458989', labelcolor='white', handles=handles, prop={'weight':'bold', 'size':11})
            plt.xlim(100,170)
            plt.ylim(200,256)
            #plt.savefig('./circle_ADI/ADI_32px_'+str(i))
            plt.show()

    return res, SN 

# ADI data
ADI_res, ADI_SN = get_photometry("./ADI_WITH_MASK")
ADI_res_32, ADI_SN_32 = get_photometry("./ADI_WITH_MASK_32")
#print(ADI_res_32)

# RDI data 1 target 2 ref stars
#RDI_res_2_ref, RDI_2_SN = get_photometry("./RDI_ref_2_star")
#print(RDI_res_2_ref)

# RDI data 1 target 4 ref stars
#RDI_res_4_ref, RDI_4_SN = get_photometry("./RDI_ref_4_star")
#print(RDI_res_4_ref)

#
RDI_flux_3_best, RDI_SN_3_best = get_photometry("./RDI_WITH_MASK_3_best")
RDI_flux_5_best, RDI_SN_5_best = get_photometry("./RDI_WITH_MASK_5_best")

RDI_flux_3_best_32, RDI_SN_3_best_32 = get_photometry("./RDI_WITH_MASK_3_best_32")
RDI_flux_5_best_32, RDI_SN_5_best_32 = get_photometry("./RDI_WITH_MASK_5_best_32")


sns.set(style="darkgrid")

# nb of data
nb_data = 6
data = np.zeros((len(ADI_res), nb_data))
for i in range(len(ADI_res)):
    data[i][0] = ADI_res[i]
    data[i][1] = RDI_flux_3_best[i]
    data[i][2] = RDI_flux_5_best[i]
    data[i][3] = ADI_res_32[i]
    data[i][4] = RDI_flux_3_best_32[i]
    data[i][5] = RDI_flux_5_best_32[i]

data_SN = np.zeros((len(ADI_res), nb_data))
for i in range(len(ADI_res)):
    data_SN[i][0] = ADI_SN[i]
    data_SN[i][1] = RDI_SN_3_best[i]
    data_SN[i][2] = RDI_SN_5_best[i]
    data_SN[i][3] = ADI_SN_32[i]
    data_SN[i][4] = RDI_SN_3_best_32[i]
    data_SN[i][5] = RDI_SN_5_best_32[i]

data_total = pd.DataFrame(data[:,:], columns=['ADI_16', 'RDI_3_best_with_mask','RDI_5_best_with_mask','ADI_32pxs','RDI_3_best_with_mask_32pxs','RDI_5_best_with_mask_32pxs'])
data_total.index = data_total.index + 1
print("######### Flux of companion #######")
print(data_total)
#data_total.plot(kind='line', style='--o', title='comparation')
sns.relplot(kind='line',data=data_total)
plt.title("Target:GJ667c Ref: CJ3998/CJ442/61Vir/CJ674/GJ682")
plt.xlabel("K_kilp")
plt.ylabel("Flux of the companion absolute - diameter 4 px")
plt.show()

data_total_SN = pd.DataFrame(data_SN[:,:], columns=['ADI_16_SN','RDI_SN_3_best_with_mask','RDI_SN_5_best_with_mask','ADI_32_SN','RDI_SN_3_best_with_mask_32pxs','RDI_SN_5_best_with_mask_32pxs'])
data_total_SN.index = data_total_SN.index + 1
print("######### S/N ########")
print(data_total_SN)
#data_total.plot(kind='line', style='--o', title='comparation')
sns.relplot(kind='line',data=data_total_SN)
plt.title("Target:GJ667c Ref: CJ3998/CJ442/61Vir/CJ674/GJ682")
plt.xlabel("K_kilp")
plt.ylabel("S/N - diameter 4 px")
plt.show()
