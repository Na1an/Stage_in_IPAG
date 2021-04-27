import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from astropy.io import fits
from photutils.aperture import CircularAperture, aperture_photometry, CircularAnnulus

positions = [(125.99284, 248.92338)]
aperture = CircularAperture(positions, r=2.)
annulus = CircularAnnulus(positions, r_in=4., r_out=6.)

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
    res = np.zeros((len(files)))
    SN = np.zeros(len(files))
    for i in range(len(res)):
        data = fits.getdata(path+'/'+files[i])
        flux_companion = aperture_photometry(data, [aperture, annulus])
        flux_companion['aperture_sum_0','aperture_sum_1'].info.format = '%.8g'
        bkg_mean = flux_companion['aperture_sum_1']/annulus.area
        bkg_sum_in_companion = bkg_mean * aperture.area 
        annulus_stdev = get_stdev(data)
        res[i] = flux_companion['aperture_sum_0'] - bkg_sum_in_companion
        SN[i] = res[i] / annulus_stdev 
    return res, SN 

# ADI data
ADI_res, ADI_SN = get_photometry("./ADI")
print(ADI_res)

# RDI data 1 target 2 ref stars
RDI_res_2_ref, RDI_2_SN = get_photometry("./RDI_ref_2_star")
print(RDI_res_2_ref)

# RDI data 1 target 4 ref stars
RDI_res_4_ref, RDI_4_SN = get_photometry("./RDI_ref_4_star")
print(RDI_res_4_ref)

sns.set(style="darkgrid")

data = np.zeros((len(ADI_res),3))
for i in range(len(ADI_res)):
    data[i][0] = ADI_res[i]
    data[i][1] = RDI_res_2_ref[i]
    data[i][2] = RDI_res_4_ref[i]
    
data_SN = np.zeros((len(ADI_res),3))
for i in range(len(ADI_res)):
    data_SN[i][0] = ADI_SN[i]
    data_SN[i][1] = RDI_2_SN[i]
    data_SN[i][2] = RDI_4_SN[i]


data_total = pd.DataFrame(data, columns=['ADI','RDI_2_ref','RDI_5_ref'])
data_total.index = data_total.index + 1
print("######### Flux of companion #######")
print(data_total)
#data_total.plot(kind='line', style='--o', title='comparation')
sns.relplot(kind='line',data=data_total)
plt.title("Target:GJ667c Ref: CJ3998/CJ442/61Vir/CJ674/GJ682")
plt.xlabel("K_kilp")
plt.ylabel("Flux of the companion - diameter 4 px")
plt.show()

data_total_SN = pd.DataFrame(data_SN[:,1:], columns=['RDI_2_ref_S/N','RDI_5_ref_S/N'])
data_total_SN.index = data_total_SN.index + 1
print("######### S/N ########")
print(data_total_SN)
#data_total.plot(kind='line', style='--o', title='comparation')
sns.relplot(kind='line',data=data_total_SN)
plt.title("Target:GJ667c Ref: CJ3998/CJ442/61Vir/CJ674/GJ682")
plt.xlabel("K_kilp")
plt.ylabel("S/N - diameter 4 px")
plt.show()
