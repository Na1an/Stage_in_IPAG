import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from astropy.io import fits
from photutils.aperture import CircularAperture, aperture_photometry

positions = [(127. ,250.)]
aperture = CircularAperture(positions, r=2.)

def get_photometry(path):
    files = os.listdir(path)
    res = np.zeros((len(files)))
    for i in range(len(res)):
        flux_companion = aperture_photometry(fits.getdata(path+'/'+files[i]), aperture)
        flux_companion['aperture_sum'].info.format = '%.8g'
        res[i] = flux_companion['aperture_sum']
    return res 

# ADI data
ADI_res = get_photometry("./ADI")
print(ADI_res)

# RDI data 1 target 2 ref stars
RDI_res_2_ref = get_photometry("./RDI_ref_4_star_corre")
print(RDI_res_2_ref)

# RDI data 1 target 4 ref stars
RDI_res_4_ref = get_photometry("./RDI_ref_4_star")
print(RDI_res_4_ref)

sns.set(style="darkgrid")

data = np.zeros((len(ADI_res),3))
for i in range(len(ADI_res)):
    data[i][0] = ADI_res[i]
    data[i][1] = RDI_res_2_ref[i]
    data[i][2] = RDI_res_4_ref[i]

data_total = pd.DataFrame(data, columns=['ADI','RDI_5_ref_corr','RDI_5_ref'])
data_total.index = data_total.index + 1
print(data_total)
#data_total.plot(kind='line', style='--o', title='comparation')
sns.relplot(kind='line',data=data_total)
plt.title("Target:GJ667c Ref: CJ3998/CJ442/61Vir/CJ674/GJ682")
plt.xlabel("K_kilp")
plt.ylabel("Flux of the companion - diameter 4 px")
plt.show()
