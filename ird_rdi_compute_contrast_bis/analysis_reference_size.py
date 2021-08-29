import os
import sys
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import vip_hci as vip
import matplotlib.pyplot as plt

from astropy.io import fits
from hciplot import plot_frames, plot_cubes
from astropy.visualization import simple_norm
from photutils.aperture import CircularAperture, aperture_photometry, CircularAnnulus

if __name__ == "__main__":
    print("###### Start to process the data ######")

    nb_data = 6
    l = 6 
    
    """
    data = np.zeros((l, nb_data))
    data[0] = [-0.4467667827461027, 0.9607431218323801, 0.9806888776653295, 0.8646857262683288, 1.4807527653781036, 1.396990275918117] 
    data[1] = [-0.12083997242618855, 1.2641610342167502, 1.2909017499831377, 1.3641149092974028, 1.7843590149776698, 1.662082524385601]
    data[2] = [0.06299466463906618, 2.7373192340302266, 3.143930556532704, 1.9764236680866907, 2.5901748610169433, 3.020383968615745] 
    data[3] = [0.07779613812228453, 2.3112495874449777, 2.504749816729505, 1.7396240608514233, 2.1425034369777376, 2.5199478783062106] 
    data[4] = [0.04984205524889294, 2.9945048310623967, 2.0390510218074196, 2.2394076587614764, 1.6010538369834595, 2.339927022675774]  
    data[5] = [0.16263245464874143, 1.1058966346215848, 1.3772360373204204, 1.7716665992629665, 1.5996924622354836, 1.295037762380921]
    
    ##################
    # plot te result #
    ##################
    
    data_total = pd.DataFrame(data[:,:], columns=["ncorr_100_ncomp_020","ncorr_100_ncomp_040","ncorr_100_ncomp_060","ncorr_100_ncomp_080","ncorr_100_ncomp_100","ncorr_100_ncomp_150"])
    data_total.index = [1283, 6418, 12836, 25673, 64184, 128368]

    print("######### Contrast of companion #######")
    print(data_total)
    #data_total.to_csv("Flux_of_companion.csv")
    #data_total.plot(kind='line', style='--o', title='comparation')
    sns.set(font_scale = 1.5)
    sns.relplot(kind='line',data=data_total)
    plt.title("The signal-to-noise ratio changes with the size of the reference library", fontsize = 20)
    plt.xlabel("The size of reference library", fontsize = "18")
    plt.ylabel("SNR", fontsize = "18")
    #plt.ylim(0,70)
    plt.show()
    """
    data = np.zeros((10, 6))

    # n_corr = 150, ncomp = 20 ~ 200 
    data[0] = [-0.535421623464843, 1.0285072472819161, 0.677029314032496, 1.59978302904129, 1.271588180984381, 2.039458753720442]
    data[1] = [-0.0699839315500407, 1.3521001459165631, 1.1432234599967244, 1.1031776087944587, 1.769349512927069, 1.8300804814198024]
    data[2] = [0.34708658536072445, 2.9100781559433497, 2.839673621373633, 1.7941460419813853, 2.4338979767106377, 2.3860087669230707]
    data[3] = [0.2520678793764617, 3.0677524058520307, 2.469703794917576, 1.8723662065370084, 1.7100766341923936, 2.4975813583949074]
    data[4] = [0.38992840371252946, 2.5892901903036867, 1.9807435612593298, 2.356980862343945, 2.16980513550114, 2.061867331654142]
    data[5] = [0.24561374987490805, 2.147571782896199, 1.8364902397975602, 1.987715042698089, 2.0576123454925574, 1.3675960340719493]
    data[6] = [0.2526509850605604, 1.6153786022897665, 1.7948068458070001, 1.717457959681953, 1.6479688365084528, 1.3740090180511417]
    data[7] = [0.2982495888472051, 1.1295259552077672, 0.8802170380564819, 1.2810243723771757, 2.0179711021782145, 1.3457906933592347]
    data[8] = [0.33063057259761003, 0.9071548985346716, 1.0850725143661555, 2.003405061237857, 2.0838154087855356, 1.5052935938220664]
    data[9] = [0.34877596146858114, 1.4433775131092932, 1.3251632219761749, 1.7596195105283645, 2.148522572704197, 2.2492166195324965]


    ##################
    # plot te result #
    ##################
    
    data_total = pd.DataFrame(data.T, columns=["ncomp_020","ncomp_040","ncomp_060","ncomp_080","ncomp_100","ncomp_120", "ncomp_140","ncomp_160","ncomp_180","ncomp_200"])
    data_total.index = [1283, 6418, 12836, 25673, 64184, 128368]

    print("######### Contrast of companion #######")
    print(data_total)
    #data_total.to_csv("Flux_of_companion.csv")
    #data_total.plot(kind='line', style='--o', title='comparation')
    sns.set(font_scale = 1.5)
    sns.relplot(kind='line',data=data_total)
    plt.title("The signal-to-noise ratio changes with the size of the reference library, science target Beta Pictoris, ncorr=150", fontsize = 20)
    plt.xlabel("The size of reference library", fontsize = "18")
    plt.ylabel("SNR", fontsize = "18")
    #plt.ylim(0,70)
    plt.show()
    
    print("###### End of program ######")
