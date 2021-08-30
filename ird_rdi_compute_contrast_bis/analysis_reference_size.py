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
    data = np.zeros((9, 11))

    # n_corr = 150, ncomp = 20 ~ 200 
    data[0] = [-0.3686951617658657, -0.2285493846142516, 0.16316690236372547, 0.46338230756102516, 0.48082404047, 0.47004725770927674, 0.4539387780490161, 0.4909024201700399, 0.6023755994778509, 0.9783448852598373, 0.9991456500294687]
    data[1] =[-0.26692705499964886, -0.3329925834144311, 1.0387556351277114, 1.2162714378408253, 1.3157912417725988, 1.2794584377624187, 1.1327349116528083, 1.2807080653559426, 0.8491841385419369, 1.4446462693969269, 1.7076532798200619] 
    data[2] =[-0.3978950504717747, 0.6435375653866412, 1.6223176631952605, 2.504877569939859, 2.263076004455707, 2.4901531283098155, 2.5034158807431197, 1.2602378886957148, 1.330364218156237, 1.8162084418791267, 1.4188107059861164]
    data[3] =[-0.44495112290456396, 0.2062767427042406, 2.2094044157298276, 3.122324455378017, 2.737000300704315, 2.517196819294629, 2.751799425777127, 1.448845209500694, 1.9110763683358036, 1.2362399757071632, 0.9269745089340595]
    data[4] =[-0.23685229363457025, 0.5669134678594006, 2.136972149338596, 1.4739768739339483, 1.9403128438338564, 1.396943117657149, 1.3234373682563463, 1.9297848509640183, 2.0350419940152613, 1.019994283419066, 1.9829945823102122]
    data[5] =[-0.19684646038328682, 0.3545933133683571, 1.6065029322914506, 1.3734638121563112, 1.44004334187974, 0.8496656718665944, 0.6269381062271261, 1.174728504228164, 1.2160454715021247, 0.9972761339826787, 1.5792747061076942]
    data[6] =[0.2293976386471256, 0.216249340859169, 1.8517636015924215, 1.4781340413184891, 1.7836410762646864, 2.352382670315469, 1.4552797463009097, 1.808395490831543, 2.1663645522862596, 1.2512278189646824, 1.0670525200571777]
    data[7] =[-0.1751809311303769, 0.216249340859169, 1.479515473304608, 1.30936706083232, 1.2255293329750794, 0.9876530582913616, 0.9509867936114238, 2.6703937406444527, 1.757947125570223, 1.0375981437047328, 1.07087232582596]
    data[8] =[-0.10452655433143608, 0.216249340859169, 1.8843239294178076, 0.9817520026101113, 1.1518108335977737, 0.7551701594824215, 1.0844704792059072, 2.5353104032056315, 1.291851075601125, 1.0471737507096222, 1.545549028482012]


    ##################
    # plot te result #
    ##################
    
    data_total = pd.DataFrame(data.T, columns=["ncomp_020","ncomp_050","ncomp_100","ncomp_150","ncomp_200","ncomp_250", "ncomp_300","ncomp_350","ncomp_400"])
    data_total.index = [641,1283,2567,5134,7702,10269,12836,25673,38510,64184,128368]

    print("######### Contrast of companion #######")
    print(data_total)
    #data_total.to_csv("Flux_of_companion.csv")
    #data_total.plot(kind='line', style='--o', title='comparation')
    sns.set(font_scale = 1.5)
    sns.relplot(kind='line',data=data_total)
    plt.title("The signal-to-noise ratio changes with the size of the reference library, science target Beta Pictoris, ncorr=200", fontsize = 20)
    plt.xlabel("The size of reference library", fontsize = "18")
    plt.ylabel("SNR", fontsize = "18")
    #plt.ylim(0,70)
    plt.show()
    
    print("###### End of program ######")
