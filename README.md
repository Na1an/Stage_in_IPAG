# Stage_in_IPAG

*Me : Yuchen BAI*

*Supervisor : Julien MILLI*

### 1. Installation

---------

The project will continue to be maintained on github.

* ``` git clone https://github.com/Na1an/Stage_in_IPAG.git```



### 2. Environment

----------

Nothing special, the project should work in all Linux OS. ```Python3``` is needed.

* ```uname -r``` : 5.12.3-arch1-1
* ```python3 --version``` : Python 3.7.9
  * ```numpy```
  * ```matplotmib```
  * ```astropy```
  * ```vip_hci``` : **Serious mistake**. 
    * Since version 0.16 of [scikit-image](https://scikit-image.org/). There is a function was renamed from `skimage.measure.compare_ssim` to `skimage.metrics.structural_similarity`. So you should change the function name in the related file. 
    
    * Should run ```pip install pandas --upgrade``` to upgrade pandas, because the requirement of pandas is out of date. We may meet the problem of **TypeError: Cannot interpret '<attribute 'dtype' of 'numpy.generic' objects>**. 
    
      ```
      ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
      vip-hci 0.9.11 requires pandas~=0.18, but you have pandas 1.2.4 which is incompatible.
      ```
  
  

### 3. Routine on Cobrex-dc

--------------------

On the branch '**test**' of cobrex-dc, 4 routines have been implemented and tested successfully, they are:

* ***ird_rdi_fake_injection_bis*** : Injecte a fake companion into a IRD_SCIENCE_REDUCED_MASTER_CUBE

  * ```contrast``` : the contrast we want for our fake companion, the unit is factor of 6, type=float, default=0.00001.
  * ```rad_dist``` : the distance/radius from the fake companion to star, type=int, default=25.
  * ```theta``` : the azimuth angle of fake injection, type=int, default=60.
  * ```n_branches``` : how many brances we want, type=int, default=1.
  * ``wl_channels`` : spectral channel to use (to choose between 0 for channel 0, 1 for channel 1, 2 for both channels), type=int, default=2.
  * ```diameter``` : diameter of the fake companion", type=float, default=4.0.

  The source code is in this repository, so running the code locally is possible. But you should make sure that the data is on your laptop, an example is below :

  * ```
    /home/yuchen/Documents/IPAG/SPHERE_DC_DATA/HD_156384Cf_DB_H23_2017-06-27_ird_convert_recenter_dc5_PUBLIC_208368/ird_convert_recenter_dc5-IRD_SCIENCE_REDUCED_MASTER_CUBE-center_im.fits IRD_SCIENCE_REDUCED_MASTER_CUBE
    /home/yuchen/Documents/IPAG/SPHERE_DC_DATA/HD_156384Cf_DB_H23_2017-06-27_ird_convert_recenter_dc5_PUBLIC_208368/ird_convert_recenter_dc5-IRD_SCIENCE_PARA_ROTATION_CUBE-rotnth.fits IRD_SCIENCE_PARA_ROTATION_CUBE
    /home/yuchen/Documents/IPAG/SPHERE_DC_DATA/HD_156384Cf_DB_H23_2017-06-27_ird_convert_recenter_dc5_PUBLIC_208368/ird_convert_recenter_dc5-IRD_SCIENCE_PSF_MASTER_CUBE-median_unsat.fits IRD_SCIENCE_PSF_MASTER_CUBE
    /home/yuchen/Documents/IPAG/SPHERE_DC_DATA/BD-12_4523b_DB_H23_2017-07-28_ird_convert_recenter_dc5_PUBLIC_209689/ird_convert_recenter_dc5-IRD_SCIENCE_REDUCED_MASTER_CUBE-center_im.fits IRD_SCIENCE_REDUCED_MASTER_CUBE
    ```

  * ``` python ird_rdi_fake_injection_bis.py process_282814.sof ```

    

* ***ird_rdi_build_corr_matrix_bis*** : Reduce a IRD_SCIENCE_REDUCED_MASTER_CUBE, build a Pearson Correlation Coefficient matrice.

  * ```inner_radius``` : inner radius of the annulus/interest zone, type=int, default=10.

  * ```outer_radius``` : outer radius of the annulus/interest zone, type=int, default=100.

  * ```science_object``` : the OBJECT keyword of the science target, type=str, default='unspecified'.

  * ```wl_channels``` : spectral channel to use (to choose between 0 for channel 0, 1 for channel 1, 2 for both channels)", type=int, choices=[0,1,2], default=0.

  * ```python ird_rdi_build_corr_matrix_bis.py process_282814.sof --inner_radius=10 --outer_radius=100 --science_object="GJ 667C" --wl_channels=2```

    

* ***ird_rdi_reduce*** : Reduce a IRD_SCIENCE_REDUCED_MASTER_CUBE

  * ```score``` : which decide how we choose the reference frame (>=1), type=int, default=1.

  * ```n_corr``` : the number of best correlated frames for each frame of science target, a list of integer, type=str, default="empty".

  * ```ncomp``` : number of principal components to remove, a list of integer, type=str, default="empty".

  * ```pct``` : the percentage we want to use the reference library, a list of float, type=str, default="empty".

  * ```scaling``` : scaling for the PCA (to choose between 0 for spat-mean, 1 for spat-standard, 2 for temp-mean, 3 for temp-standard or 4 for None)", type=int, choices=[0,1,2,3,4], default=0.

  * ```python ird_rdi_reduce.py process_282814.sof --n_corr="1 2 3" --ncomp="1 2 3" --pct="0.1 0.2 0.5 0.92"```

    

* ***ird_rdi_compute_contrast_bis*** : 

  * ```coordinates``` : positions of fake companion, a string, type=str, default="empty".
  * ```diam``` : the diameter for calculating snr", type=float, default=4.0.
  * ```r_aperture``` : radius to compute the flux/contrast", type=float, default=2.0.
  * ```r_in_annulus``` : ~~inner radius of annulus around the fake companion", type=float, default=4.0.~~ Don't care this option for instant.
  * ```r_out_annulus``` : ~~outer radius of annulus around the fake companion", type=float, default=6.0.~~ Don't care this option for instant.
  * ```python ird_rdi_compute_contrast_bis.py --coordinates="(25.03, 9.93);(123, 123)" --diam=4 --r_aperture=2 --r_in_annulus=4 --r_out_annulus=6 process_282814.sof```



### 4. Local Launch

--------

During the stage of research, many programs have been created, these programs are listed below. By executing ```main.py```, you can use these options :

* mode **cADI** : execute the algo Angular Differential Imaging (ADI) 

  ```
  python3 RDI.py cADI ../SPHERE_DC_DATA/HD\ 156384Cf_DB_H23_2017-06-27_ird_convert_recenter_dc5_PUBLIC_208368 0.5
  ```

  * **arg1** : algo mode. 
  * **arg2** : path of target object repository
  * **arg3**: scale, default = 0.125

* mode **PCA** : execute the algo Reference Differential Imaging (RDI)

  ```
  python3 RDI.py PCA ../SPHERE_DC_DATA/HD\ 156384Cf_DB_H23_2017-06-27_ird_convert_recenter_dc5_PUBLIC_208368 ../SPHERE_DC_DATA 0.25
  ```

  * **arg1** : algo mode. 
  * **arg2** : path of target object repository
  * **arg3** : reference object repository
  * **arg4**: scale, default = 0.25

* mode **SCAL** : execute RDI with 5 different scaling - the package **vip_hci** provide 5 different scalings : [spat-mean, temp-mean, spat-standard, temp-standard, None]

  ```python
  python main.py SCAL ../SPHERE_DC_DATA/HD\ 156384Cf_DB_H23_2017-06-27_ird_convert_recenter_dc5_PUBLIC_208368  ../SPHERE_DC_DATA 0.25                                                                    
  ```
  
  * **arg1** : algo mode. 
  * **arg2** : path of target object repository
  * **arg3** : reference object repository
  * **arg4** : scale, default = 0.25

* mode **INJECTION** : inject a planet or a disk, **arg5** provide 2 modes

  ```python
  python main.py INJECTION ../SPHERE_DC_DATA/HIP\ 55042_DB_H23_2018-01-28_ird_convert_recenter_dc5_PUBLIC_209733 ../SPHERE_DC_DATA 0.25 PLANETE                                                                            
  ```

  * **arg1** : algo mode. 
  * **arg2** : path of target object repository
  * **arg3** : reference object repository
  * **arg4**: scale, default = 0.25
  * **arg5**: injection mode

* mode **RDI_scores** : investigate the performance of RDI in different scores

  ```python
  python main.py RDI_scores ../SPHERE_DC_DATA/HD_156384Cf_DB_H23_2017-06-27_ird_convert_recenter_dc5_PUBLIC_208368 ../SPHERE_DC_DATA 0.25 200 MASTER_CUBE-center                                                                      
  ```

  * **arg1** : algo mode. 
  * **arg2** : path of target object repository
  * **arg3** : reference object repository
  * **arg4**: scale, default = 0.25
  * **arg5**: nb_best_frames, ncorr
  * **arg6**: keyword of target object

* mode **SAM** : this algo Spat Annular Mean, didn't give us a better result, so failed

  ```python
  python main.py SAM ../SPHERE_DC_DATA/HD\ 156384Cf_DB_H23_2017-06-27_ird_convert_recenter_dc5_PUBLIC_208368  ../SPHERE_DC_DATA 0.25 3
  ```

  * **arg1** : algo mode. 
  * **arg2** : path of target object repository
  * **arg3** : reference object repository
  * **arg4**: scale, default = 0.25
  * **arg5**: count, the number of we want to chose for the selection

* mode **RDI_frame** : investigate the performance of RDI in different best frame numbers

  ```python
  python main.py RDI_frame ../SPHERE_DC_DATA/HD\ 156384Cf_DB_H23_2017-06-27_ird_convert_recenter_dc5_PUBLIC_208368 /run/media/yuchen/YUCHEN_DISK/super-earths_project/SPHERE_DC_DATA 0.25 200 fake_disk_far ./K_kilp_ADI_RDI/res_0907_presentation/no_scale/far_disk_100pxs/pos1
  ```

  * **arg1** : algo mode. 
  * **arg2** : path of target object repository
  * **arg3** : reference object repository
  * **arg4**: scale, default = 0.25
  * **arg5**: nb_best_frames, ncorr
  * **arg6**: keyword of target object

* mode **RDI_frame_bis** : investigate the performance of RDI in different best frame numbers

  ```python
  python main.py RDI_frame_bis ../SPHERE_DC_DATA/BD-12\ 4523b_DB_H23_2017-07-28_ird_convert_recenter_dc5_PUBLIC_209689 /run/media/yuchen/YUCHEN_DISK/super-earths_project/SPHERE_DC_DATA 0.25 200 Wolf_fake_comp_27px_bis.fits                           
  ```

  * **arg1** : algo mode. 
  * **arg2** : path of target object repository
  * **arg3** : reference object repository
  * **arg4**: scale, default = 0.25
  * **arg5**: nb_best_frames, ncorr
  * **arg6**: keyword of target object
  
  

### 5. About *.xml files

---------

For deploying a **new recipe** on the **Cobrex-DC**, there is two *.xml files to care about. **ExecutableRecipe.xml** and **AssociationRule.xml**.

* **ExecutableRecipe.xml**

  * If you check the recipes running on the cobrex server, you will definitely find that there is a file with the same name in each recipe folder: ExecutableRecipe.xml. You can think of it as a necessary configuration file for each recipe. In this file, we need to define **the recipe name, all data types used by the recipe, parameters, parameter data types, output** and **output data types**. 

* **AssociationRule.xml**

  * This file defines some rules to help us extract data from the database. Imagine when a recipe needs to take the data of tens of thousands of stars as input, it would be crazy to select one by one. So at this time, if there is a rule saying that when the optimize button is pressed, the server will automatically find and take all data of type xx as input. Let us cheer the greatness of computers!

  * Note that when you define a new **association rule** for the recipe, you must pay attention to the grammatical format, even if  **<font color='red'>an extra space</font>** is added, the optimize button will not take effect.

  * The Association rules of RDI are all in **RDI.xml**, this file provides an example for the future implementation :

    ```xml
    <recipe id="ird_rdi_build_corr_matrix.*">
        <frame_type include="IRD_SCIENCE_REDUCED_MASTER_CUBE">
            <attribute id="inst_filter" restriction="equal"/>
            <attribute id="coro" restriction="equal"/>
        </frame_type>
    </recipe>
    <recipe id="ird_rdi_reduce.*">
        <frame_type include="IRD_CORR_MATRIX,IRD_SCIENCE_PARA_ROTATION_CUBE">
            <attribute id="object" restriction="equal"/>
            <attribute id="obs_date" restriction="equal"/>
            <attribute id="exp_time" restriction="equal"/>
            <attribute id="coro" restriction="equal"/>
            <attribute id="inst_filter" restriction="equal"/>
        </frame_type>
    </recipe>
    <recipe id="ird_rdi_fake_injection.*">
        <frame_type include="IRD_SCIENCE_REDUCED_MASTER_CUBE">
            <attribute id="inst_filter" restriction="equal"/>
            <attribute id="coro" restriction="equal"/>
            <attribute id="exp_time" restriction="equal"/>
            <attribute id="neutral_density" restriction="equal" allowNull="true"/>
            <attribute id="parent_process_id" use="equal"/>
        </frame_type>
        <frame_type include="IRD_SCIENCE_PSF_MASTER_CUBE" limit="1">
            <attribute id="parent_process_id" restriction="equal"/>
        </frame_type>
        <frame_type include="IRD_SCIENCE_PARA_ROTATION_CUBE" limit="1">
            <attribute id="exp_time" restriction="equal"/>
            <attribute id="neutral_density" restriction="equal" allowNull="true"/>
            <attribute id="coro" restriction="equal"/>
            <attribute id="inst_filter" restriction="equal"/>
            <attribute id="parent_process_id" restriction="equal"/>
        </frame_type>
    </recipe>
    <recipe id="ird_rdi_compute_contrast.*">
        <frame_type include="IRD_RDI_RES_FAKE_INJECTION">
            <attribute id="obs_night" restriction="equal"/>
            <attribute id="object" restriction="equalOrValue"/>
            <attribute id="inst_filter" restriction="equal"/>
            <attribute id="coro" restriction="equal"/>
        </frame_type>
    </recipe>
    ```
  
  * You may find all the association rules here : **/home/spherdwh/workspace/sphere-server/properties/AssociationRules.xml**
  
    * Many useful examples can be found here, but adding a new association rule requires the help of a cobrex engineer.
  
    

### 6. Contact

--------

*Yuchen BAI : yuchenbai@hotmail.com* 