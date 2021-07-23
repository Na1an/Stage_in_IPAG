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
* 

### 3. Launch

--------

It is easy to run the program. We have ```cADI```, ```PCA``` and ```RDI``` three modes for now.

* mode **cADI**

  ```
  python3 RDI.py cADI ../SPHERE_DC_DATA/HD\ 156384Cf_DB_H23_2017-06-27_ird_convert_recenter_dc5_PUBLIC_208368 0.5
  ```

  * **arg1** : algo mode. 
  * **arg2** : path of target object repository
  * **arg3**: scale, default = 0.125

* mode **PCA** 

  ```
  python3 RDI.py PCA ../SPHERE_DC_DATA/HD\ 156384Cf_DB_H23_2017-06-27_ird_convert_recenter_dc5_PUBLIC_208368 ../SPHERE_DC_DATA 0.25
  ```

  * **arg1** : algo mode. 
  * **arg2** : path of target object repository
  * **arg3** : reference object repository
  * **arg4**: scale, default = 0.25

* Exemple of execution

  ```
  python main.py Algo_RDI ../SPHERE_DC_DATA/HD\ 156384Cf_DB_H23_2017-06-27_ird_convert_recenter_dc5_PUBLIC_208368 ../SPHERE_DC_DATA 0.25 ./tmp                                                                                                 
  ```



### 4. Sujet

--------



### 5. RDI

-------



### 6. 

-------



### 7. Contack

--------

*Yuchen BAI : yuchen.bai@univ-grenoble-alpes.fr* 