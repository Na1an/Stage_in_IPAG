# Stage_in_IPAG

*Me : Yuchen BAI*

*Supervisor : Julien MILLI*

### 1. Installation

---------

The project will continue to be maintained on github.

* ``` git clone https://github.com/Na1an/Stage_in_IPAG.git```

### 2. Environment

----------

Nothing special, the project should work in all Linux OS. Python3 is needed.

* ```uname -r``` : 4.15.0-29deepin-generic
* ```cat /etc/debian_version``` : 9.0
* ```python3 --version``` : Python 3.6.5
  * **numpy**
  * **matplotmib**
  * **astropy**
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
  python3 RDI.py PCA ../SPHERE_DC_DATA/HD\ 156384Cf_DB_H23_2017-06-27_ird_convert_recenter_dc5_PUBLIC_208368 ../SPHERE_DC_DATA/HD\ 156384Cf_DB_H23_2017-06-27_ird_convert_recenter_dc5_PUBLIC_208368 0.25
  ```

  * **arg1** : algo mode. 
  * **arg2** : path of target object repository
  * **arg3** : reference object repository
  * **arg4**: scale, default = 0.125

  


### 4. Sujet

--------
Description of the topic
I solved the conflict


### 5. RDI

-------



### 6. 

-------



### 7. Contack

--------

*Yuchen BAI : yuchen.bai@univ-grenoble-alpes.fr* 
