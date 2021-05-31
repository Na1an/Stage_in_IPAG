# use the methode radial data
# https://github.com/jmilou/image_utilities/blob/master/radial_data.py
import sys
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import norm,t
#sys.path.append('/Users/jmilli/Dropbox/lib_py/image_utilities') # add path to our file
#from image_tools import *

class Radial_data():
    """ Object containing some radial properties of the image
    INPUT:
    ------
    data   - whatever data you are radially averaging.  Data is
        binned into a series of annuli of width 'annulus_width'
        pixels.
    annulus_width - width of each annulus.  Default is 1.
    mask - array of same size as 'data', with zeros at
                  whichever 'data' points you don't want included
                  in the radial data computations.
    x,y - coordinate system in which the data exists (used to set
            the center of the data).  By default, these are set to
            integer meshgrids
    rmax -- maximum radial value over which to compute statistics
    
    
     OUTPUT:
     -------
      r - a data structure containing the following statistics, computed across each annulus:
          .r      - the radial coordinate used (mean radius of the pixels used
                    in the annulus)
          .mean   - mean of the data in the annulus
          .std    - standard deviation of the data in the annulus
          .median - median value in the annulus
          .max    - maximum value in the annulus
          .min    - minimum value in the annulus
          .numel  - number of elements in the annulus
    """ 
    def __init__(self,data,annulus_width=1,mask=None,xvect=None,yvect=None,rmax=None):
        """            
        INPUT:
            ------
            data   - whatever data you are radially averaging.  Data is
                binned into a series of annuli of width 'annulus_width'
                pixels.
            annulus_width - width of each annulus.  Default is 1.
            mask - array of same size as 'data', with zeros at
                          whichever 'data' points you don't want included
                          in the radial data computations.
            x,y - coordinate system in which the data exists (used to set
                    the center of the data).  By default, these are set to
                    integer meshgrids
            rmax -- maximum radial value over which to compute statistics
          """
        data = np.array(data)
        if len(data.shape) != 2 :
            raise ValueError('The input array should be a 2D image')
        if mask is None:
            mask = np.ones(data.shape,bool)        
        self.npix, self.npiy = data.shape
        if xvect is None or yvect is None:
            if np.mod(self.npix,2)==0:
                xvect = np.arange(-self.npix/2.,self.npix/2.)                
            else:
                xvect = np.arange(-self.npix/2.,self.npix/2.)+0.5          
            if np.mod(self.npiy,2)==0:
                yvect = np.arange(-self.npiy/2.,self.npiy/2.)                
            else:
                yvect = np.arange(-self.npiy/2.,self.npiy/2.)+0.5               
#            xvect = np.arange(-self.npix/2.,self.npix/2.)
#            yvect = np.arange(-self.npiy/2.,self.npiy/2.)
        xmap,ymap = np.meshgrid(xvect,yvect)
        self.distmap = np.abs(xmap+1j*ymap)
        if rmax==None:
            rmax = np.max(self.distmap[mask])

        #---------------------
        # Prepare the data container
        #---------------------
        dr = np.abs([xmap[0,0] - xmap[0,1]]) * annulus_width
        radial = np.arange(rmax/dr)*dr + dr/2. # this is changed later (JMi)
        nrad = len(radial)
        self.mean = np.zeros(nrad)
        self.std = np.zeros(nrad)
        self.median = np.zeros(nrad)
        self.numel = np.zeros(nrad)
        self.max = np.zeros(nrad)
        self.min = np.zeros(nrad)
        self.r = radial
        self.noisemap = np.empty(data.shape)
        self.azimuthalmedianmap = np.empty(data.shape)
        self.noisemap.fill(np.nan)
        #---------------------
        # Loop through the bins
        #---------------------
        for irad in range(nrad): #= 1:numel(radial)
            minrad = irad*dr
            maxrad = minrad + dr
            thisindex = (self.distmap>=minrad) * (self.distmap<maxrad) * mask
            if not thisindex.ravel().any():
                self.mean[irad] = np.nan
                self.std[irad]  = np.nan
                self.median[irad] = np.nan
                self.numel[irad] = np.nan
                self.max[irad] = np.nan
                self.min[irad] = np.nan
            else:
                self.r[irad] = self.distmap[thisindex].mean()        
                self.mean[irad] = data[thisindex].mean()
                self.std[irad]  = data[thisindex].std()
                self.median[irad] = np.median(data[thisindex])
                self.numel[irad] = data[thisindex].size
                self.max[irad] = data[thisindex].max()
                self.min[irad] = data[thisindex].min()
                self.noisemap[thisindex] = self.std[irad]
                self.azimuthalmedianmap[thisindex] = self.median[irad]
    
    def get_noise_function(self,fwhm=None,sigma=5.,curve1d=True,verbose=True):
        """
        Returns a function that returns the noise as a function of the separation.
        In case the keyword fwhm is set then the penalty term from the theory of 
        small sample statistics (Mawet et al 2014) is included
        in the noise term.
        """
        if fwhm is not None:
            if verbose:
                print('You have included the small sample correction ! That is great !')
            noise_curve_corrected=sigma*self.std*self.get_penalty(fwhm,sigma,
                            curve1d=curve1d,verbose=verbose)
            id_ok = np.isfinite(noise_curve_corrected) & (noise_curve_corrected>0)
            return interp1d(self.r[id_ok],noise_curve_corrected[id_ok],kind='linear',
                            bounds_error=False,fill_value=np.nan)            
        else:
            if verbose:
                print('You have not included the small sample correction ! Shame ! ')
            return interp1d(self.r,sigma*self.std,kind='cubic',bounds_error=False,fill_value=np.nan)

    def get_noise_map(self,fwhm=None,sigma=5.):
        """
        Returns a 2D noise map corresponding to the 1D profile made 2D.
        In case the keyword fwhm is set then the penalty term from the theory of 
        small sample statistics (Mawet et al 2014) is included
        in the noise map, with a number of dof corresponding to the 2D case.        
        """
        noise_func=self.get_noise_function(fwhm=fwhm,sigma=sigma,curve1d=False)
        noisemap_nan_corrected = noise_func(self.distmap)
#        noisemap_nan_corrected=np.array(self.noisemap)
#        nb_wo_noise_value=0
#        for (i,j), value in np.ndenumerate(self.noisemap):
#            if np.isnan(value):
#                try:
#                    noisemap_nan_corrected[i,j] = noise_func(self.distmap[i,j])
#                except:
#                    noisemap_nan_corrected[i,j] = np.nan
#                    nb_wo_noise_value += 1
#        if nb_wo_noise_value>0:
#            print('Warning: the noise map could not be estimated everywhere, {0:5} pixels have no noise value'.format(nb_wo_noise_value))
        return noisemap_nan_corrected

    def get_penalty(self,fwhm,sigma=5.,curve1d=True,verbose=False):
        """
        Returns an array containing the penalty term to apply to the noise curve
        to account for the small number statistics.
        Input:
            - fwhm: the size of a resolution element in pixel
            -sigma: the confidence level expressed in number of sigma for a gaussian
                    density distribution (by default 5)
            -curve1d: if True, it return the penalty term for a 1D contrast curve,
                      if False it assumes you test each resel independantly and 
                      (for a contrast map for instance) and the penalty term is higher.
        """
        # number of resolution elements at each radius r
        nbResels = np.array(np.round(2*np.pi*self.r/float(fwhm)),dtype=int) 
        # Convidence level corresponding to the given sigma level (gaussian)
        confidenceLevel = norm.cdf(sigma)
        if verbose:
            print('The false alarm probability for {0:f} sigma is {1:6.2e}'.format(sigma,1-confidenceLevel))
            if curve1d:
                print('You chose a 1D contrast curve')
            else:
                print('You chose a 2D contrast map')                
        #ppf is the percent point function (inverse of cdf - percentiles)
        if curve1d:
            return t.ppf(confidenceLevel, nbResels-1)*np.sqrt(1.+1./nbResels)/sigma
        else:
            return t.ppf(confidenceLevel, nbResels-2)*np.sqrt(1.+1./(nbResels-1))/sigma                        
            
if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from image_tools import distance_array

    sigma = 3    
    size=10
#    fake_img = np.random.normal(np.random.randint(-5,5),np.random.rand(),(size,size))
    fake_img = distance_array((size,size))#,centerx=size/2.-0.5,centery=size/2.-0.5)
#    fake_img = distance_array((size,size))


    plt.figure(0)
    plt.imshow(fake_img,origin='lower')
    plt.colorbar()
    
#    rd=Radial_data(fake_img,xvect=np.arange(-size/2,size/2.)+0.5,yvect=np.arange(-size/2,size/2.)+0.5)
    rd=Radial_data(fake_img)

    # example of use 
    plt.figure(1)
    plt.plot(rd.r,rd.mean,'ro',label='Mean')
    plt.plot(rd.r,rd.std,'g:',label='Std')
    plt.plot([0,size/2.*np.sqrt(2)],[0,size/2.*np.sqrt(2)],'b-',label='y=x')
    plt.xlabel('Separaton in px')
    plt.xlabel('Value in ADU')
    plt.grid()
    plt.legend()        
#
    print(rd.r)
    print(rd.mean)
    # example to compute the penalty factor due to small sample statistics
    penalty_factor_1d=rd.get_penalty(1,sigma,verbose=True)
    penalty_factor_2d=rd.get_penalty(1,sigma,verbose=True,curve1d=False)

    # we double check the result here
    sep = rd.r #np.arange(1,11)
    nbResels=np.round(2*np.pi*sep)
    confidenceLevel = norm.cdf(sigma)
    penalty_2d=t.ppf(confidenceLevel, nbResels-2)*np.sqrt(1.+1./(nbResels-1))/sigma
    penalty_1d=t.ppf(confidenceLevel, nbResels-1)*np.sqrt(1.+1./(nbResels))/sigma

    #we plot the comparison
    plt.figure(2)
    plt.plot(rd.r,penalty_factor_2d,'ro',label='2D from function',fillstyle='none')
    plt.plot(rd.r,penalty_factor_1d,'bo',label='1D from function',fillstyle='none')
    plt.plot(sep,penalty_2d,'rx',label='2D check')
    plt.plot(sep,penalty_1d,'bx',label='1D check')
    plt.xlabel('Separation in resolution elements')
    plt.ylabel('Penalty term for a {0:d}$\sigma$ threshold'.format(5))
    plt.legend(frameon=False)
    plt.grid()
