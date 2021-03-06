import numpy as np
import matplotlib.pyplot as plt
import sys
from astropy.io import fits
from matplotlib.widgets import Slider

fig, ax = plt.subplots()

# 1. Open file & get data -> note the data is two dimensions, data[0][0] offers one image n*n
#hd = fits.open('test2.fits')
imageFits = str(sys.argv).split("/")[-1]
hd = fits.open(str(sys.argv[1]))
data = hd[0].data
plt.subplots_adjust(left=0.25, bottom=0.25)

# 2. Find the position interest, not now
'''
res = np.where(data==np.max(data))
print(res.tolist())
print(res[0][0], res[1][0])
'''
print(len(data))
print(len(data[1]))

# 3. matplotlib display our image
indexInitF = 0
indexInitS = 0
currentImage = plt.imshow(data[indexInitF][indexInitS], cmap=plt.cm.hot) #cmap=plt.cm.viridis
axid3 = plt.axes([0.25, 0.15, 0.65, 0.03])
axid4 = plt.axes([0.25, 0.1, 0.65, 0.03])

slider = Slider(axid3, 'Spectral Channel - 1', 0, len(data[0])-1, valinit=indexInitS, valstep = 1)
sliderBis = Slider(axid4, 'Spectral Channel - 2', 0, len(data)-1, valinit=indexInitF, valstep = 1)

def update(val):
    indF = slider.val
    indS = sliderBis.val
    currentImage.set_data(data[int(indS)][int(indF)])
    fig.canvas.draw_idle()

slider.on_changed(update)
sliderBis.on_changed(update)
#plt.imshow(data[0][1], cmap=plt.cm.viridis)
#plt.colorbar()
plt.show()
