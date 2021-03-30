from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

# 1. Open file & get data -> note the data is two dimensions, data[0][0] offers one image n*n
hd = fits.open('test.fits')
data = hd[0].data

# 2. Find the position interest, not now
'''
res = np.where(data==np.max(data))
print(res.tolist())
print(res[0][0], res[1][0])
'''
# 3. matplotlib display our image
indexInitF = 0
indexInitS = 0
currentImage = plt.imshow(data[indexInitF][indexInitS], cmap=plt.cm.viridis)
axidx = plt.axes([0.25, 0.15, 0.65, 0.03])
slider = Slider(axidx, 'index', 0, len(data[0]), valinit=indexInitS, valfmt='%d')

def update(val):
    indF = slider.val
    currentImage.set_data(data[0][int(indF)])
    #fig.canvas.draw_idle()
slider.on_changed(update)
#plt.imshow(data[0][1], cmap=plt.cm.viridis)
plt.xlabel('x-pixels ')
plt.ylabel('y-pixels ')
plt.colorbar()
plt.show()
