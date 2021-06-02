import numpy as np
import matplotlib.pyplot as plt

SIZE = 255
center = SIZE//2

def dist_r(x,y):
    return ((x-center)**2+(y-center)**2)**0.5

img = np.zeros((SIZE,SIZE))

for i in range(SIZE):
    for j in range(SIZE):
        d = dist_r(i,j)
        if d>112 and d <= 113:
            img[i,j] = 455
        if d>113 and d <=114:
            img[i,j] = 100

c = plt.imshow(img, interpolation='nearest', origin='lower',extent=(0,3,0,3))
plt.colorbar(c)
plt.title('lala')
plt.show()
