#Erosion
import numpy as np
import NoiseModels
import scipy.ndimage as nd
import RGB
import matplotlib.pyplot as plt
image =plt.imread('morph_dilate.png')
image=RGB.rgb2gray(image)

image[image>0]=1
r,c =image.shape
print(r,c)

Padmage= np.zeros([r+2,c+2])

Eroded= np.zeros([r,c])
Dilated= np.zeros([r,c])

Padmage[1:r+1,1:c+1]=image[:,:]
SE= np.ones([3,3])
r1,c1 =image.shape
rowcount=0
for i in range(1,r1-1):
    colcount=0
    for j in range(1, c1-1):
        subimage=Padmage[i-1:rowcount+3,j-1:colcount+3]
        g = SE * subimage
        Eroded[i,j]=np.min(g[:])
        Dilated[i, j] = np.max(g[:])
        colcount+=1
    rowcount+=1


plt.subplot(3,1,1)
plt.title("original")
plt.imshow(image,cmap=plt.get_cmap('gray'))
plt.subplot(3,1,2)
plt.imshow(Eroded,cmap=plt.get_cmap('gray'))
plt.title("Eroded")
plt.subplot(3,1,3)
plt.title("Dilated")
plt.imshow(Dilated,cmap=plt.get_cmap('gray'))
plt.show()
