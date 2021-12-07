import numpy as np
import random
import scipy.ndimage as nd
import matplotlib.pyplot as plt

def getImage():
 image= np.zeros([500,500])
 image[:,:]=0
 # Cnt=250
 # factor=0
 r,c = image.shape

 # Create Inner Square With 0.5 intensity
 image[50:450,50:450]=0.5
 # for i in range(200,250):  ### diamond pattern
 #         image[i,Cnt-factor:Cnt+factor]=255
 #         factor=factor+2
 #
 # for i in range(250,300):
 #         image[i,Cnt-factor:Cnt+factor]=255
 #         factor=factor-2


 radius=150 # radius of inner circle
 shiftr = int((r/2) - radius)  # shift index to center as python's origin is defined at top-left of the image
 shiftc = int((c/2) - radius)
 for row in range(0,r):
     for col in range(0,c):
         if (col-radius)**2 + (row-radius)**2 <= radius**2: # circle equation
             image[row+shiftr,col+shiftc] = 1
 return image
# image=getImage()
# r,c = image.shape
# selectnoise=1 #select noise
# Copyimage=image
# noiseimage=np.zeros([r,c]) #initialise noise image
#
# noise=np.zeros([r,c])
# if selectnoise==1:  ### GAUSSiAN NOISE
#  noise = np.random.normal(0,0.1,[r,c])
#  noiseimage = noise + image
#  filtered_image = nd.gaussian_filter(noiseimage, 1)
#
# elif selectnoise==2:  ### Rayleigh NOISE
#
#  noise = np.random.rayleigh(0.1,[r,c])
#  noiseimage = noise + image
#  filtered_image = nd.gaussian_filter(noiseimage, 1)
#
# elif selectnoise==3: ### Gamma NOISE
#
#  noise = np.random.gamma(1,0.1,[r,c])
#  noiseimage = noise + image
#  filtered_image = nd.gaussian_filter(noiseimage, 1)
#
# elif selectnoise==4: ### Exponential NOISE
#
#  noise = np.random.exponential(0.1,[r,c])
#  noiseimage = noise + image
#  filtered_image = nd.gaussian_filter(noiseimage, 1)
#
# elif selectnoise==5: ### Uniform NOISE
#
#  noise = np.random.uniform(0,1,[r,c])
#  noiseimage = noise + image
#  filtered_image = nd.gaussian_filter(noiseimage, 1)
#
# elif selectnoise==6: ### Salt and Pepper
#  # percentage of salt and Pepper
#  Nimage=np.zeros([r,c])
#  Ps=0.2
#  Pp=0.2
#  s=[]
#  t=[]
#  Nimage[:,:]=Copyimage
#  for i in range(0,r):
#   for j in range(0,c):
#    if Nimage[i,j]==1:
#     s.append(i)
#     t.append(j)
#
#  ind=[]
#  ind = random.sample(range(1,len(s)),int(Pp*len(s))) #randomly selecting indices
#
#  for i in (ind):
#    Nimage[s[i],t[i]]=0 # assigning Pepper Noise
#  s = []
#  t = []
#  for i in range(0,r):
#   for j in range(0,c):
#
#    if Nimage[i,j]<=0.5:
#     s.append(i)
#     t.append(j)
#  ind=[]
#  ind = random.sample(range(1,len(s)),int(Ps*len(s)))
#
#  for i in (ind):
#    Nimage[s[i],t[i]]=1  # assigning Salt Noise
#
#  noiseimage=Nimage
#  filtered_image = nd.gaussian_filter(Nimage,1)
#
# else: ### Poisson
#
#  noise = np.random.poisson(image,[r,c])
#  noiseimage =noise
#  filtered_image = nd.gaussian_filter(noiseimage, 1)
#
# #histogram initialisations
# valnoise = noiseimage.flatten() #vectorise noise image into 1D
# valim =image.flatten() #vectorise image into 1D
# valfilt = filtered_image.flatten()  #vectorise filtered image into 1D
#
# #plotting
# plt.subplot(3,3,1)
# plt.imshow(image, cmap=plt.get_cmap('gray'))
# plt.title('Original Image')
#
# plt.subplot(3,3,2)
# plt.hist(valim)
# plt.title('Histogram Original Image')
#
# plt.subplot(3,3,3)
# f = np.fft.fftshift(np.fft.fft2(image))
# plt.imshow(20*np.log(abs(f)), cmap=plt.get_cmap('gray'))
# plt.title('Fourier Transform')
#
# plt.subplot(3,3,4)
# plt.imshow(noiseimage, cmap=plt.get_cmap('gray'))
# plt.title('Noise Image')
#
#
# plt.subplot(3,3,5)
# plt.hist(valnoise,256)
# plt.title('Histogram Noise Image')
#
# plt.subplot(3,3,6)
# f = np.fft.fftshift(np.fft.fft2(noiseimage))
# plt.imshow(20*np.log(abs(f)), cmap=plt.get_cmap('gray'))
# plt.title('Fourier Transform')
#
# plt.subplot(3,3,7)
# plt.imshow(filtered_image, cmap=plt.get_cmap('gray'))
# plt.title('Filtered Image')
#
# plt.subplot(3,3,8)
# plt.hist(valfilt,256)
# plt.title('Histogram Filtered Image')
#
#
# plt.subplot(3,3,9)
# f = np.fft.fftshift(np.fft.fft2(filtered_image))
# plt.imshow(20*np.log(abs(f)), cmap=plt.get_cmap('gray'))
# plt.title('Fourier Transform')
#
# plt.show()
#
