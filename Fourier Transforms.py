import numpy as np

import random
import scipy.ndimage as nd
import matplotlib.pyplot as plt

def dft(input):
    ft = np.fft.ifftshift(input)
    ft = np.fft.fft2(ft)
    return np.fft.fftshift(ft)

def dift(input):
    ift = np.fft.ifftshift(input)
    ift = np.fft.ifft2(ift)
    ift = np.fft.fftshift(ift)
    return ift.real



# image= np.zeros([50,50])
# r,c = image.shape
# st=0
# sp=10
# # image[:,26:50]=255
# for i in range(0,3):
#     image[:,st:sp]=1
#     st=sp+10
#     sp=sp+st
x = np.arange(-50,50, 1)
X, Y = np.meshgrid(x, x)
wavelength = 50
image = np.sin(2 * np.pi * Y / wavelength)


plt.subplot(2,1,1)
plt.imshow(image, cmap=plt.get_cmap('gray'))
plt.title('Original Image')

plt.subplot(2,1,2)
plt.imshow(((abs(dft(image)))), cmap=plt.get_cmap('gray'))
plt.title('Fourier Transform')
plt.show()