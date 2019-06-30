discriminatorA=DiscriminatorA((64,64,6)).discriminatorA()
import numpy as np
import os
import matplotlib.pyplot as plt
os.chdir('/content/drive/My Drive/Pixel level domain transfer')
img1=plt.imread('im1.jpg')
img2=plt.imread('im1.jpg')
l=[]
l.append(img1)
l.append(img2)
#print(plt.imshow(l[0]))
#print(np.sum(img1-img2))
im=np.concatenate((img1,img2),axis=2)
output=discriminatorA.predict(im.reshape(1,im.shape[0],im.shape[1],im.shape[2]))
print(output)
#print(plt.imshow(img2))
