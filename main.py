import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
os.chdir('/content/drive/My Drive/Pixel level domain transfer')
image1=plt.imread('im1.jpg')  #unrelated image    
#image2=plt.imread('im1.jpg')
image=plt.imread('im1.jpg') #Let this be ground truth means target
imageS=plt.imread('im1.jpg') #image source

print(imageS.shape)
converter = Converter((64,64,3)).converter()
#image = np.array(image)

#image = denormalize(image)

#converter.compile(loss='binary_crossentropy', optimizer='sgd')
output=converter.predict(imageS.reshape(1,imageS.shape[0],imageS.shape[1],imageS.shape[2]))
output1=output
output=output.reshape(64,64,3)
output_from_converter=output
plt.imshow(output)


############################################# Selecting an image with equal probability ###


list=[]
list.append(image)
list.append(output)
list.append(image1)
imagefordisR =random_image_picking_R(list)

############################################# Real/Fake Discriminator #######################
print("Discriminator Rake/fake network")
#loss_discriminatorR1(image,imagefordisR)
discriminator = DiscriminatorR(output.shape).discriminatorR()                        #Get model
              #Model compile
output=discriminator.predict(image.reshape(1,image.shape[0],image.shape[1],image.shape[2]))             #ground truth true output
output123=discriminator.predict(imagefordisR.reshape(1,64,64,3)) 
print(output)
print(output123)
if(np.sum(imagefordisR-image)==0):
  print("here")
  discriminator.compile(loss=loss_discriminatorR1, optimizer = 'Adam',metrics=['acc'])  
  d_loss = discriminator.evaluate(imagefordisR.reshape(1,64,64,3),[[[1]]])    #was output in place of 1                       #Chosen image and target technique
else:
  discriminator.compile(loss=loss_discriminatorR1, optimizer = 'Adam',metrics=['acc'])  
  d_loss = discriminator.evaluate(imagefordisR.reshape(1,64,64,3),[[[0]]])
#d_loss=d_loss[0]
print("Discriminator Real/fake loss "+ str(d_loss[0]) + ' acc ' +str(d_loss[1]))



############################################## Domain Discriminator #########################
print("Discriminator Domain network")

im=np.concatenate((imagefordisR,imageS),axis=2)                     #picked an image and concatenated with source image

discriminatorA = DiscriminatorA(im.shape).discriminatorA()           #Getting model

output=discriminatorA.predict(im.reshape(1,im.shape[0],im.shape[1],im.shape[2]))                #Chosen image(concatenated with source image) output

im_true=np.concatenate((image,imageS),axis=2)                             #Ground truth concatenated image

output_true=discriminatorA.predict(im_true.reshape(1,im.shape[0],im.shape[1],im.shape[2]))                 #Ground truth(concatenated with source image) output

#discriminatorA.compile(loss='binary_crossentropy', optimizer = 'Adam')    #Model compile
if(np.sum(imagefordisR-image)==0):
  print("here")
  discriminatorA.compile(loss=loss_discriminatorR1, optimizer = 'Adam',metrics=['acc'])  
  d_loss1 = discriminatorA.evaluate(im.reshape(1,64,64,6),[[[1]]])    #was output in place of 1                       #Chosen image and target technique
else:
  discriminatorA.compile(loss=loss_discriminatorR1, optimizer = 'Adam',metrics=['acc'])  
  d_loss1 = discriminatorA.evaluate(im.reshape(1,64,64,6),[[[0]]])
#d_loss1=d_loss1[0]
#d_loss1 = discriminatorA.train_on_batch(im.reshape(1,64,64,6),output_true)                         #used source target technique

print("Discriminator Domain loss "+ str(d_loss1[0]) + ' acc '+str(d_loss1[1]))

############################################### Converter Loss ################################
#converter.save('/content/drive/My Drive/Pixel level domain transfer/model.h5')
#converter.load_weights('/content/drive/My Drive/Pixel level domain transfer/model.h5')
converter.compile(loss='binary_crossentropy', optimizer='sgd')
converter_loss=converter.evaluate(image.reshape(1,image.shape[0],image.shape[1],image.shape[2]),output_from_converter.reshape(1,output_from_converter.shape[0],output_from_converter.shape[1],output_from_converter.shape[2]))
#print("converter loss "+str(converter_loss[0]) + ' acc ' +str(converter_loss[1]))
print("converter loss "+str(converter_loss))
