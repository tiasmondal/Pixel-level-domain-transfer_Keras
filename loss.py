def loss_for_discriminator_A(imageS,chosenimage,set_of_images):
  if(np.sum(chosenimage-set_of_images[0])==0):      #Checking similarity
    t=1
  else:
    t=0;
  image=np.concatenate((chosenimage,imageS),axis=2)
  discriminatorA=DiscriminatorA((64,64,6)).discriminatorA()
  output=discriminatorA.predict(image.reshape(1,image.shape[0],image.shape[1],image.shape[2]))
  output=output.reshape(1,)
  output1=output[0]
  print(output1)
  
  loss=-t*np.log(output1)+(t-1)*(np.log(1-output1))
  
  return loss(image,output)

def loss1(imageS,set_of_images):
  def loss2(imageS=imageS,chosenimage=imagefordisR):
    return(loss_for_discriminator_A(imageS,chosenimage,list))
  return loss2

def loss_for_discriminator_R(image,set_of_images):
  if(np.sum(image-set_of_images[0])==0):
    t=1
  else:
    t=0;
  discriminatorR=DiscriminatorR((64,64,3)).discriminatorR()
  output=discriminatorR.predict(image.reshape(1,image.shape[0],image.shape[1],image.shape[2]))
  output=output.reshape(1,)
  output=output[0]
  print(output)
  loss=-t*np.log(output)+(t-1)*(np.log(1-output))
  return loss
  
