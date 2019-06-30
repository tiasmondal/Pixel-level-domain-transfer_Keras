from keras.layers import Dense,ZeroPadding2D,UpSampling2D,Flatten
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D 
from keras.layers.core import Flatten
from keras.layers import Input
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers import add,Conv2DTranspose

from keras.layers.core import Reshape

class DiscriminatorR(object):
  def __init__(self, noise_shape):
    self.noise_shape = noise_shape
  def discriminatorR(self):
    gen_input = Input(shape = self.noise_shape)
    model=ZeroPadding2D(padding=2)(gen_input)
    model = Conv2D(filters=128,kernel_size=(5,5),strides=2)(model)
    model=LeakyReLU(alpha=0.2)(model)
    print(model.shape)
    model= Conv2D(filters=256,kernel_size=(5,5),strides=2,padding='same')(model)
    model=LeakyReLU(alpha=0.2)(model)
    model = BatchNormalization()(model)
    print(model.shape)
    model = Conv2D(filters=512,kernel_size=(5,5),strides=2,padding='same')(model)
    model=LeakyReLU(alpha=0.2)(model)
    model = BatchNormalization()(model)
    print(model.shape)
    model = Conv2D(filters=1024,kernel_size=(5,5),strides=2,padding='same')(model)
    model=LeakyReLU(alpha=0.2)(model)
    model = BatchNormalization()(model)
    print(model.shape)
    model=ZeroPadding2D(padding=0)(model)
    model = Conv2D(filters=1 ,kernel_size=(4,4), strides=1)(model)
    model=Activation('sigmoid')(model)
    model=Flatten()(model)
    
    
    model = Dense(1)(model)
    model = Activation('sigmoid')(model)
    
        
    print(model.shape)
    
    discriminator_model=Model(inputs=gen_input , outputs=model)
    return (discriminator_model)
    
    
    
    
