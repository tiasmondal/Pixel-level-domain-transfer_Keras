import keras.backend as K
global d_loss
global d_loss1
def loss_converter(y_true,y_pred):
  return(K.sum(y_pred))
def loss_discriminatorR1(y_true,y_pred):
  
  
  #return(K.sum(y_pred))
  loss=-K.sum(y_true)*K.log(K.abs(K.sum(y_pred)))+(K.sum(y_true)-1)*(K.log(1-K.abs(K.sum(y_pred))))
  
  
    
  return loss
    
