import numpy as np
import scipy


def LRmodel(y_train,x_train):
    y_train = np.array(y_train)
    x_train = np.array(x_train)

    x_train =np.array(x_train)
    row = len(x_train)

    if x_train.ndim == 1:
        col = x_train.ndim
    else:   
        row,col  = x_train.shape
        
    params = np.ones(col+1)

    y_pred =  np.multiply(x_train,params[1:col+1]) + params[0]

    return y_pred

