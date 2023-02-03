import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from numpy.random import seed
# from tensorflow import set_random_seed --> tf.set_random_seed

import tensorflow as tf

import pathlib
import json
from utils import *

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from tensorflow.keras.layers import Input, Dense, Lambda, Layer, Activation, Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import Sequential
from tensorflow.keras import Model
from tensorflow.keras import backend as K

from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras import metrics, optimizers
from tensorflow.keras.callbacks import Callback



## Define one layer AE model
def create_1AE_model(hidden_nodes1, drop):
   
    # Reconstruction loss
    def reconstruction_loss(x_input, x_decoded):
        return metrics.mse(x_input, x_decoded)

    # Set hyperparameters
    original_dim = data_df.shape[1]
    latent_dim = 50
    
    init_mode = 'glorot_uniform'
    batch_size = 50
    epochs = 50
    learning_rate = 0.0005

    # Define encoder
    x = Input(shape=(original_dim, ))
    d0 = Dropout(drop)(x)
        
    net = Dense(hidden_nodes1, kernel_initializer=init_mode)(d0)
    net2 = BatchNormalization()(net)
    net3 = Activation('relu')(net2)
    d1 = Dropout(drop)(net3)
    
    z = Dense(latent_dim, kernel_initializer=init_mode)(d1)
   
    # Define decoder
    d3 = Dropout(drop)
    decoder_h = Dense(hidden_nodes1, activation='relu', kernel_initializer=init_mode)
    d4 = Dropout(drop)
    decoder_mean = Dense(original_dim, kernel_initializer=init_mode)
    
    h_decoded = d3(z)
    h_decoded2 = decoder_h(h_decoded)
    h_decoded3 = d4(h_decoded2)
    x_decoded_mean = decoder_mean(h_decoded3)

    # AE model
    ae = Model(x, x_decoded_mean)

    adam = optimizers.Adam(learning_rate=learning_rate)
    ae.compile(optimizer=adam,
               loss = reconstruction_loss,
               metrics = [reconstruction_loss])

    ae.summary()

    return ae


## Define two layers AE model
def create_2AE_model(hidden_nodes1, hidden_nodes2, drop):
   
    # Reconstruction loss
    def reconstruction_loss(x_input, x_decoded):
        return metrics.mse(x_input, x_decoded)

    # Set hyperparameters
    original_dim = data_df.shape[1]
    latent_dim = 50
    
    init_mode = 'glorot_uniform'
    batch_size = 50
    epochs = 50
    learning_rate = 0.0005

    # Define encoder
    x = Input(shape=(original_dim, ))
    d0 = Dropout(drop)(x)
        
    net = Dense(hidden_nodes1, kernel_initializer=init_mode)(d0)
    net2 = BatchNormalization()(net)
    net3 = Activation('relu')(net2)
    d1 = Dropout(drop)(net3)
    
    net4 = Dense(hidden_nodes2, kernel_initializer=init_mode)(d1)
    net5 = BatchNormalization()(net4)
    net6 = Activation('relu')(net5)
    d2 = Dropout(drop)(net6)
    
    z = Dense(latent_dim, kernel_initializer=init_mode)(d2)
    
    # Define decoder
    d3 = Dropout(drop)
    decoder_h = Dense(hidden_nodes2, activation='relu', kernel_initializer=init_mode)
    d4 = Dropout(drop)
    decoder_h2 = Dense(hidden_nodes1, activation='relu', kernel_initializer=init_mode)
    d5 = Dropout(drop)
    decoder_mean = Dense(original_dim, kernel_initializer=init_mode)
    
    h_decoded = d3(z)
    h_decoded2 = decoder_h(h_decoded)
    h_decoded3 = d4(h_decoded2)
    h_decoded4 = decoder_h2(h_decoded3)
    h_decoded5 = d5(h_decoded4)
    x_decoded_mean = decoder_mean(h_decoded5)

    # AE model
    ae = Model(x, x_decoded_mean)

    adam = optimizers.Adam(learning_rate=learning_rate)
    ae.compile(optimizer=adam,
               loss = reconstruction_loss,
               metrics = [reconstruction_loss])

    ae.summary()

    return ae


## Define three layers AE model
def create_3AE_model(hidden_nodes1, hidden_nodes2, hidden_nodes3, drop):
   
    # Method for calculating the reconstruction loss
    def reconstruction_loss(x_input, x_decoded):
        return metrics.mse(x_input, x_decoded)

    # Set hyperparameters
    original_dim = data_df.shape[1]
    latent_dim = 50
    
    init_mode = 'glorot_uniform'
    batch_size = 50
    epochs = 50
    learning_rate = 0.0005

    # Define encoder
    x = Input(shape=(original_dim, ))
    d0 = Dropout(drop)(x)
        
    net = Dense(hidden_nodes1, kernel_initializer=init_mode)(d0)
    net2 = BatchNormalization()(net)
    net3 = Activation('relu')(net2)
    d1 = Dropout(drop)(net3)
    
    net4 = Dense(hidden_nodes2, kernel_initializer=init_mode)(d1)
    net5 = BatchNormalization()(net4)
    net6 = Activation('relu')(net5)
    d2 = Dropout(drop)(net6)
    
    net7 = Dense(hidden_nodes3, kernel_initializer=init_mode)(d2)
    net8 = BatchNormalization()(net7)
    net9 = Activation('relu')(net8)
    d3 = Dropout(drop)(net9)
    
    z = Dense(latent_dim, kernel_initializer=init_mode)(d3)
    
    # Define decoder
    d3 = Dropout(drop)
    decoder_h = Dense(hidden_nodes3, activation='relu', kernel_initializer=init_mode)
    d4 = Dropout(drop)
    decoder_h2 = Dense(hidden_nodes2, activation='relu', kernel_initializer=init_mode)
    d5 = Dropout(drop)
    decoder_h3 = Dense(hidden_nodes1, activation='relu', kernel_initializer=init_mode)
    d6 = Dropout(drop)
    decoder_mean = Dense(original_dim, kernel_initializer=init_mode)
    
    h_decoded = d3(z)
    h_decoded2 = decoder_h(h_decoded)
    h_decoded3 = d4(h_decoded2)
    h_decoded4 = decoder_h2(h_decoded3)
    h_decoded5 = d5(h_decoded4)
    h_decoded6 = decoder_h3(h_decoded5)
    h_decoded7 = d6(h_decoded6)
    x_decoded_mean = decoder_mean(h_decoded7)

    # AE model
    ae = Model(x, x_decoded_mean)

    adam = optimizers.Adam(learning_rate=learning_rate)
    ae.compile(optimizer=adam,
               loss = reconstruction_loss,
               metrics = [reconstruction_loss])

    ae.summary()

    return ae


def get_param_grid(nlayers):
    switcher = {
        1: dict(hidden_nodes1 = dims1, drop = dropouts),
        2: dict(hidden_nodes1 = dims1, hidden_nodes2 = dims2, drop = dropouts),
        3: dict(hidden_nodes1 = dims1, hidden_nodes2 = dims2, hidden_nodes3 = dims3, drop = dropouts),
    }
 
    return switcher.get(nlayers, 'Error')



def make_AE_optimization(data_df, dims1, dims2, dims3, dropouts):

    '''
        data_df: dataframe of data
        dims1: list of number of nodes in first hidden layer
        dims2: list of number of nodes in second hidden layer
        dims3: list of number of nodes in third hidden layer
        dropouts: list of dropout values
    '''

    seed(123456)
    tf.random.set_seed(123456)

    # Tuning dimensions
    models = ['create_1AE_model', 'create_2AE_model', 'create_3AE_model']
    params_dict = {m: {} for m in models}


    # Build model(s)
    for i, modelname in enumerate(models):

    	model = KerasRegressor(build_fn = modelname,
                               epochs = 50,
                               batch_size = 50)
    	nlayers = i+1

    	print("------ Training AE with %f layer(s): ------" % (grid_result))

    	param_grid = get_param_grid(nlayers)

    	grid = GridSearchCV(estimator = model,
                            cv = 5, 
                            param_grid = param_grid,
                            scoring = 'neg_root_mean_squared_error',
                            n_jobs = 5)
    	grid_result = grid.fit(np.array(data_df),
                               np.array(data_df),
                               shuffle = True,
                               epochs = 50,
                               batch_size = 50,
                               verbose = 0)

    	# Print results
    	print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    	means = grid_result.cv_results_['mean_test_score']
    	stds = grid_result.cv_results_['std_test_score']
    	params = grid_result.cv_results_['params']
        
    	for mean, stdev, param in zip(means, stds, params):
    	    print("%f (%f) with: %r" % (mean, stdev, param))

    	# Memorizing results
    	params_dict[modelname]['score'] = grid_result.best_score_
    	params_dict[modelname]['params'] = grid_result.best_params_


    # Saving performance and hyperparameters
    param_dict_name = param_path + 'best_AE_param_one_layer.json'
    to_dump = params_dict['create_1AE_model']['params']

    with open(param_dict_name, "w") as fp:
        json.dump(to_dump, fp)

    param_dict_name = param_path + 'best_AE_param_two_layer.json'
    to_dump = params_dict['create_2AE_model']['params']

    with open(param_dict_name, "w") as fp:
        json.dump(to_dump, fp)

    param_dict_name = param_path + 'best_AE_param_three_layer.json'
    to_dump = params_dict['create_3AE_model']['params']

    with open(param_dict_name, "w") as fp:
        json.dump(to_dump, fp)


    print("Saved parameters to disk")











