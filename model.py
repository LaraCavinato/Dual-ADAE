import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt

import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf

from numpy.random import seed
# from tensorflow import set_random_seed

import keras as ke
import keras.backend as K
from keras.layers import Input, Dense, Dropout
from keras.models import Model

from utils import *

import json

create_gif = False


# Class for Dual AD-AE training with 2 layers AE
class DualADAE(object):

    def __init__(self,
                 n_features,
                 latent_dim1,
                 latent_dim2,
                 lambda_val,
                 random_seed,
                 n_centers,
                 n_scanners):
        
        # Set random seeds
        seed(123456 * random_seed)
        tf.random.set_seed(123456 * random_seed)

        # Set lambda parameters
        self.lambda_val = lambda_val
        self.latent_dim1 = latent_dim1
        self.latent_dim2 = latent_dim2
        
        # Define inputs
        ae_inputs = Input(shape=(n_features,)) 
        adv_inputs = Input(shape=(latent_dim2,))
        
        # Define autoencoder net
        [ae_net, encoder_net, decoder_net] = self._create_autoencoder_net(ae_inputs,
                                                                          n_features,
                                                                          latent_dim1,
                                                                          latent_dim2) 
        print("AE net ")
        ae_net.summary()
        print("Encoder net ")
        encoder_net.summary()
        print("Decoder net ")
        decoder_net.summary()
            
        # Define adversarial nets
        adv_net_1 = self._create_adv_net_1(adv_inputs)
        adv_net_2 = self._create_adv_net_2(adv_inputs)
        
        # Turn on/off network weights
        self._trainable_ae_net = self._make_trainable(ae_net) 
        self._trainable_adv_net_1 = self._make_trainable(adv_net_1)
        self._trainable_adv_net_2 = self._make_trainable(adv_net_2) 
        
        # Compile models
        self._ae = self._compile_ae(ae_net) 
        self._encoder =  self._compile_encoder(encoder_net) 
        self._decoder =  self._compile_decoder(decoder_net) 
        self._ae_w_adv = self._compile_ae_w_adv(ae_inputs,
                                                ae_net,
                                                encoder_net,
                                                adv_net_1,
                                                adv_net_2) 
        self._adv = self._compile_adv(ae_inputs,
                                      ae_net,
                                      encoder_net,
                                      adv_net_1,
                                      adv_net_2)
        
        print("Autoencoder net with adv ")
        self._ae_w_adv.summary()
        
        # Define metrics
        self._val_metrics = None
        self._fairness_metrics = None
        
    # make trainable
    def _make_trainable(self, net):

        def make_trainable(flag):
            net.trainable = flag

            for layer in net.layers:
                layer.trainable = flag

        return make_trainable

       
    # Create autoencoder
    def _create_autoencoder_net(self, inputs, n_features, latent_dim1, latent_dim2):
        
        # Define encoder layers
        dense1 = Dense(n_features, activation='relu')(inputs)
        dropout1 = Dropout(best_ae_params['drop'])(dense1)
        latent_layer1 = Dense(latent_dim1)(dropout1)
        dropout2 = Dropout(best_ae_params['drop'])(latent_layer1)
        latent_layer2 = Dense(latent_dim2)(dropout2)
        
        # Define decoder layers
        dense2 = Dense(latent_dim2, activation='relu')
        dropout3 = Dropout(best_ae_params['drop'])
        dense3 = Dense(latent_dim1, activation='relu')
        dropout4 = Dropout(best_ae_params['drop'])
        outputs = Dense(n_features)
        
        decoded = dense2(latent_layer2)
        decoded = dropout3(decoded)
        decoded = dense3(decoded)
        decoded = dropout4(decoded)
        decoded = outputs(decoded)
        
        autoencoder = Model(inputs=[inputs], outputs=[decoded], name = 'autoencoder')
        encoder = Model(inputs=[inputs], outputs=[latent_layer2], name = 'encoder')
        
        decoder_input = Input(shape=(latent_dim2, )) 
        decoded = dense2(decoder_input)
        decoded = dropout3(decoded)
        decoded = dense3(decoded)
        decoded = dropout4(decoded)
        decoded = outputs(decoded)

        decoder = Model(inputs = decoder_input, outputs=[decoded],  name = 'decoder')
        
        return [autoencoder, encoder, decoder]
     
    # Create adversarial networks:
    # 1. Adv for center
    def _create_adv_net_1(self, inputs):
        dense1 = Dense(50, activation='relu')(inputs)
        dense2 = Dense(50, activation='relu')(dense1)
        outputs = Dense(n_centers, activation='softmax')(dense2)

        return Model(inputs=[inputs], outputs = [outputs],  name = 'adversary_1')
    
    # 2. Adv for scanner
    def _create_adv_net_2(self, inputs):
            dense1 = Dense(50, activation='relu')(inputs)
            dense2 = Dense(50, activation='relu')(dense1)
            outputs = Dense(n_scanners, activation='softmax')(dense2)

            return Model(inputs=[inputs], outputs = [outputs],  name = 'adversary_2')

    # Compile models
    def _compile_ae(self, ae_net):
        ae = ae_net
        self._trainable_ae_net(True)
        ae.compile(loss='mse', metrics = ['mse'], optimizer='adam')

        return ae
    
    # Compile encoder
    def _compile_encoder(self, encoder_net):
        ae = encoder_net
        self._trainable_ae_net(True)
        ae.compile(loss='mse', metrics = ['mse'], optimizer='adam')

        return ae
      
    # Compile decoder
    def _compile_decoder(self, decoder_net):
        ae = decoder_net
        self._trainable_ae_net(True) 
        ae.compile(loss='mse', metrics = ['mse'], optimizer='adam')
        return ae
    
    def auroc(y_true, y_pred):
        return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)

    # Compile autoencoder with adv losses
    # The model takes input features as input
    # Outputs AE + adversarial prediction from the two branches
    
    def _compile_ae_w_adv(self, inputs, ae_net, encoder_net, adv_net_1, adv_net_2):
        ae_w_adv = Model(inputs=[inputs], outputs = [ae_net(inputs)] + 
                                                    [adv_net_1(encoder_net(inputs))] + 
                                                    [adv_net_2(encoder_net(inputs))])
        self._trainable_ae_net(True) # Classifier is trainable
        self._trainable_adv_net_1(False) # Freeze the adversary 1
        self._trainable_adv_net_2(False) # Freeze the adversary 2
        loss_weights = [1., -1 * self.lambda_val,  -1 * self.lambda_val] 
        # ae loss - adversarial losses

        # Now compile the model with three losses and defined weights
        ae_w_adv.compile(loss=['mse', 
                               'sparse_categorical_crossentropy',
                               'sparse_categorical_crossentropy'], # ADV1 = center, ADV2 = scanner
                          metrics= {'autoencoder': 'mse',
                                    'adversary_1':'accuracy',
                                    'adversary_2':'accuracy'}, 
                          loss_weights=loss_weights,
                          optimizer='adam')
        return ae_w_adv

    # Compile adversarial model
    # Takes input features and outputs adversarial prediction
    def _compile_adv(self, inputs, ae_net, encoder_net, adv_net_1, adv_net_2):
        adv = Model(inputs=[inputs], outputs=[adv_net_1(encoder_net(inputs))] + 
                                             [adv_net_2(encoder_net(inputs))])
        self._trainable_ae_net(False) #Freeze the classifier
        self._trainable_adv_net_1(True) #adversarial net is trainable
        self._trainable_adv_net_2(True) #adversarial net is trainable
        adv.compile(loss=['sparse_categorical_crossentropy',
                          'sparse_categorical_crossentropy'], 
                    metrics = {"adversary_1": "accuracy",
                               "adversary_2":"accuracy"},
                    optimizer='adam') 

        return adv
        
    # Pretrain all models
    def pretrain(self, x, z1, z2, validation_data=None, epochs=10):
        self._trainable_ae_net(True)
        self._ae.fit(x.values, x.values, epochs=epochs)
        self._trainable_ae_net(False)
        self._trainable_adv_net_1(True)
        self._trainable_adv_net_2(True)
        
        if validation_data is not None:
            x_val, z1_val, z2_val = validation_data

        self._adv.fit(x.values, (z1.values, z2.values), 
                      validation_data = (x_val.values, (z1_val.values, z2_val.values)),
                        epochs=epochs, verbose=2)
        
    # Now do adversarial training

    def fit(self, x, z1, z2, validation_data=None, T_iter=250, batch_size=128):
        
        if validation_data is not None:
            x_val, z1_val, z2_val = validation_data

        self._val_metrics = pd.DataFrame()
        self._train_metrics = pd.DataFrame()
        
        # Go over all iterations
        for idx in range(T_iter):
            print("Iter ", idx)
            
            if validation_data is not None:
                
                # Predict with encoder
                x_pred = pd.DataFrame(self._ae.predict(x_val), index = x_val.index)
                self._val_metrics.loc[idx, 'MSE'] = mean_squared_error(x_val, x_pred)
                
            # Train adversary
            self._trainable_ae_net(False)
            self._trainable_adv_net_1(True)
            self._trainable_adv_net_2(True)

            print("Training adversaries...")

            history = self._adv.fit(x.values, (z1.values, z2.values), 
                                    validation_data = (x_val.values,
                                                       z1_val.values,
                                                       z2_val.values),
                                    batch_size=batch_size,
                                    epochs=1,
                                    verbose=1)
            self._train_metrics.loc[idx, 'Adversary 1 accuracy'] = history.history['adversary_1_accuracy'][0]
            self._train_metrics.loc[idx, 'Adversary 2 accuracy'] = history.history['adversary_2_accuracy'][0]

            self._val_metrics.loc[idx, 'Adversary 1 accuracy'] = history.history['val_adversary_1_accuracy'][0]
            self._val_metrics.loc[idx, 'Adversary 2 accuracy'] = history.history['val_adversary_2_accuracy'][0]
            
            # Train autoencoder
            self._trainable_ae_net(True)
            self._trainable_adv_net_1(False)
            self._trainable_adv_net_2(False)
            indices = np.random.permutation(len(x))[:batch_size]

            print("Training adversarial autoencoder...")

            history = self._ae_w_adv.fit(x.values[indices],
                                     [x.values[indices]] + [z1.values[indices]] + [z2.values[indices]],
                                     batch_size=batch_size,
                                     epochs=1,
                                     verbose=1,
                                     validation_data = (x_val.values,
                                     ([x_val.values] + [z1_val.values] + [z2_val.values]))) 
            
            print("Autoencoder loss ",  history.history)
            keys = self._ae_w_adv.metrics_names
            
            # Record of interest results
            self._train_metrics.loc[idx, 'Total autoencoder loss'] = history.history['loss'][0]
            self._val_metrics.loc[idx, 'Total autoencoder loss'] = history.history['val_loss'][0]
             
            self._train_metrics.loc[idx, 'Autoencoder MSE'] = history.history['autoencoder_mse'][0]
            self._val_metrics.loc[idx, 'Autoencoder MSE'] = history.history['val_autoencoder_mse'][0]
                
            self._train_metrics.loc[idx, 'Adversary 1 accuracy'] = history.history['adversary_1_accuracy'][0]
            self._val_metrics.loc[idx, 'Adversary 1 accuracy'] = history.history['val_adversary_1_accuracy'][0]

            self._train_metrics.loc[idx, 'Adversary 2 accuracy'] = history.history['adversary_2_accuracy'][0]
            self._val_metrics.loc[idx, 'Adversary 2 accuracy'] = history.history['val_adversary_2_accuracy'][0]
              
    
        # Create plot of losses
        fig, ax = plt.subplots()
        fig.set_size_inches(100, 15)

        SMALL_SIZE = 50
        MEDIUM_SIZE = 60
        BIGGER_SIZE = 70

        plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=MEDIUM_SIZE)   # fontsize of the tick labels
        plt.rc('ytick', labelsize=MEDIUM_SIZE)   # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        plt.plot(self._train_metrics['Total autoencoder loss'], 
                 label = 'Total autoencoder training loss',
                 lw = 5,
                 color = '#27ae60')
        plt.plot(self._val_metrics['Total autoencoder loss'], 
                 label = 'Total autoencoder validation loss',
                 lw = 5,
                 color = '#f39c12')
        
        plt.xlabel('epochs')
        
        # Don't allow the axis to be on top of your data
        ax.set_axisbelow(True)
        ax.minorticks_on()
        ax.grid(which='major', linestyle='-', linewidth='0.5', color='black')
        ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
        ax.legend(bbox_to_anchor=(1.1, 1.05))
        
        plt.show()
        
        # Create plot of losses
        fig, ax = plt.subplots()
        fig.set_size_inches(100, 15)

        SMALL_SIZE = 50
        MEDIUM_SIZE = 60
        BIGGER_SIZE = 70

        plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=MEDIUM_SIZE)   # fontsize of the tick labels
        plt.rc('ytick', labelsize=MEDIUM_SIZE)   # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        plt.plot(self._train_metrics['Autoencoder MSE'], 
                 label = 'Autoencoder training MSE',
                 lw = 5,
                 color = '#3498db')
        plt.plot(self._val_metrics['Autoencoder MSE'], 
                 label = 'Autoencoder validation MSE',
                 lw = 5,
                 color = '#e74c3c')
           
        plt.plot(self._train_metrics['Adversary 1 accuracy'], 
                 label = 'Adversary Center training accuracy',
                 lw = 5,
                 color = '#16a085')

        plt.plot(self._val_metrics['Adversary 1 accuracy'], 
                 label = 'Adversary Center validation accuracy',
                 lw = 5,
                 color = '#9b59b6')
        
        plt.plot(self._train_metrics['Adversary 2 accuracy'], 
                 label = 'Adversary Scanner training accuracy',
                 lw = 5,
                 color = '#ffbb11')

        plt.plot(self._val_metrics['Adversary 2 accuracy'], 
                 label = 'Adversary Scanner validation accuracy',
                 lw = 5,
                 color = '#aabbcc')
         
            
        plt.xlabel('epochs')
        
        # Don't allow the axis to be on top of your data
        ax.set_axisbelow(True)
        ax.minorticks_on()
        ax.grid(which='major', linestyle='-', linewidth='2.0', color='black')
        ax.grid(which='minor', linestyle=':', linewidth='1.5', color='black')
        ax.legend(bbox_to_anchor=(1.1, 1.05))
        
        plt.show()



def make_DualADAE(X_train, Y_train, X_test, Y_test, name_dict_param, run,
                  lambda_val, iternations, n_centers, n_scanners):
    
    # Importing best AE parameter
    param_dict_name = param_path + name_dict_param

    with open(param_dict_name, "r") as fp:
        best_ae_params = json.load(fp)

    latent_dim1 = best_ae_params['hidden_nodes1']
    latent_dim2 = best_ae_params['hidden_nodes2']

    # Define model
    adv_model = DualADAE(n_features=X_train.shape[1], 
                         latent_dim1 = latent_dim1,
                         latent_dim2 = latent_dim2,
                         lambda_val = lambda_val,
                         random_seed = 1,
                         n_centers = 2,
                         n_scanners = 5)

    # Split data
    y1_train = Y_train.iloc[:,0] 
    y2_train = Y_train.iloc[:,1]
    y1_test = Y_test.iloc[:,0]
    y2_test = Y_test.iloc[:,1]


    # Pretrain both networks
    adv_model.pretrain(X_train,
                       y1_train,
                       y2_train,
                       validation_data=(X_test, y1_test, y2_test),
                       epochs=10)

    # Joint training with 6000 iterations  
    adv_model.fit(X_train, Y_train.iloc[:,0], Y_train.iloc[:,1],
                  validation_data=(X_test, Y_test.iloc[:,0], Y_test.iloc[:,1]),
                  T_iter = iternations)

    # Generate embedding for all samples
    embedding = adv_model._encoder.predict(X)
    embedding_df = pd.DataFrame(embedding, index = X.index)

    filename_embedding = adv_path_dual + 
                         '2_layer_parallel_ADV_Embedding_' + 
                         str(latent_dim1) + '_' + 
                         str(latent_dim2) + 'L_lam' + 
                         str(lambda_val) + '_fold' + 
                         str(run) + '_6k_epochs.csv'
    embedding_df.to_csv(filename_embedding)

    # Record models
    model_json = adv_model._encoder.to_json()
    filename_encoder_json = adv_path_dual + 
                            '2_layer_parallel_ADV_encoder_' + 
                            str(latent_dim1) + '_' + 
                            str(latent_dim2) + 'L_lam'+ 
                            str(lambda_val) +'_fold'+ 
                            str(run) +'_6k_epochs.json'

    filename_encoder_h5 = adv_path_dual + 
                          '2_layer_parallel_ADV_encoder_' + 
                          str(latent_dim1) + '_' + 
                          str(latent_dim2) + 'L_lam'+ 
                          str(lambda_val) +'_fold'+ 
                          str(run) +'_6k_epochs.h5'
    
    with open(filename_encoder_json, "w") as json_file:
        json_file.write(model_json)
    
    adv_model._encoder.save_weights(filename_encoder_h5)
    print("Saved model to disk")
     
    model_json = adv_model._decoder.to_json()
    filename_decoder_json = adv_path_dual + 
                            '2_layer_parallel_ADV_decoder_' + 
                            str(latent_dim1) + '_' + 
                            str(latent_dim2) + 'L_lam'+ 
                            str(lambda_val) +'_fold'+ 
                            str(run) +'_6k_epochs.json'

    filename_decoder_h5 = adv_path_dual +
                          '2_layer_parallel_ADV_decoder_' + 
                          str(latent_dim1) + '_' + 
                          str(latent_dim2) + 'L_lam'+ 
                          str(lambda_val) +'_fold'+ 
                          str(run) +'_6k_epochs.h5'
    
    with open(filename_decoder_json, "w") as json_file:
        json_file.write(model_json)
    
    adv_model._decoder.save_weights(filename_decoder_h5)
    print("Saved model to disk")

    return embedding_df
        

