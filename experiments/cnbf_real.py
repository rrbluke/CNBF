# -*- coding: utf-8 -*-
__author__ = "Lukas Pfeifenberger"


import time
import numpy as np
import argparse
import json
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append(os.path.abspath('../'))

from keras.models import  Model
from keras.layers import Dense, Activation, LSTM, Input, Lambda
import keras.backend as K
import tensorflow as tf

from loaders.feature_generator import feature_generator
from utils.mat_helpers import *
from utils.keras_helpers import *
from ops.complex_ops import *
from ops.complex_layers import *
from ops.kernelized_layers import *
from algorithms.audio_processing import *
from utils.matplotlib_helpers import *


np.set_printoptions(precision=3, threshold=3, edgeitems=3)



#-------------------------------------------------------------------------
#-------------------------------------------------------------------------


class cnbf(object):

    def __init__(self, config, fgen):

        self.config = config
        self.fgen = fgen
        self.name = 'cnbf_real'
        self.logger = Logger(self.name)

        self.creation_date = os.path.getmtime(self.name+'.py')                                     # timestamp of this script
        self.weights_file = self.config['weights_path'] + self.name + '.h5'
        self.predictions_file = self.config['predictions_path'] + self.name + '.mat'

        self.nbatch = 10
        self.nfram = self.fgen.nfram
        self.nbin = self.fgen.nbin
        self.nmic = self.fgen.nmic

        self.create_model()



    #---------------------------------------------------------
    def layer0(self, inp):

        Fz = tf.cast(inp, tf.complex64)                                             # shape = (nbatch, nfram, nbin, nmic)

        Pz = elementwise_abs2(Fz)                                                   # shape = (nbatch, nfram, nbin, nmic)
        Pz = tf.reduce_mean(Pz, axis=-1)                                            # shape = (nbatch, nfram, nbin)
        Lz = tf.math.log(Pz+1e-3)[...,tf.newaxis]                                   # shape = (nbatch, nfram, nbin, 1)

        vz = vector_normalize_magnitude(Fz)                                         # shape = (nbatch, nfram, nbin, nmic)
        vz = vector_normalize_phase(vz)                                             # shape = (nbatch, nfram, nbin, nmic)

        X = cast_to_float(vz)                                                       # shape = (nbatch, nfram, nbin, nmic, 2)
        X = tf.reshape(X, [self.nbatch, self.nfram, self.nbin, self.nmic*2])        # shape = (nbatch, nfram, nbin, nmic*2)
        X = tf.concat([X, Lz], axis=-1)                                             # shape = (nbatch, nfram, nbin, nmic*2+1)
        Y = tf.reshape(X, [self.nbatch, self.nfram, self.nbin*(self.nmic*2+1)])     # shape = (nbatch, nfram, nbin*(nmic*2+1))

        return [X,Y]



    #---------------------------------------------------------
    def layer1(self, inp):

        X = inp[0]                                                                  # shape = (nbatch, nfram, nbin, nmic*2+1)
        Y = inp[1]                                                                  # shape = (nbatch, nfram, nbin)
        
        X = tf.concat([X,Y[...,tf.newaxis]], axis=-1)

        return X



    #---------------------------------------------------------
    def layer2(self, inp):

        Fs = tf.cast(inp[0], tf.complex64)                                          # shape = (nbatch, nfram, nbin, nmic)
        Fn = tf.cast(inp[1], tf.complex64)                                          # shape = (nbatch, nfram, nbin, nmic)
        W = inp[2]                                                                  # shape = (nbatch, nfram, nbin, nmic*2)

        # beamforming
        W = tf.reshape(W, [self.nbatch, self.nfram, self.nbin, self.nmic, 2])       # shape = (nbatch, nfram, nbin, nmic, 2)
        W = cast_to_complex(W[...,0], W[...,1])
        W = vector_normalize_magnitude(W)                                           # shape = (nbatch, nfram, nbin, nmic)
        W = vector_normalize_phase(W)                                               # shape = (nbatch, nfram, nbin, nmic)
        Fys = vector_conj_inner(Fs, W)                                              # shape = (nbatch, nfram, nbin)
        Fyn = vector_conj_inner(Fn, W)                                              # shape = (nbatch, nfram, nbin)

        # energy of the input
        Ps = tf.reduce_mean(elementwise_abs2(Fs), axis=-1)    # input (desired source)
        Pn = tf.reduce_mean(elementwise_abs2(Fn), axis=-1)    # input (unwanted source)
        Ls = 10*log10(Ps + 1e-2)
        Ln = 10*log10(Pn + 1e-2)

        # energy of the beamformed outputs
        Pys = elementwise_abs2(Fys)                           # output (desired source)
        Pyn = elementwise_abs2(Fyn)                           # output (unwanted source)
        Lys = 10*log10(Pys + 1e-2)
        Lyn = 10*log10(Pyn + 1e-2)

        delta_snr = Lys-Lyn - (Ls-Ln)

        cost = -tf.reduce_mean(delta_snr, axis=(1,2))

        return [Fys, Fyn, cost]




    #---------------------------------------------------------
    def create_model(self):

        print('*** creating model: %s' % self.name)

        # shape definitions: (nbatch, ...)
        Fs = Input(batch_shape=(self.nbatch, self.nfram, self.nbin, self.nmic), dtype=tf.complex64)
        Fn = Input(batch_shape=(self.nbatch, self.nfram, self.nbin, self.nmic), dtype=tf.complex64)

        Fz = Fs+Fn
        X,Y = Lambda(self.layer0)(Fz)
        Y = Dense(units=50, activation='tanh')(Y)                                              # shape = (nbatch, nfram, 50)
        Y = Dense(units=self.nbin, activation='tanh')(Y)                                       # shape = (nbatch, nfram, nbin)
        X = Lambda(self.layer1)([X,Y])
        X = Kernelized_LSTM(units=self.nmic*2, activation='tanh')(X)                           # shape = (nbatch, nfram, nbin, nmic*2)
        X = Kernelized_Dense(units=self.nmic*2, activation='tanh')(X)                          # shape = (nbatch, nfram, nbin, nmic*2)
        W = Kernelized_Dense(units=self.nmic*2, activation='linear')(X)                        # shape = (nbatch, nfram, nbin, nmic*2)

        Fys, Fyn, cost = Lambda(self.layer2)([Fs, Fn, W])

        self.model = Model(inputs=[Fs, Fn], outputs=[Fys, Fyn])
        self.model.add_loss(cost)
        self.model.compile(loss=None, optimizer='adam')

        print(self.model.summary())
        try:
            self.model.load_weights(self.weights_file)
        except:
            print('error loading weights file: %s' % self.weights_file)



    #---------------------------------------------------------
    def train(self):

        Fs, Fn = self.fgen.generate_mixtures(self.nbatch)
        self.model.fit([Fs, Fn], None, batch_size=self.nbatch, epochs=1, verbose=0, shuffle=False, callbacks=[self.logger])



    #---------------------------------------------------------
    def save_weights(self):

        self.model.save_weights(self.weights_file)

        return



    #---------------------------------------------------------
    def save_prediction(self):

        Fs, Fn = self.fgen.generate_mixtures(self.nbatch)
        Fys, Fyn = self.model.predict([Fs, Fn])

        data = {
            'Fs': np.transpose(Fs, [0,2,1,3])[0,:,:,0],                    # shape = (nbin, nfram)
            'Fn': np.transpose(Fn, [0,2,1,3])[0,:,:,0],                    # shape = (nbin, nfram)
            'Fys': np.transpose(Fys, [0,2,1])[0,:,:],                      # shape = (nbin, nfram)
            'Fyn': np.transpose(Fyn, [0,2,1])[0,:,:],                      # shape = (nbin, nfram)
        }
        save_numpy_to_mat(self.predictions_file, data)



    #---------------------------------------------------------
    def check_date(self):

        if (self.creation_date == os.path.getmtime(self.name+'.py')):
            return True
        else:
            return False



    #---------------------------------------------------------
    def inference(self):

        Fs, Fn = self.fgen.generate_mixtures(self.nbatch)
        Fys, Fyn = self.model.predict([Fs, Fn])

        Fs = Fs[0,...,0].T         # input (desired source)
        Fn = Fn[0,...,0].T         # input (unwanted source)
        Fys = Fys[0,...].T         # output (desired source)
        Fyn = Fyn[0,...].T         # output (unwanted source)
        Fz = Fs+Fn                 # noisy mixture
        Fy = Fys+Fyn               # enhanced output

        data = (Fz, Fy)
        filenames = (
                     self.config['predictions_path'] + self.name + '_noisy.wav',
                     self.config['predictions_path'] + self.name + '_enhanced.wav',
                    )
        convert_and_save_wavs(data, filenames)

        Lz = ( 20*np.log10(np.abs(Fs)+1e-1) - 20*np.log10(np.abs(Fn)+1e-1) )/30
        Ly = ( 20*np.log10(np.abs(Fys)+1e-1) - 20*np.log10(np.abs(Fyn)+1e-1) )/30
        legend = ('noisy', 'enhanced')
        clim = (-1, +1)
        filename = self.config['predictions_path'] + self.name + '_prediction.png'
        draw_subpcolor((Lz, Ly), legend, clim, filename)




#---------------------------------------------------------
#---------------------------------------------------------
if __name__ == "__main__":


    # parse command line args
    parser = argparse.ArgumentParser(description='CNBF')
    parser.add_argument('--config_file', help='name of json configuration file', default='../cnbf.json')
    parser.add_argument('--predict', help='inference', action='store_true')
    args = parser.parse_args()


    # load config file
    try:
        print('*** loading config file: %s' % args.config_file )
        with open(args.config_file, 'r') as f:
            config = json.load(f)
    except:
        print('*** could not load config file: %s' % args.config_file)
        quit(0)



    if args.predict is False:
        fgen = feature_generator(config, 'train')
        bf = cnbf(config, fgen)
        print('training')
        i = 0
        while (i<config['epochs']) and bf.check_date():

            bf.train()
            i += 1
            if (i%10)==0:
                bf.save_weights()
                bf.save_prediction()

    else:
        fgen = feature_generator(config, 'test')
        bf = cnbf(config, fgen)
        print('inference')
        bf.inference()




