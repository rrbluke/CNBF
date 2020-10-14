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
from keras.layers import Dense, Activation, LSTM, Input, Lambda, Bidirectional
import keras.backend as K
import tensorflow as tf

from loaders.feature_generator import feature_generator
from utils.mat_helper import save_numpy_to_mat, load_numpy_from_mat
from utils.keras_helpers import *
from ops.complex_ops import *
from ops.complex_layers import *
from ops.kernelized_layers import *


np.set_printoptions(precision=3, threshold=3, edgeitems=3)



#-------------------------------------------------------------------------
#-------------------------------------------------------------------------


class cnbf(object):

    def __init__(self, config, fgen):

        self.config = config
        self.fgen = fgen
        self.name = 'cnbf_complex'
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

        vz = vector_normalize_magnitude(Fz)                                         # shape = (nbatch, nfram, nbin, nmic)
        vz = vector_normalize_phase(vz)                                             # shape = (nbatch, nfram, nbin, nmic)

        return vz



    #---------------------------------------------------------
    def layer1(self, inp):

        X = tf.cast(inp, tf.complex64)                                             # shape = (nbatch, nfram, nbin, nmic)
        
        G = tf.reduce_mean(X[...,-1], axis=-1, keepdims=True)
        G = tf.concat([G]*self.nbin, axis=-1)[...,tf.newaxis]
        X = tf.concat([X,G], axis=-1)

        return X



    #---------------------------------------------------------
    def layer2(self, inp):

        Fs = tf.cast(inp[0], tf.complex64)                                          # shape = (nbatch, nfram, nbin, nmic)
        Fn = tf.cast(inp[1], tf.complex64)                                          # shape = (nbatch, nfram, nbin, nmic)
        W = tf.cast(inp[2], tf.complex64)                                           # shape = (nbatch, nfram, nbin, nmic)

        # beamforming
        W = vector_normalize_magnitude(W)                                           # shape = (nbatch, nfram, nbin, nmic)
        W = vector_normalize_phase(W)                                               # shape = (nbatch, nfram, nbin, nmic)
        Fys = vector_conj_inner(Fs, W)                                              # shape = (nbatch, nfram, nbin)
        Fyn = vector_conj_inner(Fn, W)                                              # shape = (nbatch, nfram, nbin)
        Fy = Fys+Fyn

        # energy of the beamformed outputs
        Pys = elementwise_abs2(Fys)
        Pyn = elementwise_abs2(Fyn)
        Lys = 10*log10(tf.reduce_mean(Pys, axis=(1,2)) + 1e-3)
        Lyn = 10*log10(tf.reduce_mean(Pyn, axis=(1,2)) + 1e-3)

        # energy of the inputs
        Ps = elementwise_abs2(Fs)
        Pn = elementwise_abs2(Fn)
        Ls = 10*log10(tf.reduce_mean(Ps, axis=(1,2,3)) + 1e-3)
        Ln = 10*log10(tf.reduce_mean(Pn, axis=(1,2,3)) + 1e-3)

        delta_snr = Lys-Lyn - (Ls-Ln)

        cost = -tf.reduce_mean(delta_snr)

        return [Fy, cost]




    #---------------------------------------------------------
    def create_model(self):

        print('*** creating model: %s' % self.name)

        # shape definitions: (nbatch, ...)
        Fs = Input(batch_shape=(self.nbatch, self.nfram, self.nbin, self.nmic), dtype=tf.complex64)
        Fn = Input(batch_shape=(self.nbatch, self.nfram, self.nbin, self.nmic), dtype=tf.complex64)

        Fz = Fs+Fn
        X = Lambda(self.layer0)(Fz)                                                            # shape = (nbatch, nfram, nbin, nmic)
        X = Kernelized_Complex_LSTM(units=self.nmic)(X)                                        # shape = (nbatch, nfram, nbin, nmic)
        X = Kernelized_Complex_Dense(units=self.nmic, activation='tanh')(X)                    # shape = (nbatch, nfram, nbin, nmic)
        X = Lambda(self.layer1)(X)
        X = Kernelized_Complex_LSTM(units=self.nmic)(X)                                        # shape = (nbatch, nfram, nbin, nmic)
        X = Kernelized_Complex_Dense(units=self.nmic, activation='tanh')(X)                    # shape = (nbatch, nfram, nbin, nmic)
        W = Kernelized_Complex_Dense(units=self.nmic, activation='linear')(X)                  # shape = (nbatch, nfram, nbin, nmic)

        Fy, cost = Lambda(self.layer2)([Fs, Fn, W])

        self.model = Model(inputs=[Fs, Fn], outputs=Fy)
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
        Fy = self.model.predict([Fs, Fn])

        data = {
            'Fs': np.transpose(Fs, [0,2,1,3])[0,:,:,0],                    # shape = (nbin, nfram)
            'Fn': np.transpose(Fn, [0,2,1,3])[0,:,:,0],                    # shape = (nbin, nfram)
            'Fy': np.transpose(Fy, [0,2,1])[0,:,:],                        # shape = (nbin, nfram)
        }
        save_numpy_to_mat(self.predictions_file, data)



    #---------------------------------------------------------
    def check_date(self):

        if (self.creation_date == os.path.getmtime(self.name+'.py')):
            return True
        else:
            return False




#---------------------------------------------------------
#---------------------------------------------------------
if __name__ == "__main__":


    # parse command line args
    parser = argparse.ArgumentParser(description='speaker identification')
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
        print('train the model')
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
        print('predict the model')
        bf.save_prediction()




