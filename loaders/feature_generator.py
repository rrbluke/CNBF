# -*- coding: utf-8 -*-
__author__ = "Lukas Pfeifenberger"


import time
import glob
import argparse
import json
import os
import sys
import numpy as np
import pyroomacoustics as pra

sys.path.append(os.path.abspath('../'))

from loaders.audio_loader import audio_loader
from loaders.rir_generator import rir_generator
from algorithms.audio_processing import *
from utils.mat_helpers import *



#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
class feature_generator(object):


    #--------------------------------------------------------------------------
    def __init__(self, config, set='train'):

        self.set = set
        self.config = config
        self.fs = config['fs']
        self.wlen = config['wlen']
        self.shift = config['shift']
        self.samples = int(self.fs*config['duration'])
        self.nfram = int(np.ceil( (self.samples-self.wlen+self.shift)/self.shift ))
        self.nbin = int(self.wlen/2+1)

        self.nsrc = config['nsrc']
        assert(self.nsrc == 2)                              # only 2 sources are supported

        self.audio_loader = audio_loader(config, set)
        self.rgen = rir_generator(config, set)
        self.nmic = self.rgen.nmic



    #---------------------------------------------------------
    def generate_mixture(self,):

        hs, hn = self.rgen.load_rirs()
        s = self.audio_loader.concatenate_random_files()                    # shape = (samples,)
        n = self.audio_loader.concatenate_random_files()                    # shape = (samples,)

        Fhs = rfft(hs, n=self.samples, axis=0)                              # shape = (samples/2+1, nmic)
        Fhn = rfft(hn, n=self.samples, axis=0)                              # shape = (samples/2+1, nmic)

        Fs = rfft(s, n=self.samples, axis=0)                                # shape = (samples/2+1,)
        Fn = rfft(n, n=self.samples, axis=0)                                # shape = (samples/2+1,)

        Fs = Fhs*Fs[:,np.newaxis]
        Fn = Fhn*Fn[:,np.newaxis]

        s = irfft(Fs, n=self.samples, axis=0)                               # shape = (samples, nmic)
        n = irfft(Fn, n=self.samples, axis=0)                               # shape = (samples, nmic)

        Fs = mstft(s.T, self.wlen, self.shift)                              # shape = (nmic, nfram, nbin)
        Fs = np.transpose(Fs, (1,2,0))                                      # shape = (nfram, nbin, nmic)

        Fn = mstft(n.T, self.wlen, self.shift)                              # shape = (nmic, nfram, nbin)
        Fn = np.transpose(Fn, (1,2,0))                                      # shape = (nfram, nbin, nmic)

        Fs = self.rgen.whiten_data(Fs)
        Fn = self.rgen.whiten_data(Fn)

        return Fs, Fn



    #---------------------------------------------------------
    def generate_mixtures(self, nbatch=10):

        Fs = np.zeros(shape=(nbatch, self.nfram, self.nbin, self.nmic), dtype=np.complex64)
        Fn = np.zeros(shape=(nbatch, self.nfram, self.nbin, self.nmic), dtype=np.complex64)
        for b in np.arange(nbatch):

            Fs[b,...], Fn[b,...] = self.generate_mixture()

        return Fs, Fn





#---------------------------------------------------------
#---------------------------------------------------------
if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='mcss feature generator')
    parser.add_argument('--config_file', help='name of json configuration file', default='../cnbf.json')
    args = parser.parse_args()


    with open(args.config_file, 'r') as f:
        config = json.load(f)


    fgen = feature_generator(config, set='train')


    t0 = time.time()
    Fs, Fn = fgen.generate_mixture()
    t1 = time.time()
    print(t1-t0)

    data = {
            'Fs': Fs,
            'Fn': Fn,
           }
    save_numpy_to_mat('../matlab/fgen_check.mat', data)



