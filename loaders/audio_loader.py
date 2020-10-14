# -*- coding: utf-8 -*-
__author__ = "Lukas Pfeifenberger"


import numpy as np
import glob
import sys
import os

sys.path.append(os.path.abspath('../'))
from algorithms.audio_processing import *



# loader class for mono wav files, i.e. wsj0

class audio_loader(object):

    # --------------------------------------------------------------------------
    def __init__(self, config, set):

        self.fs = config['fs']
        self.wlen = config['wlen']
        self.shift = config['shift']
        self.samples = int(self.fs*config['duration'])
        self.nfram = int(np.ceil( (self.samples-self.wlen+self.shift)/self.shift ))
        self.nbin = int(self.wlen/2+1)


        if set == 'train':
            path = config['train_path']
        elif set == 'test':
            path = config['test_path']
        elif set == 'eval':
            path = config['eval_path']
        else:
            print('unknown set name: ', set)
            quit(0)

        self.file_list = glob.glob(path+'*.wav')
        self.numof_files = len(self.file_list)

        print('*** audio_loader found %d files in: %s' % (self.numof_files, path))



    #-------------------------------------------------------------------------
    def concatenate_random_files(self,):

        x = np.zeros((self.samples,), dtype=np.float32)
        n = 0
        while n<self.samples:
            f = np.random.choice(self.file_list)
            s, fs = audioread(f)
            length = s.shape[0]
            n1 = min(n+length, self.samples)
            x[n:n1] = s[0:n1-n]
            n = n1

        return x


