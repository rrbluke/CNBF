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

from utils.mat_helpers import *



#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
class rir_generator(object):


    #--------------------------------------------------------------------------
    def __init__(self, config, set='train'):

        self.set = set
        self.config = config
        self.fs = config['fs']
        self.samples = int(self.fs*1.0)
        self.wlen = config['wlen']
        self.shift = config['shift']
        self.nbin = int(self.wlen/2+1)
        self.set = set

        self.nsrc = config['nsrc']
        assert(self.nsrc == 2)                              # only 2 sources are supported

        self.rir_file = '../loaders/rir_cache.mat'

        self.define_mic_array()
        self.generate_whitening_matrix()
        self.cache_rirs()



    #-------------------------------------------------------------------------
    def define_mic_array(self):

        self.nmic = 6                                       # number of microphones
        self.radius = 46.3/1000                             # radius of the respeaker core v2 microphone array
        self.c = 343.0                                      # speed of sound at 20°C

        #mics 1..6 are on a circle
        self.micpos = np.zeros((self.nmic, 3))
        for m in np.arange(self.nmic):
            a = -2*np.pi*m/self.nmic                        # microphones are arranged clockwise!
            self.micpos[m,0] = self.radius*np.cos(a)
            self.micpos[m,1] = self.radius*np.sin(a)
            self.micpos[m,2] = 0



    #----------------------------------------------------------------------------
    def generate_whitening_matrix(self):

        dist = self.micpos[:,np.newaxis,:] - self.micpos[np.newaxis,:,:]    # shape = (self.nmic, self.nmic, 3)
        dist = np.linalg.norm(dist, axis=-1)                                # shape = (self.nmic, self.nmic)
        tau = dist/self.c

        self.U = np.zeros((self.nbin, self.nmic, self.nmic), dtype=np.complex64)      # whitening matrix
        for k in range(self.nbin):
            fc = self.fs*k/((self.nbin-1)*2)
            Cnn = np.sinc(2*fc*tau)                                         # spherical coherence matrix
            d, E = np.linalg.eigh(Cnn)
            d = np.maximum(d.real, 1e-3)
            iD = np.diag(1/np.sqrt(d))

            # ZCA whitening
            self.U[k,:,:] = np.dot(E, np.dot(iD, E.T.conj()))               # U = E*D^-0.5*E'



    #----------------------------------------------------------------------------
    def whiten_data(self, Fs):

        # U.shape = (nbin, nmic, nmic)
        # Fs.shape = (..., nbin, nmic)
        Fus = np.einsum('kdc, ...kc->...kd', self.U, Fs)                     # shape = (..., nbin, nmic)

        return Fus



    #----------------------------------------------------------------------------
    def cache_rirs(self,):

        if os.path.isfile(self.rir_file):
            data = load_numpy_from_mat(self.rir_file)
            self.rir_A = data['rir_A']                                      # shape = (nrir, samples, nmic)
            self.rir_B = data['rir_B']                                      # shape = (nrir, samples, nmic)
            self.nrir = self.rir_A.shape[0]
            print('Loaded', self.nrir, 'RIRs from', self.rir_file)

        else:
            self.nrir = 500                                                 # pre-calculate <nrir> RIRs
            print('Generating', self.nrir, 'RIRs ...')

            # define room/shoebox
            rt60 = 0.250                                                    # define rt60 of the generated RIRs
            room_dim = np.asarray([6.0, 4.0, 2.5])                          # define room dimensions in [m]
            absorption, max_order = pra.inverse_sabine(rt60, room_dim)      # invert Sabine's formula to obtain the parameters for the ISM simulator

            # create the room
            room = pra.ShoeBox(room_dim, fs=self.fs, materials=pra.Material(absorption), max_order=max_order)

            # place the array in the room
            array_center = np.asarray([2.5 , 1.5, 0.8])
            pos = self.micpos.T + array_center[:,np.newaxis]
            room.add_microphone_array(pos)

            # add <nrir> sources for region A and B to the room
            for r in range(self.nrir):

                # source 1 is randomly placed within region A
                x = np.random.uniform(1.0, 2.0)
                y = np.random.uniform(2.0, 3.0)
                z = np.random.uniform(1.5, 2.0)
                room.add_source([x, y, z], signal=0, delay=0)

                # source 2 is randomly placed within region B
                x = np.random.uniform(3.0, 4.0)
                y = np.random.uniform(2.0, 3.0)
                z = np.random.uniform(1.5, 2.0)
                room.add_source([x, y, z], signal=0, delay=0)


            # compute all RIRs and extend their length to <samples>
            t0 = time.time()
            room.compute_rir()
            t1 = time.time()
            print('Generated', self.nrir, 'RIRs in', t1-t0, 'seconds')


            self.rir_A = np.zeros((self.nrir, self.samples, self.nmic), dtype=np.float32)
            self.rir_B = np.zeros((self.nrir, self.samples, self.nmic), dtype=np.float32)
            for r in range(self.nrir):
                for m in range(self.nmic):

                    h_A = room.rir[m][r*2+0]
                    n = min(self.samples, h_A.size)
                    self.rir_A[r,:n,m] = h_A[:n]

                    h_B = room.rir[m][r*2+1]
                    n = min(self.samples, h_B.size)
                    self.rir_B[r,:n,m] = h_B[:n]


            data = {
                'rir_A': self.rir_A,
                'rir_B': self.rir_B,
            }
            save_numpy_to_mat(self.rir_file, data)



    #----------------------------------------------------------------------------
    def load_rirs(self,):

        h_A = self.rir_A[np.random.choice(self.nrir),:,:]
        h_B = self.rir_B[np.random.choice(self.nrir),:,:]

        return h_A, h_B


