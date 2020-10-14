# -*- coding: utf-8 -*-
__author__ = "Lukas Pfeifenberger"


import os
import sys

import tensorflow as tf
from keras.layers import Layer, RNN
from keras import activations

sys.path.append(os.path.abspath('../'))
from utils.keras_helpers import *
from ops.complex_ops import *



#-------------------------------------------------------------------

class Kernelized_Dense(Layer):

    def __init__(self, units, activation='tanh'):

        super(Kernelized_Dense, self).__init__()
        self.units = units
        self.activation = activations.get(activation)


    def build(self, input_shape):

        # input_shape = (..., nkernels, n_in)
        nkernels = input_shape[-2]
        n_in = input_shape[-1]

        self.W = self.add_weight(name='W', shape=(nkernels, n_in, self.units), initializer='random_normal', dtype=tf.float32)
        self.b = self.add_weight(name='b', shape=(nkernels, self.units), initializer='zeros', dtype=tf.float32)

        super(Kernelized_Dense, self).build(input_shape)


    def call(self, inputs):

        x = inputs              # shape = (..., nkernels, n_in)

        z = tf.einsum('...ki,kij->...kj', x, self.W) + self.b      # shape = (..., nkernels, units)

        if self.activation is not None:
            z = self.activation(z)

        return z


    def compute_output_shape(self, input_shape):

        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)





#-------------------------------------------------------------------

class Kernelized_LSTM(Layer):

    def __init__(self, units, activation='tanh', recurrent_activation='hard_sigmoid', return_sequences=True, go_backwards=False):

        super(Kernelized_LSTM, self).__init__()
        self.units = units
        self.activation = activation
        self.recurrent_activation = recurrent_activation
        self.return_sequences = return_sequences
        self.go_backwards = go_backwards


    def build(self, input_shape):

        # input to the kernelized LSTM is a 4D tensor:
        nbatch, nfram, kernels, n_in = input_shape

        cell = self.Cell(kernels, self.units, self.activation, self.recurrent_activation)
        self.rnn = RNN(cell, return_sequences=self.return_sequences, go_backwards=self.go_backwards)

        # the Keras RNN implementation does only work with 3D tensors, hence we flatten the last two dimensions of the input:
        self.rnn.build(input_shape=(nbatch, nfram, kernels*n_in))
        self._trainable_weights = self.rnn.trainable_weights
        super(Kernelized_LSTM, self).build(input_shape)


    def call(self, inputs):

        x = inputs                        # shape = (nbatch, nfram, kernels, n_in)

        # reshape input to 3D
        nbatch = tf.shape(x)[0]
        nfram = tf.shape(x)[1]
        kernels = tf.shape(x)[2]
        x = tf.reshape(x, [nbatch, nfram, -1])

        # reshape output to 4D
        y = self.rnn(x)
        y = tf.reshape(y, [nbatch, nfram, kernels, self.units])

        # reverse time axis back to normal
        if self.go_backwards is True:
            y = tf.reverse(y, axis=[1])
        
        return y


    def compute_output_shape(self, input_shape):

        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)



    class Cell(Layer):

        def __init__(self, kernels, units, activation='tanh', recurrent_activation='hard_sigmoid'):

            super(Kernelized_LSTM.Cell, self).__init__()
            self.activation = activations.get(activation)
            self.recurrent_activation = activations.get(recurrent_activation)
            self.units = units                                              # = data size of the output
            self.kernels = kernels                                          # = kernel size of the output
            self.state_size = (kernels*units, kernels*units)                # = flattened sizes of the hidden and carry state
            self.output_size = kernels*units                                # = flattened size of the output


        def build(self, input_shape):

            # the input of the Cell is a 3D tensor with shape (nbatch, nfram, kernels*n_in)
            n_in = int(input_shape[-1]/self.kernels)

            self.W = self.add_weight(shape=(self.kernels, n_in, self.units*4), name='W', initializer='glorot_uniform')
            self.U = self.add_weight(shape=(self.kernels, self.units, self.units*4), name='U', initializer='orthogonal')
            self.b = self.add_weight(shape=(self.kernels, self.units*4), name='b', initializer='zeros')

            super(Kernelized_LSTM.Cell, self).build(input_shape)


        # this function is called every <nfram> time steps
        def call(self, inputs, states, training=None):

            x = inputs                      # shape = (nbatch, kernels*n_in)
            nbatch = tf.shape(x)[0]
            x = tf.reshape(x, [nbatch, self.kernels, -1])                          # expand input to 3D
            h_tm1 = tf.reshape(states[0], [nbatch, self.kernels, self.units])      # expand previous hidden state to 3D
            c_tm1 = tf.reshape(states[1], [nbatch, self.kernels, self.units])      # expand previous carry state to 3D


            z =  tf.einsum('...ki,kij->...kj', x, self.W)                           # shape = (..., kernels, units*4)
            z += tf.einsum('...ki,kij->...kj', h_tm1, self.U)                       # shape = (..., kernels, units*4)
            z += self.b

            a, i, f, o = [ z[..., i*self.units:(i+1)*self.units] for i in range(4) ]

            a = self.activation(a)
            i = self.recurrent_activation(i)
            f = self.recurrent_activation(f)
            o = self.recurrent_activation(o)

            c = a*i + f*c_tm1
            h = o*self.activation(c)
            

            # flatten new hidden and carry state back to 2D
            h = tf.reshape(h, [nbatch, -1])                   # shape = (nbatch, kernels*units)
            c = tf.reshape(c, [nbatch, -1])
            
            return h, [h, c]



