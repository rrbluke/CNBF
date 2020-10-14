# -*- coding: utf-8 -*-
__author__ = "Lukas Pfeifenberger"


import os
import sys

import tensorflow as tf
from keras.layers import Layer, RNN

sys.path.append(os.path.abspath('../'))
from utils.keras_helpers import *
from ops.complex_ops import *



#-------------------------------------------------------------------

class Complex_Dense(Layer):

    def __init__(self, units, activation=None):

        super(Complex_Dense, self).__init__()
        self.units = units
        self.activation = activation


    def build(self, input_shape):

        # input_shape = (..., n_in)

        # keras does not allow dtype=tf.complex64 on trainable weights, workaround:
        self.W_real = self.add_weight(name='W_real', shape=(input_shape[-1], self.units), initializer='random_normal', dtype=tf.float32)
        self.W_imag = self.add_weight(name='W_imag', shape=(input_shape[-1], self.units), initializer='random_normal', dtype=tf.float32)

        self.b_real = self.add_weight(name='b_real', shape=(self.units,), initializer='zeros', dtype=tf.float32)
        self.b_imag = self.add_weight(name='b_imag', shape=(self.units,), initializer='zeros', dtype=tf.float32)

        super(Complex_Dense, self).build(input_shape)


    def call(self, inputs):

        x = inputs

        W = cast_to_complex(self.W_real, self.W_imag)
        b = cast_to_complex(self.b_real, self.b_imag)

        z = einsum('...i,ij->...j', x, W) + b      # shape = (..., units)

        if self.activation == 'tanh':
            z = elementwise_tanh(z)
        elif self.activation == 'norm':
            z = vector_normalize_magnitude(z)

        return z


    def compute_output_shape(self, input_shape):

        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)




#-------------------------------------------------------------------

class Complex_LSTM(Layer):

    def __init__(self, units, activation='tanh', return_sequences=True, go_backwards=False):

        super(Complex_LSTM, self).__init__()
        self.units = units
        self.activation = activation
        self.return_sequences = return_sequences
        self.go_backwards = go_backwards


    def build(self, input_shape):

        cell = self.Cell(self.units)
        self.rnn = RNN(cell, return_sequences=self.return_sequences, go_backwards=self.go_backwards)
        self.rnn.build(input_shape=input_shape)
        self._trainable_weights = self.rnn.trainable_weights
        super(Complex_LSTM, self).build(input_shape)


    def call(self, inputs):

        x = inputs                        # shape = (nbatch, nfram, n_in)

        # reshape input to 3D
        nbatch = tf.shape(x)[0]
        nfram = tf.shape(x)[1]

        # reshape output to 4D
        y = self.rnn(x)
        y = tf.reshape(y, [nbatch, nfram, self.units])

        # reverse time axis back to normal
        if self.go_backwards is True:
            y = tf.reverse(y, axis=[1])

        return y


    def compute_output_shape(self, input_shape):

        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)



    class Cell(Layer):

        def __init__(self, units, activation='tanh', recurrent_activation='sigmoid'):

            super(Complex_LSTM.Cell, self).__init__()
            self.units = units                                              # = kernel size of the output
            self.activation = activation
            self.state_size = (units, units)                                # = flattened sizes of the hidden and carry state
            self.output_size = units                                        # = flattened size of the output


        def build(self, input_shape):

            # the input of the Cell is a 3D tensor with shape (nbatch, nfram, n_in)
            n_in = input_shape[-1]

            self.W_real = self.add_weight(shape=(n_in, self.units*4), name='W_real', initializer='glorot_uniform')
            self.U_real = self.add_weight(shape=(self.units, self.units*4), name='U_real', initializer='orthogonal')
            self.b_real = self.add_weight(shape=(self.units*4), name='b_real', initializer='zeros')

            self.W_imag = self.add_weight(shape=(n_in, self.units*4), name='W_imag', initializer='glorot_uniform')
            self.U_imag = self.add_weight(shape=(self.units, self.units*4), name='U_imag', initializer='orthogonal')
            self.b_imag = self.add_weight(shape=(self.units*4), name='b_imag', initializer='zeros')

            super(Complex_LSTM.Cell, self).build(input_shape)


        # this function is called every <nfram> time steps
        def call(self, inputs, states, training=None):

            x = inputs                                              # shape = (nbatch, n_in)
            h_tm1 = states[0]                                       # shape = (nbatch, units)
            c_tm1 = states[1]                                       # shape = (nbatch, units)


            W = cast_to_complex(self.W_real, self.W_imag)
            U = cast_to_complex(self.U_real, self.U_imag)
            b = cast_to_complex(self.b_real, self.b_imag)

            z =  einsum('bi,ij->bj', x, W)                          # shape = (nbatch, units*4)
            z += einsum('bi,ij->bj', h_tm1, U)                      # shape = (nbatch, units*4)
            z += b

            a, i, f, o = [ z[:,i*self.units:(i+1)*self.units] for i in range(4) ]

            a = elementwise_tanh(a)
            i = elementwise_sigmoid(i)
            f = elementwise_sigmoid(f)
            o = elementwise_sigmoid(o)

            c = a*i + f*c_tm1

            if self.activation == 'tanh':
                h = o*elementwise_tanh(c)
            elif self.activation == 'norm':
                h = o*vector_normalize_magnitude(c)
            else:
                h = o*c
           
            return h, [h, c]




#-------------------------------------------------------------------

class Kernelized_Complex_Dense(Layer):

    def __init__(self, units, activation=None):

        super(Kernelized_Complex_Dense, self).__init__()
        self.units = units
        self.activation = activation


    def build(self, input_shape):

        # input_shape = (..., kernels, n_in)

        # keras does not allow dtype=tf.complex64 on trainable weights, workaround:
        self.W_real = self.add_weight(name='W_real', shape=(input_shape[-2], input_shape[-1], self.units), initializer='random_normal', dtype=tf.float32)
        self.W_imag = self.add_weight(name='W_imag', shape=(input_shape[-2], input_shape[-1], self.units), initializer='random_normal', dtype=tf.float32)

        self.b_real = self.add_weight(name='b_real', shape=(input_shape[-2], self.units), initializer='zeros', dtype=tf.float32)
        self.b_imag = self.add_weight(name='b_imag', shape=(input_shape[-2], self.units), initializer='zeros', dtype=tf.float32)

        super(Kernelized_Complex_Dense, self).build(input_shape)


    def call(self, inputs):

        x = inputs

        W = cast_to_complex(self.W_real, self.W_imag)
        b = cast_to_complex(self.b_real, self.b_imag)

        z = einsum('...ki,kij->...kj', x, W) + b      # shape = (..., kernels, units)

        if self.activation == 'tanh':
            z = elementwise_tanh(z)
        elif self.activation == 'norm':
            z = vector_normalize_magnitude(z)

        return z


    def compute_output_shape(self, input_shape):

        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)



#-------------------------------------------------------------------

class Kernelized_Complex_LSTM(Layer):

    def __init__(self, units, return_sequences=True, go_backwards=False):

        super(Kernelized_Complex_LSTM, self).__init__()
        self.units = units
        self.return_sequences = return_sequences
        self.go_backwards = go_backwards


    def build(self, input_shape):

        # input to the kernelized LSTM is a 4D tensor:
        nbatch, nfram, kernels, n_in = input_shape

        cell = self.Cell(kernels, self.units)
        self.rnn = RNN(cell, return_sequences=self.return_sequences, go_backwards=self.go_backwards)

        # the Keras RNN implementation does only work with 3D tensors, hence we flatten the last two dimensions of the input:
        self.rnn.build(input_shape=(nbatch, nfram, kernels*n_in))
        self._trainable_weights = self.rnn.trainable_weights
        super(Kernelized_Complex_LSTM, self).build(input_shape)


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

            super(Kernelized_Complex_LSTM.Cell, self).__init__()
            self.units = units                                              # = kernel size of the output
            self.kernels = kernels                                          # = data size of the output
            self.state_size = (kernels*units, kernels*units)                # = flattened sizes of the hidden and carry state
            self.output_size = kernels*units                                # = flattened size of the output


        def build(self, input_shape):

            # the input of the Cell is a 3D tensor with shape (nbatch, nfram, kernels*n_in)
            n_in = int(input_shape[-1]/self.kernels)

            self.W_real = self.add_weight(shape=(self.kernels, n_in, self.units*4), name='W_real', initializer='glorot_uniform')
            self.U_real = self.add_weight(shape=(self.kernels, self.units, self.units*4), name='U_real', initializer='orthogonal')
            self.b_real = self.add_weight(shape=(self.kernels, self.units*4), name='b_real', initializer='zeros')

            self.W_imag = self.add_weight(shape=(self.kernels, n_in, self.units*4), name='W_imag', initializer='glorot_uniform')
            self.U_imag = self.add_weight(shape=(self.kernels, self.units, self.units*4), name='U_imag', initializer='orthogonal')
            self.b_imag = self.add_weight(shape=(self.kernels, self.units*4), name='b_imag', initializer='zeros')

            super(Kernelized_Complex_LSTM.Cell, self).build(input_shape)


        # this function is called every <nfram> time steps
        def call(self, inputs, states, training=None):

            x = inputs                      # shape = (nbatch, kernels*n_in)
            nbatch = tf.shape(x)[0]
            x = tf.reshape(x, [nbatch, self.kernels, -1])                          # expand input to 3D
            h_tm1 = tf.reshape(states[0], [nbatch, self.kernels, self.units])      # expand previous hidden state to 3D
            c_tm1 = tf.reshape(states[1], [nbatch, self.kernels, self.units])      # expand previous carry state to 3D


            W = cast_to_complex(self.W_real, self.W_imag)
            U = cast_to_complex(self.U_real, self.U_imag)
            b = cast_to_complex(self.b_real, self.b_imag)

            z =  einsum('...ki,kij->...kj', x, W)                                 # shape = (..., kernels, units*4)
            z += einsum('...ki,kij->...kj', h_tm1, U)                             # shape = (..., kernels, units*4)
            z += b

            a, i, f, o = [ z[..., i*self.units:(i+1)*self.units] for i in range(4) ]

            a = elementwise_tanh(a)
            i = elementwise_sigmoid(i)
            f = elementwise_sigmoid(f)
            o = elementwise_sigmoid(o)

            c = a*i + f*c_tm1
            h = o*elementwise_tanh(c)


            # flatten new hidden and carry state back to 2D
            h = tf.reshape(h, [nbatch, -1])                   # shape = (nbatch, kernels*units)
            c = tf.reshape(c, [nbatch, -1])
            
            return h, [h, c]



