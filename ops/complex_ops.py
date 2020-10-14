# -*- coding: utf-8 -*-
__author__ = "Lukas Pfeifenberger"


import numpy as np
import tensorflow as tf
import os
import sys
sys.path.append(os.path.abspath('../'))

from utils.keras_helpers import *



#-------------------------------------------------------------------
def safe_conj(z):
    return tf.complex(tf.math.real(z), -tf.math.imag(z))



#-------------------------------------------------------------------
@tf.custom_gradient
def mean_square_error(z, c):

    s = tf.reduce_mean(tf.abs(z-c)**2)

    def grad(grad_s):

        grad_s = tf.cast(grad_s, tf.complex64)
        Nz = tf.cast(tf.reduce_prod(tf.shape(z)), tf.complex64)
        Nc = tf.cast(tf.reduce_prod(tf.shape(c)), tf.complex64)
        grad_z = 2*grad_s*(z-c)/Nz
        grad_c = 2*grad_s*(c-z)/Nc

        return grad_z, grad_c

    return s, grad



#-------------------------------------------------------------------
@tf.custom_gradient
def vector_conj_inner(z, c):
    
    s = tf.reduce_sum(z*tf.math.conj(c), -1)

    def grad(grad_s):

        grad_s = tf.expand_dims(grad_s, axis=-1)
        #grad_s = Debug('grad_s', grad_s)
        grad_z = grad_s*c
        grad_c = tf.math.conj(grad_s)*z

        return grad_z, grad_c

    return s, grad



#-------------------------------------------------------------------
@tf.custom_gradient
def cast_to_complex(z_real, z_imag):

    s = tf.complex(z_real, z_imag)

    def grad(grad_s):

        return tf.math.real(grad_s), tf.math.imag(grad_s)

    return s, grad



#-------------------------------------------------------------------
@tf.custom_gradient
def cast_to_float(z):

    s = tf.stack([tf.math.real(z), tf.math.imag(z)], axis=-1)

    def grad(grad_s):

        return tf.complex(grad_s[...,0], grad_s[...,1])

    return s, grad



#-------------------------------------------------------------------
@tf.custom_gradient
def elementwise_real(z):

    s = tf.math.real(z)

    def grad(grad_s):

        return tf.complex(grad_s, tf.zeros_like(grad_s))

    return s, grad



#-------------------------------------------------------------------
@tf.custom_gradient
def elementwise_complex(z_real):

    s = tf.cast(z_real, tf.complex64)

    def grad(grad_s):

        return tf.math.real(grad_s)

    return s, grad



#-------------------------------------------------------------------
@tf.custom_gradient
def elementwise_abs(z):

    s = tf.abs(z)

    def grad(grad_s):

        grad_s = tf.cast(tf.math.real(grad_s), tf.complex64)
        az = tf.cast(tf.abs(z)+1e-6, tf.complex64)
        gs = tf.cast(tf.math.real(grad_s), tf.complex64)
        grad_z = gs*z/az

        return grad_z

    return s, grad



#-------------------------------------------------------------------
@tf.custom_gradient
def elementwise_abs2(z):

    s = tf.abs(z)**2

    def grad(grad_s):

        grad_s = tf.cast(tf.math.real(grad_s), tf.complex64)
        grad_z = 2*grad_s*z

        return grad_z

    return s, grad



#-------------------------------------------------------------------
@tf.custom_gradient
def vector_normalize_magnitude(z):

    #num = tf.linalg.norm(z, axis=-1, keepdims=True) + 1e-6                        # norm over last axis
    num = tf.sqrt(tf.reduce_sum(tf.abs(z)**2, axis=-1, keepdims=True)) + 1e-6      # norm over last axis
    s = z/tf.cast(num, tf.complex64)

    def grad(grad_s):
      
        tmp = tf.reduce_sum(tf.math.real(z)*tf.math.real(grad_s) + tf.math.imag(z)*tf.math.imag(grad_s), axis=-1, keepdims=True)
        tmp /= num*num*num + 1e-6
        grad_z = grad_s/tf.cast(num, tf.complex64) - z*tf.cast(tmp, tf.complex64)

        return grad_z

    return s, grad



#-------------------------------------------------------------------
@tf.custom_gradient
def vector_normalize_phase(z):

    num = tf.abs(z[...,0]) + 1e-6
    num = tf.cast(num, tf.complex64)
    phi = z[...,0] / num
    s = z*tf.math.conj(phi)[...,tf.newaxis]


    def grad(grad_s):

        grad_z_1 = tf.cast(tf.math.real(grad_s[...,0]), tf.complex64)*phi

        tmp = tf.einsum('...i,...i->...', safe_conj(grad_s)[...,1:], z[...,1:])
        tmp -= tf.math.conj(tmp)*phi*phi
        grad_z_1 += 0.5*tmp / num

        grad_z_2 = grad_s*phi[...,tf.newaxis]
        grad_z = tf.concat([grad_z_1[...,tf.newaxis], grad_z_2[...,1:]], axis=-1)

        return grad_z

    return s, grad



#-------------------------------------------------------------------
@tf.custom_gradient
def elementwise_normalize(z):

    num = tf.abs(z) + 1e-6
    num = tf.cast(num, tf.complex64)
    s = z/num

    def grad(grad_s):

        # gradient from division
        grad_z1 = grad_s / tf.math.conj(num)
        grad_num = -grad_s*tf.math.conj(z) / (tf.math.conj(num)**2)

        # gradient of abs
        az = tf.cast(tf.abs(z)+1e-6, tf.complex64)
        grad_num = tf.cast(tf.math.real(grad_num), tf.complex64)
        grad_z2 = grad_num*z/az

        return grad_z1+grad_z2

    return s, grad


#-------------------------------------------------------------------
@tf.custom_gradient
def elementwise_mul(z, c):
    
    s = z*c

    def grad(grad_s):

        grad_z = grad_s*tf.math.conj(c)
        grad_c = grad_s*tf.math.conj(z)

        return grad_z, grad_c

    return s, grad



#-------------------------------------------------------------------
@tf.custom_gradient
def elementwise_div(z, c):
    
    s = z/c

    def grad(grad_s):

        grad_z = grad_s / tf.math.conj(c)
        grad_c = -grad_s*tf.math.conj(z) / (tf.math.conj(c)**2)

        return grad_z, grad_c

    return s, grad



#-------------------------------------------------------------------
@tf.custom_gradient
def elementwise_conj(z):

    s = safe_conj(z)

    def grad(grad_s):

        grad_z = safe_conj(grad_s)

        return grad_z

    return s, grad



#-------------------------------------------------------------------
@tf.custom_gradient
def elementwise_tanh(z):

    num = tf.abs(z) + 1e-6
    ta = tf.tanh(num)
    mag = ta/num
    s = z*tf.cast(mag, tf.complex64)

    def grad(grad_s):

        sa = 0.5-0.5*ta*ta

        tmp1 = tf.cast(sa + 0.5*mag, tf.complex64)
        tmp2 = tf.cast(sa - 0.5*mag, tf.complex64)
        tmp3 = z*z/tf.cast(num*num, tf.complex64)                # = (z/abs(z))**2
        grad_z = tf.math.conj(grad_s)*tmp2*tmp3 + grad_s*tmp1

        return grad_z

    return s, grad



#-------------------------------------------------------------------
@tf.custom_gradient
def elementwise_sigmoid(z):

    s_real = tf.tanh(tf.math.real(z)*0.5)*0.5+0.5
    s = tf.cast(s_real, tf.complex64)

    def grad(grad_s):

        grad_z_real = tf.math.real(grad_s)*s_real*(1-s_real)
        grad_z = tf.cast(grad_z_real, tf.complex64)

        return grad_z

    return s, grad



#-------------------------------------------------------------------
# einsum with complex arguments causes adjoint errors in CUDNN, workaround:
def einsum(subscripts, z, c):

    s1 = tf.einsum(subscripts, tf.math.real(z), tf.math.real(c))
    s2 = tf.einsum(subscripts, tf.math.real(z), tf.math.imag(c))
    s3 = tf.einsum(subscripts, tf.math.imag(z), tf.math.real(c))
    s4 = tf.einsum(subscripts, tf.math.imag(z), tf.math.imag(c))

    s = cast_to_complex(s1-s4, s2+s3)

    return s


