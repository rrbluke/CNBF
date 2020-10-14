# -*- coding: utf-8 -*-
__author__ = "Lukas Pfeifenberger"



import h5py
import hdf5storage
import scipy.io
import numpy as np
import os



#---------------------------------------------------------
# loads a list of numpy arrays identified by <varnames>
# from the m-file <matfile>

def load_numpy_from_mat(matfile, varnames=None, hdf5=False):

    matdata = load_dict_from_mat(matfile=matfile, hdf5=hdf5)
    if matdata is None:
        return None

    #return all variables from the mat-file
    if varnames is None:
        return matdata

    #make list with one element
    if type(varnames) is not list:
        varnames = [varnames]

    data = {}
    for varname in varnames:

        #check if matfile contains the requested variable
        if varname in matdata:

            #load variable from matdata
            x = matdata[varname]

            #copy variable to rearrange strides
            y = np.copy(x)

            data[varname] = y

    return data


#---------------------------------------------------------
# saves a dict of numpy arrays <data>
# to the m-file <matfile>

def save_numpy_to_mat(matfile, data, hdf5=False, overwrite=False):

    return save_dict_to_mat(matfile=matfile, matdata=data, hdf5=hdf5, overwrite=overwrite)


#---------------------------------------------------------
# saves a dictionary to a matfile
# existing contents are preserved

def save_dict_to_mat(matfile, matdata, hdf5=False, overwrite=False):

    if matfile is None:
        return False

    #create folder if it doesn't exist
    path = os.path.dirname(os.path.abspath(matfile))
    if not os.path.exists(path):
        os.makedirs(path)

    if hdf5 is False:
        if overwrite is False:
            #to append to the file, its contents have to be read, updated and written
            existing_data = load_dict_from_mat(matfile, hdf5=hdf5)
        else:
            existing_data = None
            
        if existing_data is not None:
            existing_data.update(matdata)
            scipy.io.savemat(matfile, existing_data)
        else:
            scipy.io.savemat(matfile, matdata)
    else:
        #appends to the file by default
        hdf5storage.savemat(matfile, matdata, compress=False)

    return True


#---------------------------------------------------------
# loads a dictionary from a matfile
# the dictionary will contain ndarray objects!
# for details, see: https://docs.scipy.org/doc/scipy-0.18.1/reference/tutorial/io.html

def load_dict_from_mat(matfile, hdf5=False):

    if matfile is None:
        return None

    if os.path.isfile(matfile):
        if hdf5 is False:
            try:
                matdata = scipy.io.loadmat(matfile)
            except:
                #avoid false positive corruption check, see: 
                #https://github.com/scipy/scipy/issues/6999
                try:
                    matdata = scipy.io.loadmat(matfile, verify_compressed_data_integrity=False)                
                except:
                    return None

        else:
            matdata = hdf5storage.loadmat(matfile)

        return matdata

    else:
        return None


