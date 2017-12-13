cimport pycgp
from libc.stdlib cimport malloc, free
from collections import defaultdict
import warnings
cimport cython
from cpython cimport array
from cpython.ref cimport PyObject
import array
import numpy as np
import math

from sklearn.base import BaseEstimator, ClassifierMixin

class BESingleton(BaseEstimator):
    __instance = None
    def __new__(cls):
        if BESingleton.__instance is None:
            BESingleton.__instance = BaseEstimator.__new__(cls)
        return BESingleton.__instance


class CMSingleton(ClassifierMixin):
    __instance = None
    def __new__(cls):
        if CMSingleton.__instance is None:
            CMSingleton.__instance = ClassifierMixin.__new__(cls)
        return CMSingleton.__instance

cdef class CGPClassifier:

    cdef chromosome *chromo

    def fit(self, X, y):
        cdef parameters *params
        cdef dataSet *traningData
        cdef chromosome *chromo

        cdef int numInputs = 4;
        cdef int numNodes = 15;
        cdef int numOutputs = 1;
        cdef int nodeArity = 2;

        cdef int numGens = 10000;
        cdef int updateFrequency = 500;
        cdef double targetFitness = 0.1;

        cdef double *cdata = <double *>malloc(len(X) * numInputs * cython.sizeof(double))
        cdef double *ctarget = <double *>malloc(len(X) * cython.sizeof(double))
        cdef int numSamples = len(X)

        cdef int dataIndex = 0
        for i in range(0, len(X)):
            
            for j in range(0, numInputs):
                cdata[dataIndex] = X[i][j]
                dataIndex += 1

            ctarget[i] = y[i]

        params = pycgp.initialiseParameters(numInputs, numNodes, numOutputs, nodeArity)

        addNodeFunction(params, "add,sub,mul,div,sin")

        setTargetFitness(params, targetFitness)

        setUpdateFrequency(params, updateFrequency)
        
        trainingData = initialiseDataSetFromArrays(numInputs, numOutputs, numSamples, cdata, ctarget)

        self.chromo = runCGP(params, trainingData, numGens)

        free(cdata)
        free(ctarget)
        freeDataSet(trainingData)
        freeParameters(params)


    def get_params(self, deep=True):
        return BESingleton().get_params(deep)

    def set_params(self, **params):
        return BESingleton().set_params(**params)
    
    def score(self, X, y, sample_weight=None):
        return CMSingleton().score(X, y, sample_weight)

    def predict(self, X):

        # check whether we have only 1D vector or multiple vectors
        x_type = type(X[0])

        if x_type is not list and x_type is not np.ndarray:
            X = [X]

        
        cdef double *inputs = <double *> malloc(len(X[0]) * cython.sizeof(double))
        cdef chromosome *chromo

        result = [] 
        for vector in X:
            for i, val in enumerate(vector):
                inputs[i] = val
                
            executeChromosome(self.chromo, inputs)
            output = getChromosomeOutput(self.chromo, 0)
            result.append(math.floor(output))
        return result


