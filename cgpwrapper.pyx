cimport cython
from libc.stdlib cimport malloc, free

from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

cimport cgpwrapper

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
    cdef int numInputs
    cdef int numOutputs
    cdef bytes funset
    cdef int numNodes
    cdef int nodeArity
    cdef int numGens
    cdef int updateFrequency
    cdef double targetFitness

    def __cinit__(self, **kwarg):
        self.funset = kwarg.get('funset', 'add,sub,mul,div,sin').encode()
        self.numNodes = kwarg.get('n_nodes', 15)
        self.nodeArity = kwarg.get('n_arity', 2)
        self.numGens = kwarg.get('n_gens', 10000)
        self.updateFrequency = kwarg.get('update_freq', 500)
        self.targetFitness = kwarg.get('target_fitness', 0.1)


    def _get_instance_length(self, data):
        data_type = type(data[0])
        if data_type is list or data_type is np.ndarray:
            return len(data[0])
        else:
            return 1 

    def fit(self, X, y):
        cdef parameters *params
        cdef dataSet *traningData
        cdef chromosome *chromo

        self.numInputs = self._get_instance_length(X)
        self.numOutputs = self._get_instance_length(y)

        cdef double *cdata = <double *>malloc(len(X) * self.numInputs * cython.sizeof(double))
        cdef double *ctarget = <double *>malloc(len(X) * self.numOutputs * cython.sizeof(double))
        cdef int numSamples = len(X)

        cdef int dataIndex = 0
        cdef int targetIndex = 0

        for i in range(0, len(X)):
            
            for j in range(0, self.numInputs):
                if isinstance(X[i], (list, np.ndarray)):
                    cdata[dataIndex] = X[i][j]
                else:
                    cdata[dataIndex] = X[i]
                dataIndex += 1

            for j in range(0, self.numOutputs):
                if isinstance(y[i], (list, np.ndarray)):
                    ctarget[targetIndex] = y[i][j] 
                else:
                    ctarget[targetIndex] = y[i]
                targetIndex += 1


        params = initialiseParameters(self.numInputs, self.numNodes, self.numOutputs, self.nodeArity)

        addNodeFunction(params, self.funset)

        setTargetFitness(params, self.targetFitness)

        setUpdateFrequency(params, self.updateFrequency)
        
        trainingData = initialiseDataSetFromArrays(self.numInputs, self.numOutputs, numSamples, cdata, ctarget)

        self.chromo = runCGP(params, trainingData, self.numGens)

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
            for i in range(0, self.numOutputs):
                output = getChromosomeOutput(self.chromo, i)
                result.append(np.floor(output))
        return result


