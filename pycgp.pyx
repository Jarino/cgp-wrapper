cimport pycgp
from libc.stdlib cimport malloc, free
cimport cython

def run(data, target):
    cdef parameters *params
    cdef dataSet *traningData
    cdef chromosome *chromo

    cdef int numInputs = 1;
    cdef int numNodes = 15;
    cdef int numOutputs = 1;
    cdef int nodeArity = 2;

    cdef int numGens = 10000;
    cdef int updateFrequency = 500;
    cdef double targetFitness = 0.1;

    cdef double *cdata = <double *>malloc(len(data) * cython.sizeof(double))
    cdef double *ctarget = <double *>malloc(len(data) * cython.sizeof(double))
    cdef int numSamples = len(data)

    for i in  range(0, len(data)):
        cdata[i] = data[i]
        ctarget[i] = target[i]

    params = pycgp.initialiseParameters(numInputs, numNodes, numOutputs, nodeArity)

    addNodeFunction(params, "add,sub,mul,div,sin")

    setTargetFitness(params, targetFitness)

    setUpdateFrequency(params, updateFrequency)
    
    trainingData = initialiseDataSetFromArrays(numInputs, numOutputs, numSamples, cdata, ctarget)

    chromo = runCGP(params, trainingData, numGens)

    saveChromosomeDot(chromo, 0, "chromo.dot")

    freeDataSet(trainingData)
    freeChromosome(chromo)
    freeParameters(params)


