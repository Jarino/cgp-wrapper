cdef extern from "cgp.h":
    struct parameters:
        pass

    struct dataSet:
        pass

    struct chromosome:
        pass

    parameters *initialiseParameters(int numInputs, int numNodes, int numOutputs, int arity)
    void addNodeFunction(parameters *params, char *functionNames)
    void setTargetFitness(parameters *params, double targetFitness)
    void setUpdateFrequency(parameters *params, int updateFrequency)
    void printParameters(parameters *params)
    void freeParameters(parameters *params)
    dataSet *initialiseDataSetFromFile(char *file)
    void freeDataSet(dataSet *data)
    chromosome* runCGP(parameters *params, dataSet *data, int numGens) 
    
    void printChromosome(chromosome *chromo, int weights)
    void freeChromosome(chromosome *chromo)
    void saveChromosomeDot(chromosome *chromo, int weights, char *fileName)
    dataSet *initialiseDataSetFromArrays(int numInputs, int numOutputs, int numSamples, double *inputs, double *outputs)


