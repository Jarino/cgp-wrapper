#include <stdio.h>
#include <cgp.h>

int main(void) {
  struct parameters *params = NULL;
  struct dataSet *trainingData = NULL;
  struct chromosome *chromo = NULL;

  int numInputs = 1;
  int numNodes  = 15;
  int numOutputs = 1;
  int nodeArity  = 2;

  int numGens = 10000;
  int updateFrequency = 500;
  double targetFitness = 0.1;

  params = initialiseParameters(numInputs, numNodes, numOutputs, nodeArity);

  addNodeFunction(params, "add,sub,mul,div,sin");

  setTargetFitness(params, targetFitness);

  setUpdateFrequency(params, updateFrequency);

  printParameters(params);

  trainingData = initialiseDataSetFromFile("./symbolic.data");

  chromo = runCGP(params, trainingData, numGens);

  printChromosome(chromo, 0);
  saveChromosomeDot(chromo, 0, "chromo.dot");

  freeDataSet(trainingData);
  freeChromosome(chromo);
  freeParameters(params);

  return 0;
}
