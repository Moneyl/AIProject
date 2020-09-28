#include <iostream>
#include <random>
#include <time.h>
#include <vector>
#include "NeuralNet.h"

int main()
{
    //Seed random number generator. Used for initial neuron weights
    srand(time(NULL));

    //Create an empty neural net
    NeuralNet net;

    //Neural net config
    int numInputs = 2; //Note: Must match number of inputs in input data entries
    int numOutputs = 2; //Note: Must match number of outputs in output data entries
    int numHiddenNeurons = 1;
    int numTrainingRuns = 20000;

    //Add layers to neural net
    net.Layers.push_back(Layer(numHiddenNeurons, numInputs)); //Hidden layer
    net.Layers.push_back(Layer(numOutputs, numHiddenNeurons)); //Output layer
    
    //Todo: Test network with larger data set
    //Todo: Find real data to test the on the network. This is just mock data with no meaning
    std::vector<float> inputData = { 0.5f, 0.0f };
    std::vector<float> expectedOutputData = { 0.0f, 0.5f };

    //Output initial neuron values
    printf("Created a neural network with %d inputs, %d hidden neurons, and %d output neurons.\n", numInputs, numHiddenNeurons, numOutputs);
    Layer& hiddenLayer = net.Layers[0];
    Layer& outputLayer = net.Layers[1];
    printf("Hidden layer neurons:\n");
    hiddenLayer.PrintValues();
    printf("Output layer neurons:\n");
    outputLayer.PrintValues();

    //Train network
    printf("\nTraining neural network with %d runs...", numTrainingRuns);
    net.Train({ inputData }, { expectedOutputData }, numTrainingRuns);
    printf(" Done.\n");

    //Output neuron values post training
    printf("Neural net values after training: \n");
    printf("Hidden layer neurons:\n");
    hiddenLayer.PrintValues();
    printf("Output layer neurons:\n");
    outputLayer.PrintValues();
}
