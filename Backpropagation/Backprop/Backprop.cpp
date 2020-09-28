#include <iostream>
#include <random>
#include <time.h>
#include <vector>
#include "NeuralNet.h"

int main()
{
    //Seed random number generator. Used for initial neuron weights
    srand(time(NULL));

    //Create a neural net
    NeuralNet net;

    int numInputs = 2;
    int numHiddenNeurons = 1;
    int numOutputNeurons = 2;
    int numTrainingRuns = 2000;

    //Add layers to neural net
    net.Layers.push_back(Layer(numHiddenNeurons, numInputs));
    net.Layers.push_back(Layer(numOutputNeurons, numHiddenNeurons));
    
    //Todo: Test network with larger data set
    //Todo: Find real data to test the on the network. This is just mock data with no meaning
    std::vector<float> inputData = { 0.5f, 0.0f };
    std::vector<float> expectedOutputData = { 0.0f, 0.5f };

    //Output initial neuron values
    printf("Created a neural network with %d inputs, %d hidden neurons, and %d output neurons.\n", numInputs, numHiddenNeurons, numOutputNeurons);
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
