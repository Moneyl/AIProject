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

    //Add layers to neural net
    net.Layers.push_back(Layer(numHiddenNeurons, numInputs));
    net.Layers.push_back(Layer(numOutputNeurons, numHiddenNeurons));

    //Output initial neuron values
    printf("Created a neural network with %d inputs, %d hidden neurons, and %d output neurons.\n", numInputs, numHiddenNeurons, numOutputNeurons);
    Layer& hiddenLayer = net.Layers[0];
    Layer& outputLayer = net.Layers[1];
    printf("Hidden layer neurons:\n");
    hiddenLayer.PrintValues();
    printf("Output layer neurons:\n");
    outputLayer.PrintValues();

    printf("\nTesting forward propagation...");
    std::vector<float> testInput1 = { 1.0f, 0.0f };
    std::vector<float> forwardPropResult = net.ForwardPropagate(testInput1);
    printf(" Done.\n");
    printf("Forward propagation result: {");
    for (int i = 0; i < forwardPropResult.size(); i++)
    {
        if (i != 0)
            printf(", ");

        printf("%f", forwardPropResult[i]);
    }
    printf("}\n\n");


    printf("Neural net values after forward propagation: \n");
    printf("Hidden layer neurons:\n");
    hiddenLayer.PrintValues();
    printf("Output layer neurons:\n");
    outputLayer.PrintValues();

    auto a = 2;
}
