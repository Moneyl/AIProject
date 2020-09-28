#pragma once
#include "Neuron.h"
#include <cstdio>
#include <vector>

//Neural net layer. Contains neurons
class Layer
{
public:
    //Create a layer and fill it with neurons
    Layer(int numNeurons, int numNeuronWeights)
    {
        for (int i = 0; i < numNeurons; i++)
        {
            //Neuron constructor will set random weight and bias values
            Neurons.push_back(Neuron(numNeuronWeights));
        }
    }

    //Print values of all neurons in the layer
    void PrintValues()
    {
        for (auto& neuron : Neurons)
        {
            printf("    Bias: %f\n", neuron.Bias);
            printf("    LastOutput: %f\n", neuron.LastOutput);
            printf("    LastError: %f\n", neuron.LastError);
            printf("    Weights: {");
            for (int i = 0; i < neuron.Weights.size(); i++)
            {
                if (i != 0)
                    printf(", ");

                printf("%f", neuron.Weights[i]);
            }
            printf("}\n\n");
        }
    }

    //The neurons that make up this layer
    std::vector<Neuron> Neurons = {};
};