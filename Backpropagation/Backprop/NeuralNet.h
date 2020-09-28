#pragma once
#include "Neuron.h"
#include <vector>

class Layer
{
public:
    Layer(int numNeurons, int numNeuronWeights)
    {
        for (int i = 0; i < numNeurons; i++)
        {
            Neurons.push_back(Neuron(numNeuronWeights));
        }
    }
    void PrintValues()
    {
        for (auto& neuron : Neurons)
        {
            printf("    Bias: %f\n", neuron.Bias);
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

    std::vector<Neuron> Neurons = {};
};

class NeuralNet
{
public:
    std::vector<float> ForwardPropagate(const std::vector<float>& inputBase);

    std::vector<Layer> Layers = {};
};