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

    std::vector<Neuron> Neurons = {};
};

class NeuralNet
{
public:
    //Forward propagate input values through the network
    std::vector<float> ForwardPropagate(const std::vector<float>& inputBase);
    //Backpropagate output error through network by comparing expected outputs to generated outputs
    void Backpropagate(const std::vector<float>& expectedOutputs);
    //Update the weights of neurons
    void UpdateNeuronWeights(const std::vector<float>& inputBase);
    //Train the network by repeated forward propogating an input, 
    //then using backward propagation to update weights and biases by comparing output to expected output
    void Train(std::vector<std::vector<float>> inputs, std::vector<std::vector<float>> expectedOutputs, int numTrainingRuns);

    float LearningRate = 0.4f;
    std::vector<Layer> Layers = {};
};