#pragma once
#include "Neuron.h"
#include "Layer.h"
#include <vector>

class NeuralNet
{
public:
    //Forward propagate input values through the network. Returns network output.
    std::vector<float> ForwardPropagate(const std::vector<float>& inputBase);
    //Backpropagate output error through network by comparing expected outputs to calculated outputs
    void Backpropagate(const std::vector<float>& expectedOutputs);
    //Update the weights and biases of neurons
    void UpdateNeuronWeights(const std::vector<float>& inputBase);
    //Train the network by repeatedly forward propogating an input, 
    //then using backward propagation to update weights and biases by comparing output to expected output
    void Train(std::vector<std::vector<float>> inputs, std::vector<std::vector<float>> expectedOutputs, int numTrainingRuns);

    float LearningRate = 0.4f;
    std::vector<Layer> Layers = {};
};