#pragma once
#include <vector>

class Neuron
{
public:
    Neuron(int numWeights);
    //Calculate output by summing the multiples of weights
    float CalculateOutput(const std::vector<float>& inputs);
    //Used to normalize the output during forward propagation
    float Sigmoid(float value);
    //Used during backpropagation
    float SigmoidDerivative(float output);

    //Used to push the neuron towards the desired output during training
    float Bias = 0.0f;
    //The influence strength of each input on the next neuron
    std::vector<float> Weights = {};
    float LastOutput = 0.0f;
    float ErrorDelta = 0.0f;
};