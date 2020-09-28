#pragma once
#include <vector>

class Neuron
{
public:
    Neuron(int numWeights);
    float CalculateOutput(const std::vector<float>& inputs);
    float Sigmoid(float activationValue);
    float SigmoidDerivative(float output);

    float Bias = 0.0f;
    std::vector<float> Weights = {};
    float LastOutput = 0.0f;
    float LastError = 0.0f;
};