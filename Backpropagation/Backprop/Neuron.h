#pragma once
#include <vector>

class Neuron
{
public:
    Neuron(int numWeights);
    float Activate(const std::vector<float>& inputs);
    float Transfer(float activationValue);
    float TransferDelta(float output);

    float Bias = 0.0f;
    std::vector<float> Weights = {};
    float LastOutput = 0.0f;
    float LastError = 0.0f;
};