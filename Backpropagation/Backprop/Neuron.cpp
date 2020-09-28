#include "Neuron.h"

float GetRandomNumNormalized();

Neuron::Neuron(int numWeights)
{
    //Create weights and initialize with random values between 0.0 and 1.0
    for (int i = 0; i < numWeights; i++)
    {
        Weights.push_back(GetRandomNumNormalized());
    }

    //Initialize bias with random value
    Bias = GetRandomNumNormalized();
}

float Neuron::CalculateOutput(const std::vector<float>& inputs)
{
    float activationValue = Bias;
    for (int i = 0; i < Weights.size(); i++)
        activationValue += Weights[i] * inputs[i];

    return activationValue;
}

float Neuron::Sigmoid(float activationValue)
{
    //Todo: Look at alternative transfer functions. Remember to update SigmoidDerivative as well
    //Using sigmoid transfer function
    return 1.0f / (1.0f + exp(-activationValue));
}

float Neuron::SigmoidDerivative(float output)
{
    //Todo: Remember to update this whenever changing the transfer function
    //Derivative of the sigmoid function. Calculating slope on sigmoid at x = output
    return output * (1.0f - output);
}

//Returns a random float value between 0.0 and 1.0 using rand(). Should be seeded with srand by user before running
float GetRandomNumNormalized()
{
    //Get random number between 0 and 255. Then divide by 255 to normalize it.
    float randomNum = (float)(rand() % 255);
    randomNum /= 255.0f;
    return randomNum;
}