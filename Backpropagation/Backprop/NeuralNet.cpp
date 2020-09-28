#include "NeuralNet.h"

std::vector<float> NeuralNet::ForwardPropagate(const std::vector<float>& inputBase)
{
    //Feed first layer inputs (equivalent to an input layer)
    std::vector<float> inputs = inputBase;

    //Loop through each layer, and each neuron, using the activation and transfer functions as input for the next layer
    for (auto& layer : Layers)
    {
        std::vector<float> outputs = {};
        for (auto& neuron : layer.Neurons)
        {
            float activationValue = neuron.Activate(inputs);
            outputs.push_back(neuron.Transfer(activationValue));
        }

        //The input of the next layer are the outputs of the current one
        inputs = outputs;
    }

    //Finally, return the output of the final layer
    return inputs;
}
