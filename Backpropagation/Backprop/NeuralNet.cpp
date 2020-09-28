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
            //Calculate neuron output by summing the multiples of weights and inputs (WeightN * InputN)
            //Then pass that through the sigmoid function to normalize it [0.0 to 1.0]
            float activationValue = neuron.CalculateOutput(inputs);
            outputs.push_back(neuron.Sigmoid(activationValue));
            neuron.LastOutput = outputs.back();
        }

        //The input of the next layer are the outputs of the current one
        inputs = outputs;
    }

    //Finally, return the output of the final layer
    return inputs;
}

void NeuralNet::Backpropagate(const std::vector<float>& expectedOutputs)
{
    //Work backwards from output layer
    for (int i = Layers.size() - 1; i >= 0; i--)
    {
        Layer& layer = Layers[i];
        std::vector<float> outputErrors = {};

        //Special case for hidden layer
        if (i == 0)
        {
            //Calculate error for each neuron in the hidden layer
            for (int j = 0; j < layer.Neurons.size(); j++)
            {
                float error = 0.0f;
                Layer& outputLayer = Layers[i + 1];

                //For the hidden layer add the errors of all output neurons weights that correspond to the hidden layer neuron 
                for (auto& neuron : outputLayer.Neurons)
                    error += neuron.Weights[j] * neuron.LastError;

                outputErrors.push_back(error);
            }
        }
        else //Behavior for all other layers
        {
            for (int j = 0; j < layer.Neurons.size(); j++)
            {
                //Error here is difference between the output and expected output
                Neuron& neuron = layer.Neurons[j];
                outputErrors.push_back(expectedOutputs[j] - neuron.LastOutput);
            }
        }

        //Update error value of neurons in layer
        for (int j = 0; j < layer.Neurons.size(); j++)
        {
            Neuron& neuron = layer.Neurons[j];
            neuron.LastError = outputErrors[j] * neuron.SigmoidDerivative(neuron.LastOutput);
        }
    }
}

void NeuralNet::UpdateNeuronWeights(const std::vector<float>& inputBase)
{
    //Use provided input for hidden layer input
    std::vector<float> inputs = inputBase;

    for (int i = 0; i < Layers.size(); i++)
    {
        Layer& currentLayer = Layers[i];

        //If not hidden layer use outputs of previous layer as input
        if (i != 0) //Not the hidden layer
        {
            //Clear inputs and set to outputs of previous layer
            inputs.clear();
            Layer& lastLayer = Layers[i - 1];
            for (auto& neuron : lastLayer.Neurons)
                inputs.push_back(neuron.LastOutput);
        }

        //Update weights and bias for each neuron
        for (auto& neuron : currentLayer.Neurons)
        {
            for (int j = 0; j < inputs.size(); j++)
            {
                neuron.Weights[j] += LearningRate * neuron.LastError * inputs[j];
            }
            //Adjust bias towards expected output
            neuron.Bias += LearningRate * neuron.LastError;
        }
    }
}

void NeuralNet::Train(std::vector<std::vector<float>> inputs, std::vector<std::vector<float>> expectedOutputs, int numTrainingRuns)
{
    //Train the network numTrainingRuns times
    for (int run = 0; run < numTrainingRuns; run++)
    {
        //For each run forward and back propagate the network on all inputs
        for (int i = 0; i < inputs.size(); i++)
        {
            //Get selected inputs and outputs
            std::vector<float>& input = inputs[i];
            std::vector<float>& expectedOutput = expectedOutputs[i];

            //First forward propagate the input
            std::vector<float> outputs = ForwardPropagate(input);
            //Next, back propagate the network to get the error for each neuron and update their weights
            Backpropagate(expectedOutput);
            UpdateNeuronWeights(input);
        }
    }
}
