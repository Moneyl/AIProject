#include "NeuralNet.h"
#include <algorithm>
#include <fstream>

std::vector<float> NeuralNet::ForwardPropagate(const std::vector<float>& inputBase)
{
    //Feed first layer inputs (equivalent to an input layer)
    std::vector<float> inputs = inputBase;

    //Loop through each layer, and each neuron, using output of each neuron as inputs for the next layer
    for (auto& layer : Layers)
    {
        std::vector<float> outputs = {};
        for (auto& neuron : layer.Neurons)
        {
            //Calculate neuron output by summing the multiples of weights and inputs (Average of inputs)
            //Then pass that through the sigmoid function to normalize it [0.0 to 1.0]
            float output = neuron.CalculateOutput(inputs);
            outputs.push_back(neuron.Sigmoid(output));
            neuron.LastOutput = outputs.back(); //Store the output in the neuron for backprop
        }

        //The outputs of the current layer are inputs of the next one
        inputs = outputs;
    }

    //Finally, return the outputs of the final layer
    return inputs;
}

void NeuralNet::Backpropagate(const std::vector<float>& expectedOutputs)
{
    //Work backwards from output layer
    for (int i = Layers.size() - 1; i >= 0; i--)
    {
        Layer& layer = Layers[i];
        std::vector<float> outputErrors = {};

        //Loop through all neurons in layer
        for (int j = 0; j < layer.Neurons.size(); j++)
        {
            if (i == Layers.size() - 1) //Output layer
            {
                //Error for neurons in output layer is the difference between the output and expected output
                Neuron& neuron = layer.Neurons[j];
                outputErrors.push_back(expectedOutputs[j] - neuron.LastOutput);
            }
            else //Hidden layer
            {
                float error = 0.0f;
                Layer& nextLayer = Layers[i + 1];

                //For hidden layers sum the multiples of the weight and error delta for the next layers neurons
                for (auto& neuron : nextLayer.Neurons)
                    error += neuron.Weights[j] * neuron.ErrorDelta;

                outputErrors.push_back(error);
            }
        }

        //Update error delta of neurons in layer
        for (int j = 0; j < layer.Neurons.size(); j++)
        {
            Neuron& neuron = layer.Neurons[j];
            neuron.ErrorDelta = outputErrors[j] * neuron.SigmoidDerivative(neuron.LastOutput);
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

        //If not a hidden layer use outputs of previous layer as input
        if (i != 0) //Not a hidden layer
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
                neuron.Weights[j] += LearningRate * neuron.ErrorDelta * inputs[j];
            }
            //Adjust bias towards expected output
            neuron.Bias += LearningRate * neuron.ErrorDelta;
        }
    }
}

void NeuralNet::Train(std::vector<std::vector<float>> inputs, std::vector<std::vector<float>> expectedOutputs, int numTrainingRuns)
{
    //Open output stream to file to write outputs
    std::ofstream out;
    //Removes existing contents of file on each run
    out.open("./NeuralNetOut.csv", std::ios::out | std::ios::trunc);
    out << "Run,MSE,RMSE\n";

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

            //Calculate the root mean square error. Equal to the standard deviation of the errors
            float error0 = powf(expectedOutputs[i][0] - outputs[0], 2.0f);
            float error1 = powf(expectedOutputs[i][1] - outputs[1], 2.0f);
            float mse = (error0 + error1) / 2.0f;
            float rmse = sqrtf(mse);
            //Output mse and rmse to csv file
            out << run << "," << mse << "," << rmse << "\n";

            //Todo: Determine if this should be done at the end of each run instead of per input
            //Next, back propagate the network to get the error for each neuron and update their weights
            Backpropagate(expectedOutput);
            UpdateNeuronWeights(input);
        }
    }
    
    out.close();
}
