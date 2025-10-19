#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include<vector>
#include<memory>
#include "nmodule.h"

class NeuralNetwork
{
public:
    NeuralNetwork() = default;
    std::vector<float> forward(std::vector<float>& input);
    std::vector<std::unique_ptr<TensorElement>> Network;

};


#endif