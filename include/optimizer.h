#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <memory>
#include <vector>
#include "nmodule.h"
#include "neural_network.h"

class SGD
{
public:
    SGD(std::vector<std::unique_ptr<TensorElement>>& Network, float learning_rate);
    void step(std::vector<float>& loss_grad);
    std::vector<std::unique_ptr<TensorElement>>& Network; 
    private:
    float learning_rate;
    float modify_weight(float weight, float gradient);
};

#endif