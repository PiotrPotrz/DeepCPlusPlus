#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "nmodule.h"
#include <vector>


class Softmax : public TensorElement
{
public:
    Softmax() = default;
    std::vector<float> forward(std::vector<float>& x) override;
    void backward(std::vector<float>& prev_grad) override;
    std::string name() const override { return "Softmax"; }
private:
    std::vector<float> outputs;
};

class ReLU : public TensorElement
{
public:
    ReLU() = default;
    std::vector<float> forward(std::vector<float>& x) override;
    void backward(std::vector<float>& prev_grad) override;
    std::string name() const override { return "ReLU"; }
};

#endif


