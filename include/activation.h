#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "nmodule.h"
#include <vector>


// class Activation
// {
//     public:
//         Activation() = default;
//         virtual ~Activation() = default;
//         virtual std::vector<float> compute_activation(const std::vector<float> &data) = 0;
// };

// class Softmax : public Activation
// {
//     public:
//         Softmax() = default;
//         float sigmoid(float data_point);
//         std::vector<float> compute_activation(const std::vector<float> &data) override;
        
// };

// class ReLU : public Activation
// {
//     public:
//         ReLU() = default;
//         std::vector<float> compute_activation(const std::vector<float>& data) override;

// };


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


