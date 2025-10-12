#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <vector>


class Activation
{
    public:
        Activation() = default;
        virtual ~Activation() = default;
        virtual std::vector<float> compute_activation(const std::vector<float> &data) = 0;
};

// class Sigmoid : public Activation
// {
//     public:
//         Sigmoid() = default;
//         std::vector<float> compute_activation(const std::vector<float> &data) override;
// };

class Softmax : public Activation
{
    public:
        Softmax() = default;
        float sigmoid(float data_point);
        std::vector<float> compute_activation(const std::vector<float> &data) override;
        
};

class ReLU : public Activation
{
    public:
        ReLU() = default;
        std::vector<float> compute_activation(const std::vector<float>& data) override;

};
#endif