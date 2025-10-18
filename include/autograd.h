#ifndef AUTOGRAD_H
#define AUTOGRAD_H

#include <vector>
#include <string>

struct Params
{
    std::vector<float> weigths;
    std::vector<float> inputs;
    std::string operation;
};


class AutoGrad
{
public:
    AutoGrad();
    void get_params();
    std::vector<float> calculate_gradients();
private:
    std::vector<Params> params;
    float grad_sum(Params parameters);
    float grad_softmax(Params parameters);
    float grad_logarithm(Params parameters);
    float grad_relu(Params parameters);
};

#endif AUTOGRAD_H