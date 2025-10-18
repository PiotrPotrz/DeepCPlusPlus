#ifndef NEURON_H
#define NEURON_H

#include "nmodule.h"
#include <vector>

class Neuron : public ScalarElement
{
public:
    Neuron(int input_size);
    float forward(std::vector<float>& x) override;
    void backward(float prev_grad) override;
    void initialize_weights();

    std::string name() const override { return "Neuron"; }
    std::vector<float> weights;
    std::vector<float> gradients;
    float bias;
private:
    int input_size;
};

class MLPLayer : public TensorElement
{
public:
    MLPLayer(int input_size, int neuron_num);
    std::vector<float> forward(std::vector<float>& x) override;
    void backward(std::vector<float>& prev_grad) override;
    std::vector<Neuron> neurons;
    std::string name() const override { return "MLPLayer"; }

private:
    int input_size;
    int neuron_num;
    void build_layer();
};

#endif