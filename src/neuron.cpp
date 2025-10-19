#include <iostream>
#include <vector>
#include <random>

#include "neuron.h"
#include "utils.h"


using namespace std;

Neuron::Neuron(int input_size) : input_size(input_size)
{
    initialize_weights();
};


void Neuron::initialize_weights()
{
    weights = {};
    random_device rd;
    mt19937 gen(rd());

    uniform_real_distribution<float> dist(0.0f, 1.0f);

    for(int i = 0; i< input_size; i++)
    {
        weights.push_back(dist(gen));
    }
    bias = dist(gen);
}


float Neuron::forward(vector<float>& data)
{
    inputs = data;
    float out = dot_prod(data, weights);
    return out + bias;
}

void Neuron::backward(float prev_grad)
{
    gradients.clear();
    gradients.reserve(weights.size() + 1);;
    
    for(int i = 0; i<weights.size(); i++)
    {
        gradients.push_back(inputs[i] * prev_grad);
    }
    // gradient for bias 
    gradients.push_back(prev_grad);
}

MLPLayer :: MLPLayer(int input_size, int neuron_num) : input_size(input_size), neuron_num(neuron_num)
{
    build_layer();
};

void MLPLayer::build_layer()
{
    neurons.clear();
    neurons.reserve(neuron_num);
    for(int i = 0; i< neuron_num; i++)
    {
        neurons.push_back(Neuron(input_size));
    }
}

vector<float> MLPLayer::forward(std::vector<float>& x)
{
    vector<float> outputs;
    for(Neuron& n:neurons)
    {
        outputs.push_back(n.forward(x));
    }
    return outputs;
}

void MLPLayer::backward(std::vector<float>& prev_grad)
{
    for(int i = 0; i < neurons.size(); i++)
    {
        neurons[i].backward(prev_grad[i]);
    }
}
