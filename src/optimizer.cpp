#include <vector>
#include <memory>
#include <string>
#include <iostream>

#include "optimizer.h"
#include "neural_network.h"
#include "neuron.h"
#include "nmodule.h"
#include "loss.h"

using namespace std;

SGD::SGD(std::vector<std::unique_ptr<TensorElement>>& Network, float learning_rate)
    : Network(Network),
    learning_rate(learning_rate)
{};

float SGD::modify_weight(float weight, float gradient)
{
    float new_weight = weight - learning_rate * gradient;
    return new_weight; 
}

void SGD::step(vector<float>& loss_grad)
{
    vector<float> gradients = loss_grad;

    for(int i = Network.size()-1; i>=0; i--)
    {
        string name = Network[i]->name();

        Network[i]->backward(gradients);
        gradients = Network[i]->gradients;
        if(name=="MLPLayer")
        {
            MLPLayer* dense = dynamic_cast<MLPLayer*>(Network[i].get());   
            for(Neuron& n: dense->neurons)
            {
                for(int k = 0; k< n.weights.size(); k++)
                {
                    n.weights[k] = modify_weight(n.weights[k], n.gradients[k]);
                }
                n.bias = modify_weight(n.bias, n.gradients[n.weights.size()-1]);
            }
        }
    }

}