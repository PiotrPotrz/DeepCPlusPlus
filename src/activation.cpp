#include <iostream>
#include <vector>
#include <cmath>

#include "activation.h"
#include "utils.h"

using namespace std;


vector<float> ReLU::forward(std::vector<float>& x)
{
    inputs = x;
    vector<float> output;
    output.reserve(x.size());

    for(auto d:x)
    {
        output.push_back(max(0.0f,d));
    }
    return output;
}

void ReLU::backward(std::vector<float>& prev_grad)
{
    gradients.clear();
    gradients.reserve(inputs.size());
    
    for(int i = 0; i<inputs.size(); i++)
    {
        if(inputs[i]>0)
        {
            gradients.push_back(prev_grad[i]);
        }
        else
        {
            gradients.push_back(0);
        }
    }
};

vector<float> Softmax::forward(std::vector<float>& x)
{
    inputs = x;
    vector<float> output;
    output.reserve(x.size());
    float sum = 0.0f;
    for(auto d:x){
        sum+=exp(d);
    }
    for(auto d:x)
    {
        float proba = exp(d);
        output.push_back(proba/sum);
    }

    outputs = output;
    return output;
}

void Softmax::backward(vector<float>& prev_grad)
{
    gradients.clear();
    gradients.reserve(inputs.size());


    std::vector<std::vector<float>> jacobian(inputs.size(), std::vector<float>(inputs.size(), 0.0f));

    for(int i = 0; i < inputs.size(); i++)
    {
        for(int j = 0; j < inputs.size(); j++)
        {
            if(i==j)
            {
                jacobian[i][j] = outputs[i] * (1 - outputs[i]);
            }
            else
            {
                jacobian[i][j] = - outputs[i] * outputs[j];
            }
        }    
    }
    gradients = MatrixTimesVector(jacobian, prev_grad);
}
