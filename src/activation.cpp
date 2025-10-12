#include <iostream>
#include <vector>
#include <cmath>

#include "activation.h"

using namespace std;


vector<float> ReLU::compute_activation(const vector<float>& data)
{
    vector<float> output;
    output.reserve(data.size());

    for(auto d:data)
    {
        output.push_back(max(0.0f,d));
    }
    return output;
}



float Softmax::sigmoid(float data_point)
{
    return 1/(1 + exp(-data_point));
}

vector<float> Softmax::compute_activation(const vector<float>& data)
{
    vector<float> output;
    output.reserve(data.size());
    float sum = 0.0f;
    for(auto d:data){
        sum+=exp(d);
    }
    for(auto d:data)
    {
        float proba = exp(d);
        output.push_back(proba/sum);
    }
    return output;
}

