#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

#include "loss.h"
#include "utils.h"
#include "activation.h"

using namespace std;


float CrossEntropy::forward(vector<float>& probas, vector<int>& label)
{
    float loss_value = 0.0f;
    int i = 0;
    lb = label;
    pr = probas;
    while(i<label.size() && label[i]!=1)
        i++;

    if(i==label.size())
        {
            throw std::invalid_argument("Label is not in one hot format");
        }
    
    float eps = 1e-9;
    loss_value = -log(max(eps, probas[i]));
    return loss_value;
}

void CrossEntropy::backward()
{
    
    gradients.clear();
    gradients.reserve(pr.size());
    float eps = 1e-9;
    for(int i = 0; i < pr.size(); i++)
    {
        float proba = max(eps, pr[i]);
        gradients.push_back(-lb[i]/pr[i]);
    }
}
