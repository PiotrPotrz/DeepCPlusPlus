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

// CrossEntropy::CrossEntropy(bool with_softmax) : with_softmax(with_softmax)
// {

// };


// float CrossEntropy::forward(const std::vector<std::vector<float>>& pred, const std::vector<int>& labels)
// {
//     float loss_value = 0.0f;
//     Softmax softmax;
//     vector<vector<int>> one_hot_labels;
//     int num_classes = pred[0].size();
//     for(auto lbl:labels)
//     {
//         vector<int> temp_oh_lbl = encode_one_hot(num_classes, lbl);
//         one_hot_labels.push_back(temp_oh_lbl);
//     }

//     for(int i = 0; i<pred.size(); i++)
//     {
//         vector<float> probas;
//         probas = softmax.compute_activation(pred[i]);
//         for(int j = 0; j<num_classes; j++)
//         {
//             loss_value +=  -one_hot_labels[i][j] * log(std::max(probas[j], 1e-9f));
//         }
//     }
//     return loss_value / static_cast<float>(pred.size());
// }

