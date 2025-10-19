#include "utils.h"

#include <iostream>
#include <stdexcept>
#include <vector>
#include <functional>
#include "dataloader.h"

using namespace std;

float accuracy(function<int(IrisSample)> predictor, vector <IrisSample>& test)
{
    int correct = 0;
    for(auto t:test)
    {
        if(predictor(t)==t.label)
            correct++;
    }
    float accuracy = static_cast<float>(correct) / test.size();
    return accuracy;
}

vector<int> encode_one_hot(int num_classes, int value)
{
    vector<int> encoded(num_classes);
    encoded[value] = 1;
    return encoded;
}

float dot_prod(const vector<float>& v1, const vector<float>& v2)
{
    float output = 0;
    if(v1.size() == v2.size())
    {
        for(int i = 0; i < v1.size(); i++)
        {
            output+= v1[i] * v2[i];
        }
    }
    else
    {
        throw runtime_error("Could not calculate dot product! Vectors are of different sizes!");
    }
    return output;
}

vector<float> MatrixTimesVector(vector<vector<float>>& m, vector<float>& v)
{
vector<float> out_vector;
out_vector.reserve(m.size());
if(m[0].size()!=v.size())
{
    // cout<<"Incompatibile sizes!!!"<<endl;
    throw std::length_error("Incompatibile sizes!");
}

for(int i = 0; i< m.size(); i++)
{
    out_vector.push_back(dot_prod(m[i], v));
}

return out_vector;

};

int decode_one_hot(vector<int>& one_hot)
{
    int label = 0;
    for(int elem:one_hot)
    {
        if(elem!=1)
        {
            label++;
        }
        else
        {
            break;
        }
    }
    return label;
}

int argmax(vector<float>& preds)
{
    int idx=0;
    float max_val=0.0f;

    for(int i = 0; i < preds.size(); i++)
    {
        if(preds[i]>max_val)
        {
            idx = i;
            max_val = preds[i];
        }
    }
    return idx;
}

float accuracy_metric(vector<int>& y_pred, vector<int>& y_true)
{
    int size_pred = y_pred.size();
    int size_true = y_true.size();
    if(size_pred!=size_true)
    {
        throw runtime_error("Sizes are not equal");
    }
    int correct = 0;

    for(int i = 0; i < size_pred; i++)
    {
        if(y_pred[i]==y_true[i])
        {
            correct++;
        }
    }
    float accuracy = static_cast<float>(correct)/y_pred.size();
    return accuracy;
}
