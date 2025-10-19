#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <functional>
#include "dataloader.h"


float accuracy(function<int(IrisSample)> predictor, std::vector <IrisSample>& test);

std::vector<int> encode_one_hot(int num_classes, int value);

int decode_one_hot(std::vector<int>& one_hot);

float dot_prod(const vector<float>& v1, const vector<float>& v2);

template <typename T>
void print_vector(std::vector<T>& vec)
{
    for(T el:vec)
    {
        cout<<el<<", ";
    }
    cout<<endl;
}

std::vector<float> MatrixTimesVector(std::vector<std::vector<float>>& m, std::vector<float>& v);

int argmax(std::vector<float>& preds);

float accuracy_metric(std::vector<int>& y_pred, std::vector<int>& y_true);
#endif