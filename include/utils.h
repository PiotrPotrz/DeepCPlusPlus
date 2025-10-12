#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <functional>
#include "dataloader.h"

float accuracy(function<int(IrisSample)> predictor, std::vector <IrisSample>& test);
std::vector<int> encode_one_hot(int num_classes, int value);
template <typename T>
void print_vector(std::vector<T>& vec)
{
    for(T el:vec)
    {
        cout<<el<<", ";
    }
    cout<<endl;
}
#endif