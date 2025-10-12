#include "utils.h"

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


