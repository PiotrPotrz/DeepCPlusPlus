#include "utils.h"

#include <vector>
#include <functional>
#include "dataloader.h"

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
