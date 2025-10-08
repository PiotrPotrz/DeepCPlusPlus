#include <iostream>
#include <vector>
#include <utility>
#include <functional>
#include <fstream>
#include <algorithm>
#include <cmath>

#include "dataloader.h"
#include "knn.h"

using namespace std;

struct DistanceSample
{
    float distance;
    int label;  
};


kNN::kNN(const vector<IrisSample>& train_dataset, int neighbours, int classes) 
    : train_dataset(train_dataset), neighbours(neighbours), classes(classes){
}


float kNN::euclidean(IrisSample p1, IrisSample p2)
{
    float distance = 0;
    for(int i =0; i< p1.features.size(); i++)
    {
        distance = distance + (pow(p1.features[i]-p2.features[i],2)); 
    }
    distance = sqrt(distance);
    return distance;
}

int kNN::predict(IrisSample point)
{
    vector<DistanceSample> distances;
    for(auto train_point:train_dataset)
    {
        DistanceSample sample;
        sample.distance = euclidean(point, train_point);
        sample.label = train_point.label;
        distances.push_back(sample);
    }

    sort(distances.begin(), distances.end(), [](const DistanceSample& a, const DistanceSample& b) {
        return a.distance < b.distance;
    });

    vector<int> counts(classes, 0);
    for(int i = 0; i<neighbours; i++)
    {
        for(int j =0; j<classes;j++)
        {
            if(distances[i].label==j)
            {
                counts[j]++;
                continue;
            }
        }
    }

    int predicted_class = std::distance(
    counts.begin(),
    std::max_element(counts.begin(), counts.end())
    );

    return predicted_class;
}