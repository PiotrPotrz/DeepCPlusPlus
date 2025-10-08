#ifndef DATALOADER_H
#define DATALOADER_H

#include <iostream>
#include <vector>
#include <string>
#include <utility>
using namespace std;


struct IrisSample 
{
    vector<float> features;
    int label;
};

vector<IrisSample> loadIris();

pair<vector<IrisSample>,vector<IrisSample>> split_dataset(vector<IrisSample>& dataset, float train_ratio, int seed);

#endif

