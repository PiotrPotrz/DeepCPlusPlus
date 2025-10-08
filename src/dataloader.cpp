#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <sstream>
#include <algorithm>
#include <random>
#include "dataloader.h"

using namespace std;

vector<IrisSample> loadIris()
{
    ifstream datafile("../data/iris.data");
    if(!datafile){
        cerr<<"Couldnt load dataset !!!"<<endl;
    }

    unordered_map<string, int> labelMap = 
    {
        {"Iris-setosa",0},
        {"Iris-versicolor",1},
        {"Iris-virginica",2}
    };

    string line;
    vector<IrisSample> dataset;

    while(getline(datafile, line))
    {
        if (line.empty()) continue;
        stringstream ss(line);
        string cell;
        IrisSample sample;

        for(int i = 0; i< 4; i++)
        {
            getline(ss, cell, ',');
            // cout<<cell<<endl;
            sample.features.push_back(stof(cell));
        }
        getline(ss,cell,',');
        sample.label = labelMap[cell];

        dataset.push_back(sample);
    }

    return dataset;
}

pair<vector<IrisSample>,vector<IrisSample>> split_dataset(vector<IrisSample>& dataset, float train_ratio, int seed)
{
    if (train_ratio <= 0.0f || train_ratio >= 1.0f) 
        throw invalid_argument("train_ratio must be between 0 and 1");
    // random_device rd;
    mt19937 g(seed);

    int split = dataset.size() * train_ratio;
    
    shuffle(dataset.begin(), dataset.end(), g);
    vector<IrisSample> train(dataset.begin(), dataset.begin()+split);
    vector<IrisSample> test(dataset.begin()+split,dataset.end());
    return {train, test};
}