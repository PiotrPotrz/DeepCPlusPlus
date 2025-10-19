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

pair<vector<IrisSample>,vector<IrisSample>> split_dataset(vector<IrisSample>& dataset, float train_ratio, int seed, bool min_max)
{
    if (train_ratio <= 0.0f || train_ratio >= 1.0f) 
        throw invalid_argument("train_ratio must be between 0 and 1");
    // random_device rd;
    mt19937 g(seed);

    int split = dataset.size() * train_ratio;
    
    shuffle(dataset.begin(), dataset.end(), g);
    vector<IrisSample> train(dataset.begin(), dataset.begin()+split);
    vector<IrisSample> test(dataset.begin()+split,dataset.end());
    if(min_max)
    {
        train = normalize_dataset(train);
        test = normalize_dataset(test);
    }
    return {train, test};
}


vector<IrisSample> loadWine()
{
    ifstream datafile("../data/wine.data");
    if(!datafile){
        cerr<<"Couldnt load dataset !!!"<<endl;
    }   

    string line;
    vector<IrisSample> dataset;

    while(getline(datafile, line))
    {
        
        if (line.empty()) continue;
        stringstream ss(line);
        string cell;
        IrisSample sample;

        size_t featureCount = std::count(line.begin(), line.end(), ',') + 1;
        for(int i = 0; i< featureCount; i++)
        {
            getline(ss, cell, ',');
            if(i==0)
            {
                sample.label = stoi(cell)-1;
            }
            else
            {
            sample.features.push_back(stof(cell));
            }
        }
        dataset.push_back(sample);
    }

    return dataset;
}



std::vector<std::pair<float, float>> stats(std::vector<IrisSample>& dataset)
{
    // returns vector of {max, min}
    size_t num_features = dataset[0].features.size();
    vector<pair<float, float>> stats(num_features, {-INFINITY, +INFINITY});

    for(auto row:dataset)
    {
        for(int i = 0; i < num_features; i++)
        {
            if(row.features[i]>stats[i].first)
            {
                stats[i].first = row.features[i];
            }
            if(row.features[i]<stats[i].second)
            {
                stats[i].second = row.features[i];
            }

        }

    }
    return stats;
}


float normalize_value(float value, float max_val, float min_val)
{
    return (value - min_val) / (max_val - min_val);
}


std::vector<IrisSample> normalize_dataset(std::vector<IrisSample>& dataset)
{
    size_t range = dataset[0].features.size();
    std::vector<IrisSample> normalized_dataset;
    vector<pair<float, float>> min_max_vals = stats(dataset);
    for(auto row:dataset)
    {
        // vector<float> temp_vec = {};
        IrisSample temp_sample;
        // temp_sample.features = temp_vec;
        for(int i = 0; i < range; i++)
        {
            float val = row.features[i];
            temp_sample.features.push_back(normalize_value(val, min_max_vals[i].first, min_max_vals[i].second));
        }
        temp_sample.label = row.label;

        normalized_dataset.push_back(temp_sample);
    } 
    return normalized_dataset;
}