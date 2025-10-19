#ifndef DATALOADER_H
#define DATALOADER_H

#include <iostream>
#include <vector>
#include <string>
#include <utility>
#include "dataloader.h"
using namespace std;


struct IrisSample 
{
    std::vector<float> features;
    int label;
};

std::vector<IrisSample> loadIris();

std::vector<IrisSample> loadWine();

std::pair<std::vector<IrisSample>,std::vector<IrisSample>> split_dataset(std::vector<IrisSample>& dataset, float train_ratio, int seed, bool min_max);


std::vector<std::pair<float, float>> stats(std::vector<std::vector<float>>& dataset);

float normalize_value(float value, float max_val, float min_val);

std::vector<std::pair<float, float>> stats(std::vector<IrisSample>& dataset);

std::vector<IrisSample> normalize_dataset(std::vector<IrisSample>& dataset);

#endif

