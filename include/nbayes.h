#ifndef NBAYES_H
#define NBAYES_H

#include <vector>
#include <functional>


#include "dataloader.h"

class NBayes
{
    public:
        NBayes(int classes);
        void train(vector<IrisSample>& train_dataset);
        int predict(IrisSample sample);
    private:
        int classes;
        vector<vector<float>> means;
        vector<vector<float>> stds;
        vector<float> probability(IrisSample sample);
        vector<float> mean(vector<IrisSample>& train_dataset);
        vector<float> std(vector<IrisSample>& train_dataset, vector<float> means);
        vector<int> class_counts;
};

#endif