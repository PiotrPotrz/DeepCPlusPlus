#include <iostream>
#include <cmath>


#include "dataloader.h"
#include "nbayes.h"

NBayes::NBayes(int classes)
{
    this->classes=classes;
}

vector<float> NBayes::mean(vector <IrisSample>& train_dataset)
{
    int size = train_dataset[0].features.size();
    vector<float> mean_table(size);
    for(auto sample:train_dataset)
    {
        for(int i=0; i < size; i++)
        {
            mean_table[i]+=sample.features[i];
        }
    }

    for(int i=0; i<size; i++) {mean_table[i]= mean_table[i]/train_dataset.size();}
    return mean_table;
}


vector<float> NBayes::std(vector <IrisSample>& train_dataset, vector<float> means)
{
    int size = train_dataset[0].features.size();
    vector<float> std_table(size);
    int dataset_size = train_dataset.size();
    for(auto sample:train_dataset)
    {
        for(int i=0; i < size; i++)
        {
            std_table[i]+=(sample.features[i]-means[i])*(sample.features[i]-means[i]) / (dataset_size-1);
        }
    }
    for(int i=0; i<size; i++) {std_table[i]= sqrt(std_table[i]);}
    return std_table;
}

void NBayes::train(vector <IrisSample>& train_dataset)
{
    vector<vector<IrisSample>> class_sets(3);
    for(auto sample:train_dataset)
    {
        for(int i = 0; i < classes; i++)
        {
            if(sample.label==i)
            {
                class_sets[i].push_back(sample);
            }
        }
    }
    
    vector<float> temp_mean;
    vector<float> temp_std;

    for(auto set:class_sets)
    {
        temp_mean = mean(set);
        temp_std = std(set, temp_mean); 
        means.push_back(temp_mean);
        stds.push_back(temp_std);  
        class_counts.push_back(set.size());
    }
}

vector<float> NBayes::probability(IrisSample sample)
{
    float sum;
    for(auto c:class_counts) {sum+= static_cast<float>(c);}
    float PI = 3.14159265358979323846;
    vector<float> scores(classes, 1);
    for(int i = 0; i<classes; i++)
    {
        for(int j =0; j<sample.features.size(); j++)
        {
            scores[i] = scores[i] * exp(-0.5*pow(sample.features[j] - means[i][j],2)/pow(stds[i][j],2))/(stds[i][j]);
        }
    scores[i] = scores[i] / (2 * PI) * (static_cast<float>(class_counts[i])/sum);
    } 
    return scores;
}

int NBayes::predict(IrisSample sample)
{
    vector<float> scores = probability(sample);
    int prediction = 0;
    float max_pred = 0;
    for(int i =0; i < classes; i++)
    {
        if(scores[i]>max_pred)
        {
            max_pred = scores[i];
            prediction = i;
        }
    }
    return prediction;
}