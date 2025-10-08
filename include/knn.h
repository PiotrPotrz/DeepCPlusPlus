#ifndef KNN_H
#define KNN_H

#include <vector>
#include "dataloader.h"

class kNN
{
    public:
        kNN(const vector<IrisSample>& train_dataset,int neighbours, int classes);
        int predict(IrisSample sample);
        float euclidean(IrisSample p1, IrisSample p2);
    
    private:
        int neighbours;
        const vector<IrisSample>& train_dataset;
        int classes;
};

#endif