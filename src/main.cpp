#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <utility>
#include <sstream>
#include <unordered_map>
#include <cmath>
#include <random>
#include <algorithm>
#include <functional>
#include <numbers>

#include "dataloader.h"
#include "knn.h"
#include "nbayes.h"
#include "utils.h"

using namespace std;


int main()
{
    vector<IrisSample> dataset = loadIris();

    auto [train, test] = split_dataset(dataset, 0.8, 42);


    kNN knn(train,20,3);
    cout<<"KNN classifier" << endl;
    cout<<"Prediction score: "<<accuracy([&] (IrisSample s) {return  knn.predict(s);}, test) << endl;

    cout<<"Naive Bayes classifier"<<endl;
    NBayes nb(3);
    nb.train(train);
    cout<<"Prediction score: "<<accuracy([&] (IrisSample s) {return  nb.predict(s);}, test) << endl;

    return 0;
}