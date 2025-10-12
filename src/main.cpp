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
#include "activation.h"
#include "loss.h"

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

    cout<<"Some sample data for activation"<<endl;
    vector<float> data = {-0.9, 2.0, 0.1, 0};
    cout<<"ReLU"<<endl;
    ReLU relu;
    // cout << relu.compute_activation(data)<<endl;
    vector<float> relu_results = relu.compute_activation(data);
    for(auto r:relu_results)
    {
        cout<<r<<", ";
    }
    cout<<endl;

    Softmax softmax;
    vector<float> softmax_results = softmax.compute_activation(data);
        for(auto s:softmax_results)
    {
        cout<<s<<", ";
    }
    cout<<endl;

    cout<<"Test one-hot-encodding. Number of classes 4, value 3"<<endl;
    vector<int> encoded = encode_one_hot(4, 3);
    print_vector(encoded);

    CrossEntropy ce;
    float loss_value = ce.forward({{-1.5, 1.2, 0.3, 1.9}},encoded);
    cout<<"Loss value: "<<loss_value<<endl; 

    return 0;
}