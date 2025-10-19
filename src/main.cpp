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
#include <memory>

#include "dataloader.h"
#include "knn.h"
#include "nbayes.h"
#include "utils.h"
#include "activation.h"
#include "loss.h"
#include "neuron.h"
#include "neural_network.h"
#include "optimizer.h"

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

    
    cout<<"Neural Network \n\n"<<endl;

    NeuralNetwork NN;

    NN.Network.clear();
    NN.Network.push_back(std::make_unique<MLPLayer>(4, 4));
    NN.Network.push_back(std::make_unique<ReLU>());
    NN.Network.push_back(std::make_unique<MLPLayer>(4, 4));
    NN.Network.push_back(std::make_unique<ReLU>());
    NN.Network.push_back(std::make_unique<MLPLayer>(4, 3));
    NN.Network.push_back(std::make_unique<Softmax>());

    float learning_rate = 0.001;
    SGD sgd(NN.Network, learning_rate);

    CrossEntropy CrossEntropyLoss;
    int epochs = 100;
    for(int e = 0; e<epochs; e++)
    {
        float total_loss = 0;
        float total_val_loss = 0;
        for(auto d:train)
        {
            // train
            vector<float> features = d.features;
            vector<int> label = encode_one_hot(3, d.label);

            vector<float> out = NN.forward(features);
            float loss_value = CrossEntropyLoss.forward(out, label);

            total_loss += loss_value;
            CrossEntropyLoss.backward();
            vector<float> loss_grad = CrossEntropyLoss.gradients;
            sgd.step(loss_grad);
        }
        
        vector<int> y_pred;
        vector<int> y_true;
        y_pred.clear();
        y_true.clear();
        for(auto v:test)
        {
            // val
            vector<float> features = v.features;
            vector<int> label = encode_one_hot(3, v.label);

            vector<float> out = NN.forward(features);
            float loss_value = CrossEntropyLoss.forward(out, label);
            
            int amax = argmax(out);
            y_pred.push_back(amax);
            y_true.push_back(decode_one_hot(label));

            total_val_loss += loss_value;
        }
        float acc = accuracy_metric(y_pred, y_true);
    cout<<"["<<e+1<<"/"<<epochs<<"]"<<"Total loss: "<<total_loss<<" || validation loss: "<<total_val_loss<<" || validation accuracy: "<<acc<<endl;
    }


    return 0;
}