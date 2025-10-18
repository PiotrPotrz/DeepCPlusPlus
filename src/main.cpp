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

    cout<<"Some sample data for activation"<<endl;
    vector<float> data = {-0.9, 2.0, 0.1, 0};
    cout<<"ReLU"<<endl;
    ReLU relu;

    vector<float> relu_results = relu.forward(data);
    print_vector(relu_results);

    Softmax softmax;
    vector<float> softmax_results = softmax.forward(data);
    print_vector(softmax_results);

    cout<<"Test one-hot-encodding. Number of classes 4, value 3"<<endl;
    vector<int> encoded = encode_one_hot(4, 3);
    print_vector(encoded);

    CrossEntropy ce;
    vector<float> vals = {-1.5, 1.2, 0.3, 1.9};
    float loss_value = ce.forward(vals, encoded);
    cout<<"Loss value: "<<loss_value<<endl; 

    vector<float> values = {1.0, 0.5, 1.1, -1.0, 2.2};
    Neuron neuron(5);
    cout<<"Neuron forward:  "<<neuron.forward(values)<<endl;
    cout<<"Neuron weights: ";
    print_vector(neuron.weights);

    cout<<"Neuron bias: "<<neuron.bias<<endl;

    cout<<"Preforming backward"<<endl;
    neuron.backward(1.0);
    cout<<"gradients: "<<endl;
    print_vector(neuron.gradients);

    MLPLayer layer(5,5);
    vector<float> outvals = layer.forward(values);
    cout<<"MLP layer forward";
    print_vector(outvals);

    // Neural Network
    cout<<"Neural Network \n\n"<<endl;

    NeuralNetwork NN;

    NN.Network.clear();
    NN.Network.push_back(std::make_unique<MLPLayer>(4, 4));
    NN.Network.push_back(std::make_unique<ReLU>());
    NN.Network.push_back(std::make_unique<MLPLayer>(4, 4));
    NN.Network.push_back(std::make_unique<Softmax>());

    float learning_rate = 0.001;
    SGD sgd(NN.Network, learning_rate);

    // cout<<"Neural Network forward: "<<endl;
    // vector<float> neural_net_out = NN.forward(values);
    // print_vector(neural_net_out);


    // cout<<"Neural Network forward again: "<<endl;
    // vector<float> neural_net_out2 = NN.forward(values);
    // print_vector(neural_net_out2);

    CrossEntropy CrossEntropyLoss;
    int epochs = 100;
    for(int e = 0; e<epochs; e++)
    {
        float total_loss = 0;
        for(auto d:train)
        {
            vector<float> features = d.features;
            vector<int> label = encode_one_hot(3, d.label);

            vector<float> out = NN.forward(features);
            float loss_value = CrossEntropyLoss.forward(out, label);
            // cout<<"Value of loss function: "<< loss_value<<endl;
            total_loss += loss_value;
            CrossEntropyLoss.backward();
            vector<float> loss_grad = CrossEntropyLoss.gradients;
            sgd.step(loss_grad);
        }
    cout<<"["<<e+1<<"/"<<epochs<<"]"<<"Total loss: "<<total_loss<<endl;
    }


    return 0;
}