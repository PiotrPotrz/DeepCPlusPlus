// #include <iostream>
// #include <vector>
// #include <random>

// #include "logreg.h"
// #include "dataloader.h"
// #include "activation.h"

// using namespace std;

// LogisticRegression::LogisticRegression(int epochs) 
// {
//     this->epochs=epochs;
// };

// float LogisticRegression::forward_pass(vector<float> weights, IrisSample data)
// {
//     float out_value = 0;
//     for(int i = 0; i< weights.size(); i++)
//     {
//         out_value+=weights[i]*data.features[i];
//     }
//     return out_value;
// }

// void LogisticRegression::train(vector<IrisSample>& data)
// {
//     // inicjalizacja wag
//     Softmax softmax();
//     weights = {};
//     random_device rd;
//     mt19937 gen(rd());

//     uniform_real_distribution<float> dist(0.0f, 1.0f);

//     for(int i = 0; i< data[0].features.size(); i++)
//     {
//         weights.push_back(dist(gen));
//     }



// }