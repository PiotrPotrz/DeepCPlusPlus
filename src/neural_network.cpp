#include <vector>

#include "neural_network.h"
#include "nmodule.h"

using namespace std;

vector<float> NeuralNetwork::forward(vector<float>& inputs)
{
    vector<float> out;
    for(int i = 0; i < Network.size(); i++)
    {
        if(i == 0)
        {
            out = Network[0]->forward(inputs);
        }
        else
        {
            // out = Network[i].forward(out);
            out = Network[i]->forward(out);
        }
    }
    return out;
}