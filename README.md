# DeepC++ - Machine Learning in C++

## Overview
This repository contains my own implementation of several machine learning algorithms including Neural Network.  
Its purpose is purely educational.  
Entire code was written in pure C++17 - no additional libraries required.  
In this sample I attached the popular IRIS Dataset downloaded from [here](https://archive.ics.uci.edu/dataset/53/iris). Dataset Wine comes from [here](https://archive.ics.uci.edu/dataset/109/wine).

### Algorithms implemented
* kNN - K Nearest Neighbors Classifier
* NBayes - Naive Bayes Classifier
* Artificial Neuron
* Layer of neuron net
* Neural Network - a neural network consisting of simple neuron layers

### Functions implemented
* Softmax - a function to calculate probability scores for predictions
* ReLU - a commonly used activation function in neural networks
* Categorical Cross-Entropy - a loss function used in multiclass classification
* Accuracy score - a metric commonly used for benchmarking different AI models
* many others!

### Build and usage

To build the project, run the following commands.

```bash
cd build
```


```bash
cmake .
```


```bash
make 
```

Then you can run the code.


```bash
./iris_ml
```
