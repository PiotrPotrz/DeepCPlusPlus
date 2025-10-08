#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <functional>
#include "dataloader.h"

float accuracy(function<int(IrisSample)> predictor, vector <IrisSample>& test);

#endif