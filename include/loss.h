#ifndef LOSS_H
#define LOSS_H

#include "nmodule.h"
#include <vector>


class CrossEntropy : public ScalarElement
{
public:
    CrossEntropy() = default;
    float forward(std::vector<float>& probas, std::vector<int>& label);
    void backward();
    std::string name() const override { return "CrossEntropy"; }
private:
    std::vector<float> pr;
    std::vector<int> lb;
};

#endif