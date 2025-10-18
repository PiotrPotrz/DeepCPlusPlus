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

// class Loss
// {
//     public:
//         Loss() = default;
//         virtual ~Loss() = default;
//         virtual float forward(const std::vector<std::vector<float>>& pred, const std::vector<int>& labels)=0;
// };

// class BinaryCrossEntropy : public Loss
// {
//     public:
//         BinaryCrossEntropy() = default;
//         float forward(const std::vector<std::vector<float>>& pred, const std::vector<int>& labels) override;
// };

// class CrossEntropy : public Loss
// {
//     public:
//         CrossEntropy(bool with_softmax);
//         float forward(const std::vector<std::vector<float>>& pred, const std::vector<int>& labels) override;
//     private:
//         bool with_softmax;
//     };

#endif