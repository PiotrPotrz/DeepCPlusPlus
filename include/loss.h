#ifndef LOSS_H
#define LOSS_H

#include <vector>

class Loss
{
    public:
        Loss() = default;
        virtual ~Loss() = default;
        virtual float forward(const std::vector<std::vector<float>>& pred, const std::vector<int>& labels)=0;
};

class BinaryCrossEntropy : public Loss
{
    public:
        BinaryCrossEntropy() = default;
        float forward(const std::vector<std::vector<float>>& pred, const std::vector<int>& labels) override;
};

class CrossEntropy : public Loss
{
    public:
        CrossEntropy() = default;
        float forward(const std::vector<std::vector<float>>& pred, const std::vector<int>& labels) override;
};

#endif