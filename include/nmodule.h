#ifndef NMODULE_H
#define NMODULE_H

#include <vector>
#include <stdexcept>

class Module
{
public:
    virtual ~Module() = default;
    std::vector<float> inputs;    
};

class TensorElement : public Module 
{
public:
    TensorElement() = default;
    virtual ~TensorElement() = default;
    virtual std::vector<float> forward(std::vector<float>& x) = 0;
    virtual void backward(std::vector<float>& prev_grad) = 0;
    virtual std::string name() const { return "TensorElement"; }
    
    std::vector<float> weights;
    std::vector<float> gradients;
};

class ScalarElement : public Module
{
public:
    ScalarElement() = default;
    virtual ~ScalarElement() = default;

    virtual float forward(std::vector<float>& x) {
        throw std::logic_error("forward() not implemented for this type.");
    }

    virtual void backward(float prev_grad) {
        throw std::logic_error("backward() not implemented for this type.");
    }
    virtual std::string name() const { return "ScalarElement"; }
    std::vector<float> weights;
    std::vector<float> gradients;
};

#endif