#include "imagenn/layer.hpp"

#include <algorithm>
#include <cmath>

#include "imagenn/exceptions.hpp"

namespace imagenn {

namespace {
std::unique_ptr<Neuron> make_neuron(NeuronType type, const ActivationBase& activation) {
    switch (type) {
    case NeuronType::Transparent:
        return std::make_unique<TransparentNeuron>(activation);
    case NeuronType::Softmax:
        return std::make_unique<NeuronSoftmax>(activation);
    case NeuronType::Regular:
    default:
        return std::make_unique<Neuron>(activation);
    }
}

void require_size(const std::vector<double>& values, std::size_t expected) {
    if (values.size() != expected) {
        throw ValidationError("Vector size does not match the layer size");
    }
}
} // namespace

Layer::Layer(NeuronType type, std::size_t size, const ActivationBase& activation, bool use_bias) {
    neurons_.reserve(size);
    for (std::size_t i = 0; i < size; ++i) {
        neurons_.push_back(make_neuron(type, activation));
    }

    if (use_bias) {
        bias_ = std::make_unique<BiasNeuron>(transparent_activation());
        for (auto& neuron : neurons_) {
            links_.push_back(std::make_unique<Link>(*bias_, *neuron, 0.0));
            Link* link = links_.back().get();
            bias_->add_link_output(link);
            neuron->add_link_input(link);
        }
    }
}

void Layer::reset() {
    for (auto& neuron : neurons_) {
        neuron->reset();
    }
}

void Layer::reset(const std::vector<double>& input) {
    require_size(input, neurons_.size());
    for (std::size_t i = 0; i < neurons_.size(); ++i) {
        neurons_[i]->reset(input[i]);
    }
}

void Layer::calc() {
    if (bias_) {
        bias_->send();
    }
    for (auto& neuron : neurons_) {
        neuron->activation();
    }
}

double Layer::get_loss(const std::vector<double>& refs) const {
    require_size(refs, neurons_.size());
    double loss = 0.0;
    for (std::size_t i = 0; i < neurons_.size(); ++i) {
        loss += Neuron::get_loss(neurons_[i]->output(), refs[i]);
    }
    return loss;
}

void Layer::send() const {
    for (const auto& neuron : neurons_) {
        neuron->send();
    }
}

void Layer::back_propagation_output_l(const std::vector<double>& refs, double speed) {
    require_size(refs, neurons_.size());
    for (std::size_t i = 0; i < neurons_.size(); ++i) {
        neurons_[i]->back_propagation_output_l(refs[i], speed);
    }
}

void Layer::back_propagation_hidden_l(double speed) {
    for (auto& neuron : neurons_) {
        neuron->back_propagation_hidden_l(speed);
    }
}

void Layer::back_propagation_apply() {
    for (auto& neuron : neurons_) {
        neuron->back_propagation_apply();
    }
    if (bias_) {
        bias_->back_propagation_apply();
    }
}

std::vector<std::vector<double>> Layer::export_weights() const {
    std::vector<std::vector<double>> result;
    result.reserve(neurons_.size());
    for (const auto& neuron : neurons_) {
        result.push_back(neuron->export_weights());
    }
    return result;
}

void Layer::import_weights(const std::vector<std::vector<double>>& data) {
    if (data.size() != neurons_.size()) {
        throw ValidationError("Neuron count does not match the stored layer");
    }
    for (std::size_t i = 0; i < neurons_.size(); ++i) {
        neurons_[i]->import_weights(data[i]);
    }
}

LayerSoftmax::LayerSoftmax(std::size_t size)
    : Layer(NeuronType::Softmax, size, softmax_activation(), false) {}

void LayerSoftmax::calc() {
    for (auto& neuron : neurons_) {
        neuron->activation();
    }

    std::vector<double> logits;
    logits.reserve(neurons_.size());
    for (const auto& neuron : neurons_) {
        logits.push_back(neuron->output());
    }

    const std::vector<double> probabilities = ActivationSoftmax::calc_layer(logits);
    for (std::size_t i = 0; i < neurons_.size(); ++i) {
        neurons_[i]->set_output(probabilities[i]);
    }
}

double LayerSoftmax::get_loss(const std::vector<double>& refs) const {
    require_size(refs, neurons_.size());
    const double epsilon = 1e-15;
    double loss = 0.0;
    for (std::size_t i = 0; i < neurons_.size(); ++i) {
        const double clipped = std::max(std::min(neurons_[i]->output(), 1.0 - epsilon), epsilon);
        loss += -refs[i] * std::log(clipped);
    }
    return loss;
}

void LayerSoftmax::back_propagation_output_l(const std::vector<double>& refs, double speed) {
    require_size(refs, neurons_.size());
    for (std::size_t i = 0; i < neurons_.size(); ++i) {
        const double grad = neurons_[i]->output() - refs[i];
        for (Link* link : neurons_[i]->links_input()) {
            link->set_weight_delta(-grad * link->from_output() * speed);
            link->set_weight_delta_param(grad);
        }
    }
}

} // namespace imagenn
