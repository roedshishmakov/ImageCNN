#include "imagenn/neuron.hpp"

#include "imagenn/exceptions.hpp"
#include "imagenn/link.hpp"

namespace imagenn {

void Neuron::send() const {
    for (Link* link : link_output_) {
        link->send(output_);
    }
}

void Neuron::back_propagation_output_l(double ref, double speed) {
    for (Link* link : link_input_) {
        link->back_propagation_output_l(ref, speed);
    }
}

void Neuron::back_propagation_hidden_l(double speed) {
    for (Link* link : link_input_) {
        link->back_propagation_hidden_l(speed);
    }
}

void Neuron::back_propagation_apply() {
    for (Link* link : link_output_) {
        link->apply_weight_delta();
    }
}

std::vector<double> Neuron::export_weights() const {
    std::vector<double> result;
    result.reserve(link_input_.size());
    for (Link* link : link_input_) {
        result.push_back(link->export_weight());
    }
    return result;
}

void Neuron::import_weights(const std::vector<double>& data) {
    if (data.size() != link_input_.size()) {
        throw ValidationError("Weight count does not match the number of input links");
    }
    for (std::size_t i = 0; i < link_input_.size(); ++i) {
        link_input_[i]->import_weight(data[i]);
    }
}

} // namespace imagenn
