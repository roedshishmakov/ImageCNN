#include "imagenn/network.hpp"

#include <cmath>
#include <iostream>

#include "imagenn/exceptions.hpp"
#include "imagenn/neuron.hpp"
#include "imagenn/rng.hpp"

namespace imagenn {

void NeuralNetwork::add_input_layer(std::size_t size) {
    layers_.push_back(
        std::make_unique<Layer>(NeuronType::Transparent, size, transparent_activation(), false));
}

void NeuralNetwork::add_layer(std::size_t size, const ActivationBase& activation,
                              double random_radius, bool use_bias) {
    if (activation.is_softmax()) {
        layers_.push_back(std::make_unique<LayerSoftmax>(size));
    } else {
        layers_.push_back(std::make_unique<Layer>(NeuronType::Regular, size, activation, use_bias));
    }

    if (layers_.size() > 1) {
        connect_last_two(random_radius);
    }
}

void NeuralNetwork::connect_last_two(double random_radius) {
    Layer& previous = *layers_[layers_.size() - 2];
    Layer& current = *layers_.back();

    for (auto& from : previous.neurons()) {
        for (auto& to : current.neurons()) {
            links_.push_back(
                std::make_unique<Link>(*from, *to, random_uniform(-random_radius, random_radius)));
            Link* link = links_.back().get();
            from->add_link_output(link);
            to->add_link_input(link);
        }
    }
}

void NeuralNetwork::run(const std::vector<double>& input) {
    if (layers_.empty()) {
        throw ValidationError("Network has no layers");
    }
    if (input.size() != layers_.front()->size()) {
        throw ValidationError("Input size does not match the input layer");
    }

    layers_.front()->reset(input);

    for (std::size_t i = 0; i + 1 < layers_.size(); ++i) {
        layers_[i + 1]->reset();
        layers_[i]->calc();
        layers_[i]->send();
    }
    layers_.back()->calc();
}

double NeuralNetwork::train(const std::vector<TrainingExample>& data, double speed, bool verbose,
                            double clip_value, bool use_clip) {
    double loss_total = 0.0;

    for (std::size_t index = 0; index < data.size(); ++index) {
        run(data[index].first);
        const double loss = backward_step(data[index].second, speed, clip_value, use_clip, nullptr);
        loss_total += loss;

        if (verbose) {
            std::cout << "item #" << index << ", loss: " << loss << "\n";
        }
    }

    return loss_total;
}

std::vector<double> NeuralNetwork::input_gradient() const {
    const Layer& input_layer = *layers_.front();
    std::vector<double> gradient(input_layer.size(), 0.0);
    for (std::size_t i = 0; i < input_layer.neurons().size(); ++i) {
        double sum = 0.0;
        for (const Link* link : input_layer.neurons()[i]->links_output()) {
            sum += link->weight() * link->weight_delta_param();
        }
        gradient[i] = sum;
    }
    return gradient;
}

double NeuralNetwork::backward_step(const std::vector<double>& target, double speed,
                                    double clip_value, bool use_clip,
                                    std::vector<double>* input_grad_out) {
    Layer& output_layer = *layers_.back();
    const double loss = output_layer.get_loss(target);

    output_layer.back_propagation_output_l(target, speed);
    for (int i = static_cast<int>(layers_.size()) - 2; i >= 0; --i) {
        layers_[static_cast<std::size_t>(i)]->back_propagation_hidden_l(speed);
    }

    if (input_grad_out != nullptr) {
        *input_grad_out = input_gradient();
    }

    if (use_clip) {
        for (auto& layer : layers_) {
            for (auto& neuron : layer->neurons()) {
                for (Link* link : neuron->links_input()) {
                    if (std::abs(link->weight_delta()) > clip_value) {
                        link->set_weight_delta(link->weight_delta() > 0 ? clip_value : -clip_value);
                    }
                }
            }
        }
    }

    for (auto& layer : layers_) {
        layer->back_propagation_apply();
    }
    return loss;
}

std::vector<std::vector<std::vector<double>>> NeuralNetwork::export_weights() const {
    std::vector<std::vector<std::vector<double>>> result;
    result.reserve(layers_.size());
    for (const auto& layer : layers_) {
        result.push_back(layer->export_weights());
    }
    return result;
}

void NeuralNetwork::import_weights(const std::vector<std::vector<std::vector<double>>>& data) {
    if (data.size() != layers_.size()) {
        throw ValidationError("Layer count does not match the stored model");
    }
    for (std::size_t i = 0; i < layers_.size(); ++i) {
        if (data[i].size() != layers_[i]->size()) {
            throw ValidationError("Layer size does not match the stored model");
        }
    }
    for (std::size_t i = 0; i < layers_.size(); ++i) {
        layers_[i]->import_weights(data[i]);
    }
}

std::vector<double> NeuralNetwork::get_output() const {
    if (layers_.empty()) {
        throw ValidationError("Network has no layers");
    }
    std::vector<double> output;
    const Layer& last = *layers_.back();
    output.reserve(last.size());
    for (const auto& neuron : last.neurons()) {
        output.push_back(neuron->output());
    }
    return output;
}

int NeuralNetwork::get_best_index() const {
    const std::vector<double> output = get_output();
    int best_index = 0;
    double best_value = output[0];
    for (std::size_t i = 1; i < output.size(); ++i) {
        if (output[i] > best_value) {
            best_value = output[i];
            best_index = static_cast<int>(i);
        }
    }
    return best_index;
}

} // namespace imagenn
