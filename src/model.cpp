#include "imagenn/model.hpp"

#include <iostream>

#include "imagenn/exceptions.hpp"
#include "imagenn/rng.hpp"

namespace imagenn {

void Model::add_spatial(std::unique_ptr<SpatialLayer> layer) {
    feature_layers_.push_back(std::move(layer));
}

NeuralNetwork& Model::dense() {
    return dense_;
}

const NeuralNetwork& Model::dense() const {
    return dense_;
}

Tensor Model::forward_features(const Tensor& input) {
    Tensor current = input;
    for (auto& layer : feature_layers_) {
        current = layer->forward(current);
    }
    return current;
}

void Model::run(const Tensor& input) {
    const Tensor features = forward_features(input);
    dense_.run(features.data);
}

double Model::train(const std::vector<SpatialExample>& data, double speed, bool verbose,
                    double clip_value, bool use_clip) {
    double loss_total = 0.0;

    std::vector<std::size_t> order(data.size());
    for (std::size_t i = 0; i < order.size(); ++i) {
        order[i] = i;
    }
    for (std::size_t i = order.size(); i > 1; --i) {
        std::size_t j = static_cast<std::size_t>(random_uniform(0.0, 1.0) * static_cast<double>(i));
        if (j >= i) {
            j = i - 1;
        }
        std::swap(order[i - 1], order[j]);
    }

    for (std::size_t step = 0; step < order.size(); ++step) {
        const std::size_t index = order[step];
        const Tensor features = forward_features(data[index].first);
        dense_.run(features.data);

        std::vector<double> input_grad;
        const double loss =
            dense_.backward_step(data[index].second, speed, clip_value, use_clip, &input_grad);
        loss_total += loss;

        if (verbose) {
            std::cout << "item #" << index << ", loss: " << loss << "\n";
        }

        Tensor grad(1, 1, static_cast<int>(input_grad.size()));
        grad.data = input_grad;
        for (int i = static_cast<int>(feature_layers_.size()) - 1; i >= 0; --i) {
            grad = feature_layers_[static_cast<std::size_t>(i)]->backward(grad);
        }
        for (auto& layer : feature_layers_) {
            layer->apply(speed, clip_value, use_clip);
        }
    }

    return loss_total;
}

std::vector<double> Model::get_output() const {
    return dense_.get_output();
}

int Model::get_best_index() const {
    return dense_.get_best_index();
}

std::vector<std::vector<double>> Model::export_spatial() const {
    std::vector<std::vector<double>> result;
    result.reserve(feature_layers_.size());
    for (const auto& layer : feature_layers_) {
        result.push_back(layer->export_weights());
    }
    return result;
}

void Model::import_spatial(const std::vector<std::vector<double>>& data) {
    if (data.size() != feature_layers_.size()) {
        throw ValidationError("Spatial layer count does not match the stored model");
    }
    for (std::size_t i = 0; i < feature_layers_.size(); ++i) {
        feature_layers_[i]->import_weights(data[i]);
    }
}

} // namespace imagenn
