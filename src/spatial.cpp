#include "imagenn/spatial.hpp"

#include "imagenn/exceptions.hpp"
#include "imagenn/rng.hpp"

namespace imagenn {

void SpatialLayer::apply(double, double, bool) {}

std::vector<double> SpatialLayer::export_weights() const {
    return std::vector<double>();
}

void SpatialLayer::import_weights(const std::vector<double>&) {}

namespace {
void check_input(const Tensor& input, int c, int h, int w) {
    if (input.channels != c || input.height != h || input.width != w) {
        throw ValidationError("Tensor shape does not match the layer input");
    }
}

double clip(double value, double limit, bool use_clip) {
    if (use_clip && value > limit) {
        return limit;
    }
    if (use_clip && value < -limit) {
        return -limit;
    }
    return value;
}
} // namespace

ConvLayer::ConvLayer(int in_channels, int in_height, int in_width, int filters, int kernel,
                     const ActivationBase& activation, double random_radius)
    : kernel_(kernel), act_(&activation) {
    if (in_channels <= 0 || in_height <= 0 || in_width <= 0 || filters <= 0 || kernel <= 0) {
        throw ValidationError("Convolution parameters must be positive");
    }
    if (kernel > in_height || kernel > in_width) {
        throw ValidationError("Convolution kernel is larger than the input");
    }

    in_c_ = in_channels;
    in_h_ = in_height;
    in_w_ = in_width;
    out_c_ = filters;
    out_h_ = in_height - kernel + 1;
    out_w_ = in_width - kernel + 1;

    const std::size_t weight_count =
        static_cast<std::size_t>(filters) * in_channels * kernel * kernel;
    weights_.resize(weight_count);
    for (std::size_t i = 0; i < weight_count; ++i) {
        weights_[i] = random_uniform(-random_radius, random_radius);
    }
    biases_.assign(static_cast<std::size_t>(filters), 0.0);
    weight_grads_.assign(weight_count, 0.0);
    bias_grads_.assign(static_cast<std::size_t>(filters), 0.0);
}

std::size_t ConvLayer::weight_index(int f, int c, int ky, int kx) const {
    return (((static_cast<std::size_t>(f) * in_c_ + c) * kernel_) + ky) * kernel_ + kx;
}

Tensor ConvLayer::forward(const Tensor& input) {
    check_input(input, in_c_, in_h_, in_w_);
    last_input_ = input;

    Tensor output(out_c_, out_h_, out_w_);
    last_z_.assign(static_cast<std::size_t>(out_c_) * out_h_ * out_w_, 0.0);

    for (int f = 0; f < out_c_; ++f) {
        for (int oy = 0; oy < out_h_; ++oy) {
            for (int ox = 0; ox < out_w_; ++ox) {
                double sum = biases_[static_cast<std::size_t>(f)];
                for (int c = 0; c < in_c_; ++c) {
                    for (int ky = 0; ky < kernel_; ++ky) {
                        for (int kx = 0; kx < kernel_; ++kx) {
                            sum += input.at(c, oy + ky, ox + kx) *
                                   weights_[weight_index(f, c, ky, kx)];
                        }
                    }
                }
                last_z_[output.index(f, oy, ox)] = sum;
                output.at(f, oy, ox) = act_->calc(sum);
            }
        }
    }
    return output;
}

Tensor ConvLayer::backward(const Tensor& grad_output) {
    Tensor grad_input(in_c_, in_h_, in_w_);
    for (std::size_t i = 0; i < weight_grads_.size(); ++i) {
        weight_grads_[i] = 0.0;
    }
    for (std::size_t i = 0; i < bias_grads_.size(); ++i) {
        bias_grads_[i] = 0.0;
    }

    for (int f = 0; f < out_c_; ++f) {
        for (int oy = 0; oy < out_h_; ++oy) {
            for (int ox = 0; ox < out_w_; ++ox) {
                const std::size_t out_idx = grad_output.index(f, oy, ox);
                const double dz = grad_output.data[out_idx] * act_->derivative(last_z_[out_idx]);
                bias_grads_[static_cast<std::size_t>(f)] += dz;
                for (int c = 0; c < in_c_; ++c) {
                    for (int ky = 0; ky < kernel_; ++ky) {
                        for (int kx = 0; kx < kernel_; ++kx) {
                            const std::size_t w_idx = weight_index(f, c, ky, kx);
                            weight_grads_[w_idx] += dz * last_input_.at(c, oy + ky, ox + kx);
                            grad_input.at(c, oy + ky, ox + kx) += dz * weights_[w_idx];
                        }
                    }
                }
            }
        }
    }
    return grad_input;
}

void ConvLayer::apply(double speed, double clip_value, bool use_clip) {
    for (std::size_t i = 0; i < weights_.size(); ++i) {
        weights_[i] -= clip(speed * weight_grads_[i], clip_value, use_clip);
    }
    for (std::size_t i = 0; i < biases_.size(); ++i) {
        biases_[i] -= clip(speed * bias_grads_[i], clip_value, use_clip);
    }
}

void ConvLayer::load(const std::vector<double>& weights, const std::vector<double>& biases) {
    if (weights.size() != weights_.size() || biases.size() != biases_.size()) {
        throw ValidationError("Convolution weight sizes do not match the layer");
    }
    weights_ = weights;
    biases_ = biases;
}

std::vector<double> ConvLayer::export_weights() const {
    std::vector<double> result = weights_;
    result.insert(result.end(), biases_.begin(), biases_.end());
    return result;
}

void ConvLayer::import_weights(const std::vector<double>& weights) {
    if (weights.size() != weights_.size() + biases_.size()) {
        throw ValidationError("Convolution parameter size does not match the layer");
    }
    for (std::size_t i = 0; i < weights_.size(); ++i) {
        weights_[i] = weights[i];
    }
    for (std::size_t i = 0; i < biases_.size(); ++i) {
        biases_[i] = weights[weights_.size() + i];
    }
}

MaxPoolLayer::MaxPoolLayer(int in_channels, int in_height, int in_width, int pool) : pool_(pool) {
    if (in_channels <= 0 || in_height <= 0 || in_width <= 0 || pool <= 0) {
        throw ValidationError("Pooling parameters must be positive");
    }
    in_c_ = in_channels;
    in_h_ = in_height;
    in_w_ = in_width;
    out_c_ = in_channels;
    out_h_ = in_height / pool;
    out_w_ = in_width / pool;
    if (out_h_ <= 0 || out_w_ <= 0) {
        throw ValidationError("Pooling window is larger than the input");
    }
}

Tensor MaxPoolLayer::forward(const Tensor& input) {
    check_input(input, in_c_, in_h_, in_w_);

    Tensor output(out_c_, out_h_, out_w_);
    argmax_.assign(static_cast<std::size_t>(out_c_) * out_h_ * out_w_, 0);

    for (int c = 0; c < out_c_; ++c) {
        for (int oy = 0; oy < out_h_; ++oy) {
            for (int ox = 0; ox < out_w_; ++ox) {
                int best_y = oy * pool_;
                int best_x = ox * pool_;
                double best = input.at(c, best_y, best_x);
                for (int wy = 0; wy < pool_; ++wy) {
                    for (int wx = 0; wx < pool_; ++wx) {
                        const double value = input.at(c, oy * pool_ + wy, ox * pool_ + wx);
                        if (value > best) {
                            best = value;
                            best_y = oy * pool_ + wy;
                            best_x = ox * pool_ + wx;
                        }
                    }
                }
                output.at(c, oy, ox) = best;
                argmax_[output.index(c, oy, ox)] = input.index(c, best_y, best_x);
            }
        }
    }
    return output;
}

Tensor MaxPoolLayer::backward(const Tensor& grad_output) {
    Tensor grad_input(in_c_, in_h_, in_w_);
    for (std::size_t i = 0; i < argmax_.size(); ++i) {
        grad_input.data[argmax_[i]] += grad_output.data[i];
    }
    return grad_input;
}

FlattenLayer::FlattenLayer(int in_channels, int in_height, int in_width) {
    in_c_ = in_channels;
    in_h_ = in_height;
    in_w_ = in_width;
    out_c_ = 1;
    out_h_ = 1;
    out_w_ = in_channels * in_height * in_width;
}

Tensor FlattenLayer::forward(const Tensor& input) {
    check_input(input, in_c_, in_h_, in_w_);
    Tensor output(out_c_, out_h_, out_w_);
    output.data = input.data;
    return output;
}

Tensor FlattenLayer::backward(const Tensor& grad_output) {
    Tensor grad_input(in_c_, in_h_, in_w_);
    grad_input.data = grad_output.data;
    return grad_input;
}

} // namespace imagenn
