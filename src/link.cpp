#include "imagenn/link.hpp"

#include "imagenn/neuron.hpp"

namespace imagenn {

void Link::send(double signal) const {
    to_->add(signal * weight_);
}

void Link::back_propagation_output_l(double ref, double speed) {
    const double dl_dy = to_->get_activation(to_->input()) - ref;
    const double dy_dz = to_->get_activation_derivative(to_->input());
    const double dz_dw = from_->output();

    weight_delta_param_ = dl_dy * dy_dz;
    weight_delta_ = -dl_dy * dy_dz * dz_dw * speed;
}

void Link::back_propagation_hidden_l(double speed) {
    double dl_dy = 0.0;
    for (const Link* downstream : to_->links_output()) {
        dl_dy += downstream->weight_delta_param() * downstream->weight();
    }

    const double dy_dz = to_->get_activation_derivative(to_->input());
    const double dz_dw = from_->output();

    weight_delta_param_ = dl_dy * dy_dz;
    weight_delta_ = -dl_dy * dy_dz * dz_dw * speed;
}

void Link::apply_weight_delta() {
    weight_ += weight_delta_;
    weight_delta_ = 0.0;
    weight_delta_param_ = 0.0;
}

double Link::from_output() const {
    return from_->output();
}

} // namespace imagenn
