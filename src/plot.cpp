#include "imagenn/plot.hpp"

#include <iomanip>
#include <ostream>
#include <string>

#include "imagenn/exceptions.hpp"

namespace imagenn {

namespace {
constexpr int kBarWidth = 50;
} // namespace

void show_loss_ascii(const std::vector<double>& losses, std::ostream& out) {
    if (losses.empty()) {
        throw ValidationError("No loss values to display");
    }

    double max_loss = losses[0];
    for (std::size_t i = 1; i < losses.size(); ++i) {
        if (losses[i] > max_loss) {
            max_loss = losses[i];
        }
    }
    const double scale = max_loss > 0.0 ? max_loss : 1.0;

    out << "Total loss\n";
    for (std::size_t epoch = 0; epoch < losses.size(); ++epoch) {
        const int length = static_cast<int>(losses[epoch] / scale * kBarWidth);
        out << std::setw(4) << epoch << " " << std::setw(10) << std::fixed << std::setprecision(4)
            << losses[epoch] << " |" << std::string(length, '#') << "\n";
    }
}

} // namespace imagenn
