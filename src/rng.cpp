#include "imagenn/rng.hpp"

#include <random>

namespace imagenn {

namespace {
/// Состояние генератора с фиксированным зерном по умолчанию ради
/// воспроизводимости запусков на разных машинах.
std::mt19937& engine() {
    static std::mt19937 instance(42);
    return instance;
}
} // namespace

void set_random_seed(unsigned int seed) {
    engine().seed(seed);
}

double random_uniform(double lo, double hi) {
    std::uniform_real_distribution<double> dist(lo, hi);
    return dist(engine());
}

} // namespace imagenn
